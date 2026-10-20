"""Hash-safe, bounded sequential ingestion jobs; never trains or resumes research."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch

from src.backbone.export import dataset_relation
from src.backbone.external import ExternalObservationSource
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import EvaluationReference, SoccerTrackSource, load_source
from src.dynamic_football_data import FIT_MATCHES
from src.trajectory_dataset import atomic_replace, file_digest, write_json

ROOT = Path(__file__).resolve().parents[2]
VERSION = "backbone.batch.0.2"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def implementation_hashes(root=ROOT):
    files = [path for path in (root / "src" / "backbone").glob("*") if path.is_file()]
    files.extend(root / name for name in (
        "src/belief_observation.py", "src/player_centric_visibility.py", "src/player_centric_dataset.py",
        "src/hidden_defender_geometry.py", "src/models/dynamic_football_gnn.py",
        "src/models/belief_world_model.py", "src/models/candidate_ranker_receiver.py",
        "src/receiver_forecast_adapter.py", "src/trajectory_dataset.py", "src/dynamic_football_data.py"))
    return {str(path.relative_to(root)): file_digest(path) for path in files}


def model_dependencies(experiment: Path, backend: str, exo: bool):
    manifest_path = experiment / "integration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = {"G1": "world_baseline", "G3": "world_gnn_alternative", "G4": "belief_gnn_alternative"}
    if backend not in (*names, "last_seen", "constant_velocity"):
        raise ValueError("Unsupported backend")
    keys = ([names[backend]] if backend in names else []) + (["receiver_baseline"] if exo else [])
    result = {"integration_manifest": file_digest(manifest_path)}
    for key in keys:
        checkpoint = experiment / manifest["selected_checkpoints"][key]["path"]
        for path in (checkpoint, checkpoint.parent / "config.json", checkpoint.parent / "result.json"):
            result[str(path)] = file_digest(path)
    return result


def source_dependencies(root: Path, job: dict):
    adapter = job["source_adapter"]
    if adapter == "external_jsonl":
        path = root / job["input_path"]
        source = ExternalObservationSource(path)
        if str(job["match"]) != source.match or job["half"] != source.half:
            raise ValueError("Configured match/half must match external metadata; no relabeling training data")
        return {str(path.resolve()): file_digest(path)}
    if adapter == "soccertrack_cache":
        match, half = str(job["match"]), job["half"]
        path = root / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{match}_{half}.npz"
        association = root / "outputs" / "trajectory_audit_20261016" / f"associations_{match}_{half}.json"
        return {str(p.resolve()): file_digest(p) for p in (path, path.with_suffix(".json"), association)}
    raise ValueError(f"Unsupported source adapter: {adapter}")


def validate_job(job: dict):
    if job["intended_use"] not in ("training_export", "retrospective_replay"):
        raise ValueError("intended_use must be training_export or retrospective_replay")
    if job["intended_use"] == "training_export" and str(job["match"]) not in FIT_MATCHES:
        raise ValueError("Training export is restricted to the six permitted fitting matches")
    if not isinstance(job["start_frame"], int) or isinstance(job["start_frame"], bool) or job["start_frame"] < 0:
        raise ValueError("start_frame must be a nonnegative integer")
    if not isinstance(job["samples"], int) or isinstance(job["samples"], bool) or not 11 <= job["samples"] <= 1500:
        raise ValueError("Each bounded segment requires 11..1500 samples at 0.2s")
    actors = job["actors"]
    if (not isinstance(actors, list) or not 1 <= len(actors) <= 4 or len(set(actors)) != len(actors)
            or any(not isinstance(actor, str) or not actor for actor in actors)):
        raise ValueError("Each segment requires 1..4 distinct configured source-local actors")
    if not job.get("selection"):
        raise ValueError("Declare segment selection criteria, not prediction-success selection")
    destination = job.get("output_destination", job["id"])
    if (not isinstance(destination, str) or not destination
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in destination)):
        raise ValueError("output_destination must be a safe single directory name within batch/jobs")


def source_reference(source, frame):
    if isinstance(source, SoccerTrackSource):
        return EvaluationReference(source).overlay(frame)
    return {
        "label": "SOURCE OBSERVATIONS ONLY: incomplete inventory, NOT privileged truth",
        "players": ([{"source_track": m.track, "xy": list(m.xy), "team": m.team}
                     for m in frame.measurements]
                    if source.header["observation_kind"] == "exo_source" else []),
        "absent_cache_slots": frame.missing_count, "privileged_available": False,
    }


def student_record(observation, engine, result):
    """Association tokens stay in the adapter; only anonymous slots enter learning."""
    entities = {e["id"]: e for e in result["belief"]["entities"]}
    released = []
    for entry in engine.memory.entries.values():
        if entry.last_seen == observation.timestamp_s:
            entity = entities[entry.identity]
            released.append({key: entity[key] for key in (
                "id", "xy", "team", "goalkeeper", "actor", "visible", "std_m")})
    inputs = engine.memory.current_input
    return {
        "schema_version": VERSION, "sample_id": result["sample_id"],
        "timestamp_s": observation.timestamp_s, "frame_index": observation.index,
        "units": "metres", "observation": released, "causal_memory": result["belief"],
        "world_input": ({key: getattr(inputs, key).tolist()
                         for key in ("x", "mask", "actor_xy", "last_xy")} if inputs is not None else None),
        "target_availability": {
            "action": False, "event": False, "receiver_proxy": False, "teacher_scores": False,
            "meaning": "UNANNOTATED, not a negative action or no-event label",
        },
        "metadata_not_features": ["sample_id", "timestamp_s", "frame_index"],
    }


def execute_job(root: Path, output: Path, job: dict, models: ModelSession):
    validate_job(job)
    source = load_source(root, job, job["source_adapter"])
    if str(source.match) != str(job["match"]):
        raise ValueError("Source identity mismatch")
    reference = EvaluationReference(source) if isinstance(source, SoccerTrackSource) else None
    frames = list(source.frames(job["start_frame"], job["samples"]))
    if not frames:
        raise ValueError("Source returned no frames")
    sequence = {"id": job["id"], "source": source.provenance,
                "dataset_relation": dataset_relation(job["match"], models),
                "selection": job["selection"], "actors": {},
                "evaluation_only": [source_reference(source, frame) for frame in frames]}
    is_training = job["intended_use"] == "training_export"
    stats = Counter()
    latency = []
    folder = output / "training_sample"
    writers = {}
    if is_training:
        folder.mkdir(parents=True, exist_ok=True)
    try:
        if is_training:
            for name in ("student", "privileged_exo", "optional_targets"):
                writers[name] = (folder / f"{name}.tmp").open("w", encoding="utf-8")
        for actor in job["actors"]:
            observations = job.get("observation", {})
            if set(observations) - {"seed", "fov", "dropout"}:
                raise ValueError("Unknown observation configuration; supported: seed/fov/dropout")
            provider = SyntheticObservationProvider(actor, **observations)
            engine = ReplayEngine(models, job.get("expiry_s", 6.), job.get("fallback", "last_seen"),
                                  job.get("mc_samples", 128))
            rows = []
            for number, frame in enumerate(frames):
                obs = (source.observe(frame, provider) if isinstance(source, ExternalObservationSource)
                       else provider.observe(frame))
                row = engine.step(obs, include_passing_assessment=job.get("passing_assessment", True))
                row["modules"]["receiver_reference"] = row.pop("exo_reference", None)
                if row["assessment"] is not None:
                    row["assessment"]["source_coverage"] = "PARTIAL_OR_UNKNOWN"
                    row["assessment"]["coverage"] += "; conditional on modeled players, source missingness is NOT actor FOV"
                row["measurement_quality"] = {entry.identity: entry.measurement.quality
                                              for entry in engine.memory.entries.values()
                                              if entry.last_seen == obs.timestamp_s}
                if models.p0 is not None:
                    supported = reference.receiver_states(frame.index, actor) if reference and obs.actor else None
                    row["exo_reference"] = (models.exo_reference(*supported) if supported is not None else {
                        "status": "UNAVAILABLE", "reason": "No supported privileged full-state reference",
                    })
                rows.append(row)
                stats["frames"] += 1
                stats["usable_observations"] += len(obs.measurements)
                stats["actor_missing_frames"] += int(obs.actor is None)
                stats["world_supported_frames"] += int(row["belief"]["model_contract_met"])
                stats["source_absent_slots"] += frame.missing_count
                stats["unknown_team_measurements"] += sum(m.team == 0 for m in obs.measurements)
                stats["unknown_role_measurements"] += sum(m.goalkeeper is None for m in obs.measurements)
                for entity in row["belief"]["entities"]:
                    if not entity["visible"]:
                        stats["hidden_estimates"] += 1
                        stats["fallbacks"] += int(entity["backend"] in ("last_seen", "constant_velocity"))
                        stats["learned_hidden_estimates"] += int(entity["backend"] in ("G1", "G3", "G4"))
                if row["assessment"] is not None:
                    for candidate in row["assessment"]["candidates"]:
                        stats["available_assessments" if candidate["status"] == "AVAILABLE"
                              else "unavailable_assessments"] += 1
                latency.append(row["latency_ms"])
                if is_training and number >= 10:
                    student = student_record(obs, engine, row)
                    teacher = {
                        "schema_version": VERSION, "sample_id": row["sample_id"],
                        "privileged_exo": reference.overlay(frame) if reference else None,
                        "target_availability": "ANNOTATION_STATE_ONLY" if reference else "UNAVAILABLE",
                        "source_provenance": source.provenance,
                        "quality": "Annotation proxy only; not independently validated physical state",
                    }
                    optional = {
                        "sample_id": row["sample_id"], "event_target": None, "receiver_proxy_target": None,
                        "teacher_scores": None,
                        "target_availability": {
                            "action": False, "event": False, "receiver_proxy": False, "teacher_scores": False,
                            "meaning": "UNANNOTATED, not a negative action or no-event label",
                        },
                        "status": "NO_EVENT_JOIN_OR_OOF_TEACHER: arbitrary timestamp, no invented labels",
                        "belief_checkpoint_relation": dataset_relation(source.match, models),
                    }
                    for key, record in (("student", student), ("privileged_exo", teacher),
                                        ("optional_targets", optional)):
                        writers[key].write(json.dumps(record, allow_nan=False) + "\n")
                    stats["training_records"] += 1
                if (number + 1) % 50 == 0:
                    write_json(output / "progress.json", {
                        "status": "RUNNING", "actor": actor, "completed_frames": number + 1,
                        "requested_frames": job["samples"], "stats": dict(stats),
                    })
            sequence["actors"][actor] = rows
    finally:
        for stream in writers.values():
            stream.close()
    if is_training:
        for name in writers:
            atomic_replace(folder / f"{name}.tmp", folder / f"{name}.jsonl")
        write_json(folder / "manifest.json", {
            "schema_version": VERSION, "samples": stats["training_records"], "match": source.match,
            "intended_use": "training_export", "source": source.provenance, "models": models.provenance,
            "teacher_predictions": "ABSENT: no claim of out-of-fold supervision",
            "separation": {"student": "student.jsonl", "privileged_supervision": "privileged_exo.jsonl",
                           "optional_proxy_targets": "optional_targets.jsonl"},
            "association_tokens": "adapter/privileged metadata only; anonymous student slots, not model features",
            "permitted_fitting_matches": list(FIT_MATCHES),
        })
    bundle = {
        "schema_version": VERSION, "notice": "External/source observations; visibility may be simulated, NOT measured gaze.",
        "description": "Observation ingestion -> causal memory -> frozen belief -> conditional Gaussian geometry",
        "units": "metres", "sampling_seconds": .2, "config": job, "model_provenance": models.provenance,
        "model_load_counts": models.load_counts, "truth_overlay_default": False, "exo_reference_default": False,
        "sequences": [sequence],
    }
    write_json(output / "replay.json", bundle)
    payload = json.dumps(bundle, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    temporary = output / "viewer.tmp.html"
    temporary.write_text(template.replace("__REPLAY_JSON__", payload), encoding="utf-8")
    atomic_replace(temporary, output / "viewer.html")
    stats["requested_actor_frames"] = job["samples"] * len(job["actors"])
    stats["source_frames_not_supplied"] = (job["samples"] - len(frames)) * len(job["actors"])
    stats.setdefault("training_records", 0)
    stats.setdefault("unavailable_assessments", 0)
    import numpy as np
    result = {**stats, "latency_ms_p50": float(np.median(latency)), "latency_ms_p95": float(np.quantile(latency, .95)),
              "source_coverage": "PARTIAL_OR_UNKNOWN", "no_interpolation": True,
              "missingness": "Absent source slots are not a real FOV or physical missing-player count",
              "actual_first_timestamp_s": frames[0].timestamp_s, "actual_last_timestamp_s": frames[-1].timestamp_s}
    write_json(output / "metrics.json", result)
    write_json(output / "progress.json", {"status": "EXPORTED", "stats": result})
    return result


def completed_matches(folder: Path, fingerprint: str) -> bool:
    marker = folder / "completion.json"
    if not marker.exists():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        return (value["status"] == "COMPLETE" and value["fingerprint"] == fingerprint
                and bool(value["artifacts"]) and all(
                    (folder / name).is_file() and file_digest(folder / name) == expected
                    for name, expected in value["artifacts"].items()))
    except (ValueError, KeyError, OSError):
        return False


def run_batch(config_path: Path, output: Path, resume=False, root=ROOT, only=None):
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != VERSION:
        raise ValueError(f"Batch config requires schema_version={VERSION}")
    jobs = config["jobs"]
    if not isinstance(jobs, list) or not jobs or len(jobs) > 10000:
        raise ValueError("Batch requires 1..10000 jobs; frames remain bounded per segment")
    ids = [job["id"] for job in jobs]
    if (len(set(ids)) != len(ids) or any(not isinstance(name, str) or not name
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
            for name in ids)):
        raise ValueError("Job IDs must be unique safe directory names")
    if only is not None:
        if only not in ids:
            raise ValueError(f"Unknown selected job: {only}")
        jobs = [job for job in jobs if job["id"] == only]
    destinations = [(job.get("output_destination", config.get("defaults", {}).get("output_destination", job["id"])))
                    for job in jobs]
    if any(not isinstance(name, str) for name in destinations) or len(set(destinations)) != len(destinations):
        raise ValueError("Batch jobs require distinct output destinations")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    code_hashes = implementation_hashes(root)
    results, active_key, models, loads = [], None, None, 0
    defaults = config.get("defaults", {})
    for specification in jobs:
        job_start = time.perf_counter()
        job = {**defaults, **specification}
        folder = output / "jobs" / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        fingerprint = None
        try:
            validate_job(job)
            folder = output / "jobs" / job.get("output_destination", job["id"])
            folder.mkdir(parents=True, exist_ok=True)
            experiment = root / job["experiment"]
            progress = experiment / "progress_summary.json"
            research_hash = file_digest(progress)
            if json.loads(progress.read_text())["status"] != "PAUSED_FOR_INTEGRATION":
                raise ValueError("Research queue must remain PAUSED_FOR_INTEGRATION")
            dependencies = {
                "source": source_dependencies(root, job), "implementation": code_hashes,
                "models": model_dependencies(experiment, job["backend"], job.get("exo_p0", False)),
            }
            fingerprint = digest({"job": job, "dependencies": dependencies})
            # Keep artifact paths below legacy Windows rename limits in this workspace.
            folder = folder / fingerprint[:16]
            folder.mkdir(parents=True, exist_ok=True)
            identity_path = folder / "fingerprint.json"
            if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8"))["sha256"] != fingerprint:
                raise RuntimeError("Hash-prefix collision; refusing to overwrite another job generation")
            write_json(identity_path, {"sha256": fingerprint})
            if completed_matches(folder, fingerprint):
                if not resume:
                    raise FileExistsError("Completed output exists; use --resume to reuse hash-matched jobs")
                completion = json.loads((folder / "completion.json").read_text())
                results.append({"id": job["id"], "status": "SKIPPED_HASH_MATCH", "fingerprint": fingerprint,
                                "output": str(folder), "metrics": completion["metrics"]})
                print(f"{job['id']}: SKIPPED_HASH_MATCH", flush=True)
                continue
            marker = folder / "completion.json"
            if marker.exists():
                print(f"{job['id']}: INVALID_COMPLETION_ARTIFACTS, rebuilding segment without appending records", flush=True)
                marker.unlink()
            write_json(folder / "progress.json", {"status": "RUNNING", "fingerprint": fingerprint})
            write_json(folder / "resolved_job.json", job)
            key = (str(experiment.resolve()), job["backend"], job.get("exo_p0", False), job.get("device", "cpu"),
                   digest(dependencies["models"]))
            if key != active_key:
                models = ModelSession(experiment, job["backend"], job.get("exo_p0", False), job.get("device", "cpu"))
                active_key, loads = key, loads + 1
            metrics = execute_job(root, folder, job, models)
            if (source_dependencies(root, job) != dependencies["source"] or file_digest(progress) != research_hash
                    or model_dependencies(experiment, job["backend"], job.get("exo_p0", False)) != dependencies["models"]
                    or implementation_hashes(root) != code_hashes):
                raise RuntimeError("Source or paused research changed during job; output not marked complete")
            artifacts = {str(path.relative_to(folder)): file_digest(path) for path in folder.rglob("*")
                         if path.is_file() and path.name not in ("completion.json", "failure.json")}
            metrics["execution_seconds"] = time.perf_counter() - job_start
            metrics["output_bytes"] = sum((folder / name).stat().st_size for name in artifacts)
            write_json(marker, {
                "status": "COMPLETE", "fingerprint": fingerprint, "dependencies": dependencies,
                "research_queue_unchanged_sha256": research_hash, "artifacts": artifacts, "metrics": metrics,
            })
            results.append({"id": job["id"], "status": "COMPLETE", "fingerprint": fingerprint,
                            "output": str(folder), "metrics": metrics})
            print(f"{job['id']}: COMPLETE ({metrics['frames']} frames)", flush=True)
        except Exception as error:
            # A batch must continue independent jobs, but every failure is persisted and fails the CLI.
            failure = {"id": job["id"], "status": "FAILED", "fingerprint": fingerprint,
                       "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc(), "output": str(folder)}
            failure_path = output / "failures" / f"{job['id']}.json"
            write_json(failure_path, failure)
            if not (folder / "completion.json").exists():
                write_json(folder / "failure.json", failure)
                write_json(folder / "progress.json", {"status": "FAILED", "error": str(error)})
            failure["failure_record"] = str(failure_path)
            results.append(failure)
            print(f"{job['id']}: FAILED: {error}", flush=True)
        write_json(output / "progress.json", {"status": "RUNNING", "processed_jobs": len(results), "total_jobs": len(jobs)})
    totals = Counter()
    for result in results:
        for name in ("frames", "usable_observations", "fallbacks", "unavailable_assessments",
                     "available_assessments", "learned_hidden_estimates", "training_records", "output_bytes"):
            totals[name] += result.get("metrics", {}).get(name, 0)
    summary = {
        "schema_version": VERSION, "status": "FAILED" if any(r["status"] == "FAILED" for r in results) else "COMPLETE",
        "config_sha256": file_digest(config_path), "implementation": code_hashes,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "execution_seconds": time.perf_counter() - started, "model_sessions_loaded": loads,
        "jobs": results, "counts": dict(Counter(r["status"] for r in results)), "totals": dict(totals),
        "no_training": True,
    }
    write_json(output / "batch_summary.json", summary)
    write_json(output / "training_index.json", {
        "schema_version": VERSION, "status": summary["status"],
        "warning": "Consume only these completed current generations; do not glob failed or superseded attempt directories",
        "shards": [{"job": result["id"], "folder": str(Path(result["output"]) / "training_sample"),
                    "samples": result["metrics"]["training_records"], "fingerprint": result["fingerprint"]}
                   for result in results if result["status"] in ("COMPLETE", "SKIPPED_HASH_MATCH")
                   and result["metrics"].get("training_records", 0) > 0],
    })
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    write_json(output / "runs" / f"{run_id}.json", summary)
    if summary["status"] == "COMPLETE" and not (output / "first_completed_summary.json").exists():
        write_json(output / "first_completed_summary.json", summary)
    if summary["status"] == "COMPLETE" and any(result["status"] == "COMPLETE" for result in results):
        write_json(output / "last_executed_summary.json", summary)
    write_json(output / "progress.json", {"status": summary["status"], "processed_jobs": len(results), "total_jobs": len(jobs)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", help="Run one configured segment (including training-only jobs)")
    args = parser.parse_args()
    summary = run_batch(args.config.resolve(), args.output.resolve(), args.resume, only=args.only)
    print(args.output / "batch_summary.json")
    if summary["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
