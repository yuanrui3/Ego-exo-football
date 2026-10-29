"""Exercise safe mid-sequence interruption and observation-prefix reconstruction."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import torch

from src.backbone.actions import ActionModelSession
from src.backbone.models import ModelSession
from src.backbone.external import ExternalObservationSource
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.source import load_source
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]


def _interrupt_cursor(samples: int) -> int:
    return samples // 2


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _observation_digest(observation) -> str:
    return hashlib.sha256(_canonical(dataclasses.asdict(observation))).hexdigest()


def _row_for_sample(engine, source, provider, frame, query, job) -> tuple[dict, str]:
    observation = (
        source.observe(frame, provider)
        if isinstance(source, ExternalObservationSource)
        else provider.observe(frame)
    )
    row = engine.step(
        observation,
        include_passing_assessment=job.get("passing_assessment", True),
        action_query=query is not None,
    )
    if query is not None:
        row["recorded_annotation_evaluation_only"] = {
            key: query[key] for key in (
                "event_id", "annotation_index", "original_bas_label",
                "mapped_action_label", "mapping_provenance",
            ) if key in query
        }
        row["action_query_conditioning"] = query["conditioning"]
    row["modules"]["receiver_reference"] = row.pop("exo_reference", None)
    if row["assessment"] is not None:
        row["assessment"]["source_coverage"] = "PARTIAL_OR_UNKNOWN"
        row["assessment"]["coverage"] += (
            "; conditional on modeled players, source missingness is NOT actor FOV"
        )
    row["measurement_quality"] = {
        entry.identity: entry.measurement.quality
        for entry in engine.memory.entries.values()
        if entry.last_seen == observation.timestamp_s
    }
    return row, _observation_digest(observation)


def _assert_close(actual: object, expected: object, path: str, max_error: list[float]) -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        if actual is not expected:
            raise ValueError(f"Boolean replay parity mismatch at {path}")
        return
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        difference = abs(float(actual) - float(expected))
        max_error[0] = max(max_error[0], difference)
        if not math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=2e-5):
            raise ValueError(f"Numerical replay parity mismatch at {path}: {actual} vs {expected}")
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        actual_keys = set(actual) - {"latency_ms"}
        expected_keys = set(expected) - {"latency_ms"}
        if actual_keys != expected_keys:
            raise ValueError(f"Replay output keys differ at {path}")
        for key in actual_keys:
            _assert_close(actual[key], expected[key], f"{path}.{key}", max_error)
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Replay list lengths differ at {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(left, right, f"{path}[{index}]", max_error)
        return
    if actual != expected:
        raise ValueError(f"Replay output mismatch at {path}: {actual!r} vs {expected!r}")


def _load_context(config_path: Path, job_id: str):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = [job for job in config["jobs"] if job["id"] == job_id]
    if len(jobs) != 1:
        raise ValueError(f"Expected exactly one configured replay job {job_id}")
    job = {**config.get("defaults", {}), **jobs[0]}
    if job["source_adapter"] != "soccertrack_cache":
        raise ValueError("Resume probe currently requires the verified SoccerTrack cache adapter")
    torch.set_num_threads(4)
    models = ModelSession(
        ROOT / job["experiment"], job["backend"], job.get("exo_p0", False), job.get("device", "cuda"),
    )
    checkpoint = Path(job["action_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    action_model = ActionModelSession(checkpoint, device=job.get("device", "cuda"))
    source = load_source(ROOT, job, job["source_adapter"])
    return job, models, action_model, source


def interrupt(config_path: Path, job_id: str, output: Path, baseline_summary: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Resume-probe output already exists: {output}")
    job, models, action_model, source = _load_context(config_path, job_id)
    matches = [item for item in json.loads(baseline_summary.read_text(encoding="utf-8"))["jobs"]
               if item["id"] == job_id]
    if len(matches) != 1 or matches[0]["status"] != "COMPLETE":
        raise ValueError("Baseline sustained sequence must have a completed matching job")
    baseline_replay = Path(matches[0]["output"]) / "replay.json"
    baseline = json.loads(baseline_replay.read_text(encoding="utf-8"))
    actor = job["actors"][0]
    baseline_rows = baseline["sequences"][0]["actors"].get(actor, [])
    if len(baseline_rows) != job["samples"]:
        raise ValueError(f"Uninterrupted baseline does not contain {job['samples']} rows for {actor}")
    provider = SyntheticObservationProvider(actor, **job.get("observation", {}))
    engine = ReplayEngine(
        models, job.get("expiry_s", 6.), job.get("fallback", "last_seen"),
        job.get("mc_samples", 128), action_model=action_model,
    )
    query_by_frame = {int(q["frame_index"]): q for q in job.get("action_queries", [])
                      if q["actor_track"] == actor}
    frames = list(source.frames(job["start_frame"], job["samples"]))
    if len(frames) != job["samples"]:
        raise ValueError(f"Expected {job['samples']} source samples; got {len(frames)}")
    cursor = _interrupt_cursor(len(frames))
    output.mkdir(parents=True)
    rows_path = output / "reconstructed_rows.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as stream:
        for frame in frames[:cursor]:
            row, observation_hash = _row_for_sample(
                engine, source, provider, frame, query_by_frame.get(frame.index), job,
            )
            record = {
                "sample_id": row["sample_id"],
                "frame_index": row["frame_index"],
                "timestamp_s": row["timestamp_s"],
                "observation_sha256": observation_hash,
                "row": row,
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
    prefix_records = [
        json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(prefix_records) != cursor:
        raise ValueError("Safe-boundary interruption did not flush the complete prefix")
    checkpoint = {
        "schema_version": "football.sustained-replay-resume.0.6",
        "status": "CONTROLLED_INTERRUPTION",
        "job_id": job_id,
        "cursor": cursor,
        "total_samples": len(frames),
        "actor": actor,
        "config_sha256": file_digest(config_path),
        "source_cache_sha256": source.provenance["cache_sha256"],
        "baseline_replay_sha256": file_digest(baseline_replay),
        "prefix_rows_sha256": file_digest(rows_path),
        "prefix_observation_hashes": [row["observation_sha256"] for row in prefix_records],
        "prefix_sample_ids": [row["sample_id"] for row in prefix_records],
        "interruption_boundary": f"After a complete {cursor}-sample frame boundary and flushed output; memory will be reconstructed from this observation prefix.",
    }
    write_json(output / "checkpoint.json", checkpoint)
    write_json(output / "progress.json", {
        "status": checkpoint["status"],
        "completed_samples": cursor,
        "total_samples": len(frames),
        "safe_checkpoint": str((output / "checkpoint.json").resolve()),
        "prefix_rows": str(rows_path.resolve()),
    })
    return checkpoint


def resume(config_path: Path, job_id: str, output: Path, baseline_summary: Path) -> dict:
    checkpoint_path = output / "checkpoint.json"
    rows_path = output / "reconstructed_rows.jsonl"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("status") != "CONTROLLED_INTERRUPTION":
        raise ValueError("Resume requires the verified controlled-interruption boundary")
    if file_digest(rows_path) != checkpoint["prefix_rows_sha256"]:
        raise ValueError("Interrupted prefix hash changed; refusing to append")
    job, models, action_model, source = _load_context(config_path, job_id)
    cursor = _interrupt_cursor(int(job["samples"]))
    if checkpoint.get("cursor") != cursor:
        raise ValueError("Resume checkpoint cursor does not match the configured safe boundary")
    if (file_digest(config_path) != checkpoint["config_sha256"]
            or source.provenance["cache_sha256"] != checkpoint["source_cache_sha256"]):
        raise ValueError("Configuration or verified source cache changed since interruption")
    baseline_jobs = [item for item in json.loads(baseline_summary.read_text(encoding="utf-8"))["jobs"]
                     if item["id"] == job_id]
    if len(baseline_jobs) != 1 or baseline_jobs[0]["status"] != "COMPLETE":
        raise ValueError("Uninterrupted control replay is missing")
    baseline_path = Path(baseline_jobs[0]["output"]) / "replay.json"
    if file_digest(baseline_path) != checkpoint["baseline_replay_sha256"]:
        raise ValueError("Uninterrupted control replay changed after the interruption")
    baseline_replay = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows = baseline_replay["sequences"][0]["actors"][job["actors"][0]]
    records = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != cursor or [row["sample_id"] for row in records] != checkpoint["prefix_sample_ids"]:
        raise ValueError("Interrupted prefix sample IDs/count do not match checkpoint")
    actor = job["actors"][0]
    provider = SyntheticObservationProvider(actor, **job.get("observation", {}))
    engine = ReplayEngine(
        models, job.get("expiry_s", 6.), job.get("fallback", "last_seen"),
        job.get("mc_samples", 128), action_model=action_model,
    )
    query_by_frame = {int(q["frame_index"]): q for q in job.get("action_queries", [])
                      if q["actor_track"] == actor}
    frames = list(source.frames(job["start_frame"], job["samples"]))
    if len(frames) != job["samples"]:
        raise ValueError(f"Expected {job['samples']} source samples during resume; got {len(frames)}")
    max_numeric_error = [0.0]
    observation_hashes = []
    # Rebuild the causal provider/memory state using only the same released observation stream.
    for index, frame in enumerate(frames[:cursor]):
        row, observation_hash = _row_for_sample(
            engine, source, provider, frame, query_by_frame.get(frame.index), job,
        )
        observation_hashes.append(observation_hash)
        if observation_hash != records[index]["observation_sha256"]:
            raise ValueError(f"Reconstructed observation prefix changed at sample {index}")
        _assert_close(row, records[index]["row"], f"prefix[{index}]", max_numeric_error)
        _assert_close(row, baseline_rows[index], f"baseline_prefix[{index}]", max_numeric_error)

    with rows_path.open("a", encoding="utf-8", newline="\n") as stream:
        for index, frame in enumerate(frames[cursor:], start=cursor):
            row, observation_hash = _row_for_sample(
                engine, source, provider, frame, query_by_frame.get(frame.index), job,
            )
            _assert_close(row, baseline_rows[index], f"baseline_suffix[{index}]", max_numeric_error)
            record = {
                "sample_id": row["sample_id"],
                "frame_index": row["frame_index"],
                "timestamp_s": row["timestamp_s"],
                "observation_sha256": observation_hash,
                "row": row,
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    complete = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    frames_out = [int(row["frame_index"]) for row in complete]
    if len(complete) != job["samples"] or len(set(frames_out)) != job["samples"]:
        raise ValueError(f"Resumed output contains {len(complete)} rows with {len(set(frames_out))} unique frames")
    result = {
        "schema_version": "football.sustained-replay-resume-validation.0.6",
        "status": "PASS",
        "job_id": job_id,
        "controlled_interruption_after_samples": cursor,
        "resumed_samples": len(complete) - cursor,
        "total_samples": len(complete),
        "unique_frames": len(set(frames_out)),
        "duplicate_records": len(complete) - len(set(frames_out)),
        "observation_prefix_reconstructed_and_hash_matched": True,
        "prefix_samples_compared_to_control": cursor,
        "suffix_samples_compared_to_uninterrupted_control": len(complete) - cursor,
        "maximum_numeric_absolute_difference": max_numeric_error[0],
        "tolerance": {"relative": 1e-5, "absolute": 2e-5},
        "identity_history": "All prefix and suffix belief/assessment/action outputs compared recursively to the uninterrupted control, including causal memory IDs.",
        "hidden_truth_initialization": False,
        "g1_forward_calls_during_reconstruction_and_resume": models.world_forward_calls,
        "action_forward_calls_during_reconstruction_and_resume": action_model.forward_calls,
        "prefix_rows_sha256_before_resume": checkpoint["prefix_rows_sha256"],
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": file_digest(rows_path),
        "decision": "PASS_AND_ADVANCE",
    }
    write_json(output / "resume_validation.json", result)
    write_json(output / "progress.json", {
        "status": "COMPLETE",
        "completed_samples": len(complete),
        "total_samples": len(complete),
        "validation": str((output / "resume_validation.json").resolve()),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("interrupt", "resume"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    baseline = args.baseline_summary if args.baseline_summary.is_absolute() else ROOT / args.baseline_summary
    result = (
        interrupt(config.resolve(), args.job, output.resolve(), baseline.resolve())
        if args.phase == "interrupt"
        else resume(config.resolve(), args.job, output.resolve(), baseline.resolve())
    )
    print(json.dumps({
        "status": result["status"],
        "job_id": args.job,
        "cursor_or_samples": result.get("cursor", result.get("total_samples")),
        "output": str(output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
