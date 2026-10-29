"""Validate long retrospective replay, action queries and checkpoint provenance."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from src.backbone.batch import VERSION as BATCH_VERSION, validate_job
from src.backbone.source import SoccerTrackSource
from src.player_centric_dataset import ACTION_LABELS
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "outputs/backbone_v0_5_20261020/corpus/training_index.json"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_selection(config_path: Path, report_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(DEFAULT_INDEX.read_text(encoding="utf-8"))
    index_hash = file_digest(DEFAULT_INDEX)
    if index.get("status") != "COMPLETE" or index.get("schema_version") != "football.action-corpus.0.4":
        raise ValueError("Stress queries require the complete indexed action corpus")
    shards = {
        (str(shard["match"]), shard["half"]): shard
        for shard in index["shards"]
        if shard["split"] == "retrospective_evaluation"
    }
    selected_rows = []
    jobs = []
    labels = Counter()
    for specification in config["jobs"]:
        job = {**config.get("defaults", {}), **specification}
        validate_job(job)
        key = (str(job["match"]), job["half"])
        if key not in shards or job.get("intended_use") != "retrospective_replay":
            raise ValueError(f"Stress selection is not held-out retrospective data: {job['id']}")
        shard = shards[key]
        folder = Path(shard["folder"])
        for filename, digest in shard["artifact_hashes"].items():
            artifact = folder / filename
            if not artifact.is_file() or file_digest(artifact) != digest:
                raise ValueError(f"Indexed stress-selection artifact changed: {artifact}")
        metadata_rows = _read_jsonl(folder / "student_metadata.jsonl")
        target_rows = _read_jsonl(folder / "optional_targets.jsonl")
        metadata_by_id = {row["sample_id"]: row for row in metadata_rows}
        targets_by_id = {row["sample_id"]: row for row in target_rows}
        if len(metadata_by_id) != len(metadata_rows) or len(targets_by_id) != len(target_rows):
            raise ValueError(f"Duplicate event-conditioned sample IDs: {job['id']}")
        source = SoccerTrackSource(ROOT, *key)
        start = int(job["start_frame"])
        samples = int(job["samples"])
        frames = range(start, start + samples * 5, 5)
        actor_coverage = {}
        for actor in job["actors"]:
            try:
                slot = int(actor.removeprefix("track_"))
            except ValueError as error:
                raise ValueError(f"Invalid source actor slot {actor!r} in {job['id']}") from error
            if not all(bool(source.raw["observed"][frame, slot]) for frame in frames):
                raise ValueError(f"Actor is not fully observed across stress window: {job['id']}/{actor}")
            segment = int(source.raw["segments"][start, slot])
            if any(int(source.raw["segments"][frame, slot]) != segment for frame in frames):
                raise ValueError(f"Actor identity segment changes across stress window: {job['id']}/{actor}")
            actor_coverage[actor] = {"observed_samples": samples, "stable_segment": segment}
        for query in job.get("action_queries", []):
            query_frame = int(query["frame_index"])
            actor = query["actor_track"]
            if (actor not in actor_coverage or query_frame not in frames
                    or (query_frame - start) % 5 != 0):
                raise ValueError(f"Stress query is outside its actor/window sample grid: {job['id']}")
            matches = [
                row for row in target_rows
                if row.get("event_id") == query.get("event_id")
                and row.get("annotation_index") == query.get("annotation_index")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Event ID/annotation-index join is not unique: "
                    f"{query.get('event_id')}#{query.get('annotation_index')}"
                )
            target = matches[0]
            metadata = metadata_by_id.get(target["sample_id"])
            if (metadata is None or targets_by_id.get(target["sample_id"]) != target
                    or metadata.get("split") != "retrospective_evaluation"
                    or metadata.get("match") != key[0] or metadata.get("half") != key[1]
                    or metadata.get("actor_track") != actor
                    or int(metadata.get("query_frame", -1)) != query_frame
                    or target.get("mapped_action_label") != query.get("mapped_action_label")
                    or target.get("original_bas_label") != query.get("original_bas_label")
                    or target.get("mapping_provenance") != query.get("mapping_provenance")
                    or target.get("actor", {}).get("source_track") != actor
                    or target.get("timing", {}).get("no_post_event_frames") is not True
                    or max(metadata.get("input_frame_indices", [])) > query_frame
                    or max(metadata.get("input_frame_indices", [])) != query_frame
                    or metadata.get("target_availability", {}).get("action") is not False):
                raise ValueError(f"Stress event query is not joined to a causal held-out target: {job['id']}")
            if query["mapped_action_label"] not in ACTION_LABELS:
                raise ValueError(f"Untrained action class in stress query: {query['mapped_action_label']}")
            labels[query["mapped_action_label"]] += 1
            selected_rows.append({
                "job_id": job["id"], "event_id": query["event_id"],
                "annotation_index": query["annotation_index"],
                "actor_track": actor, "frame_index": query_frame,
                "mapped_action_label": query["mapped_action_label"],
                "sample_id": target["sample_id"],
            })
        jobs.append({
            "id": job["id"], "match": key[0], "half": key[1],
            "samples": samples, "fully_covered_actors": actor_coverage,
            "source_cache_sha256": source.provenance["cache_sha256"],
            "association_sha256": source.provenance["association_sha256"],
            "event_conditioned_queries": len(job.get("action_queries", [])),
            "action_classes": sorted({query["mapped_action_label"] for query in job.get("action_queries", [])}),
            "retrospective_only": True,
        })
    labels_outside_pass_family = sorted(
        label for label in labels if label not in {"PASS", "HIGH_PASS"}
    )
    if len(labels) < 3 or len(labels_outside_pass_family) < 2:
        raise ValueError("Stress selection does not cover three classes and two non-pass action classes")
    report = {
        "schema_version": "football.backbone-v0.6-stress-selection-validation.1",
        "status": "PASS",
        "selection_config_sha256": file_digest(config_path),
        "training_index_sha256": index_hash,
        "retrospective_only": True,
        "jobs": jobs,
        "totals": {
            "unique_source_frames": sum(job["samples"] for job in jobs),
            "actor_steps": sum(job["samples"] * len(job["fully_covered_actors"]) for job in jobs),
            "event_conditioned_queries": len(selected_rows),
            "action_classes": sorted(labels),
            "non_pass_action_classes": labels_outside_pass_family,
            "training_examples_used": 0,
        },
        "query_provenance": selected_rows,
        "selection_policy": "Fixed earliest retrospective windows and full-source-coverage actors; event labels used only to schedule evaluation-only queries, never model successes.",
    }
    write_json(report_path, report)
    return report


def validate_stress(config_path: Path, output: Path, report_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection_suffix = config_path.stem.removeprefix("stress_config")
    selection_validation = validate_selection(
        config_path, report_path.parent / f"selection_validation{selection_suffix}.json",
    )
    summary = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != "backbone.batch.0.2":
        raise ValueError("Unsupported stress batch configuration")
    if summary.get("status") != "COMPLETE" or summary.get("no_training") is not True:
        raise ValueError("Stress replay batch is incomplete or unexpectedly marked as training")
    if summary.get("config_sha256") != file_digest(config_path):
        raise ValueError("Stress replay summary/config hash mismatch")
    configured_jobs = {job["id"]: job for job in config["jobs"]}
    completed_jobs = {job["id"]: job for job in summary["jobs"]}
    if set(configured_jobs) != set(completed_jobs) or any(
        result["status"] != "COMPLETE" for result in completed_jobs.values()
    ):
        raise ValueError("Stress replay does not contain every configured complete job")
    if summary.get("action_model_sessions_loaded") != 1:
        raise ValueError("The selected action checkpoint was not loaded exactly once across the batch")

    reports = []
    total_actor_steps = 0
    total_queries = 0
    for job_id, job in configured_jobs.items():
        result = completed_jobs[job_id]
        metrics = result["metrics"]
        job_output = Path(result["output"])
        completion = json.loads((job_output / "completion.json").read_text(encoding="utf-8"))
        if completion.get("status") != "COMPLETE" or completion.get("fingerprint") != result["fingerprint"]:
            raise ValueError(f"Job completion fingerprint mismatch: {job_id}")
        for relative, digest in completion["artifacts"].items():
            artifact = job_output / relative
            if not artifact.is_file() or file_digest(artifact) != digest:
                raise ValueError(f"Completed stress artifact changed: {job_id}/{relative}")
        replay = json.loads((job_output / "replay.json").read_text(encoding="utf-8"))
        if replay.get("schema_version") != BATCH_VERSION:
            raise ValueError(f"Unexpected replay schema in {job_id}")
        sequence = next((item for item in replay["sequences"] if item["id"] == job_id), None)
        if sequence is None:
            raise ValueError(f"Stress job sequence missing from replay: {job_id}")
        query_map = {
            (int(query["frame_index"]), query["actor_track"]): query
            for query in job.get("action_queries", [])
        }
        actor_reports = []
        for actor in job["actors"]:
            rows = sequence["actors"].get(actor, [])
            expected = int(job["samples"])
            if len(rows) != expected:
                raise ValueError(f"{job_id}/{actor}: expected {expected} actor rows, got {len(rows)}")
            frame_indices = [int(row["frame_index"]) for row in rows]
            sample_ids = [row["sample_id"] for row in rows]
            if (len(set(frame_indices)) != expected or len(set(sample_ids)) != expected
                    or any(b - a != 5 for a, b in zip(frame_indices, frame_indices[1:]))):
                raise ValueError(f"{job_id}/{actor}: duplicate rows or non-causal sampling grid")
            scheduled = [
                (frame, query) for (frame, query_actor), query in query_map.items()
                if query_actor == actor
            ]
            for frame, query in scheduled:
                matching = [row for row in rows if int(row["frame_index"]) == frame]
                if len(matching) != 1:
                    raise ValueError(f"{job_id}/{actor}: scheduled query did not resolve uniquely")
                row = matching[0]
                inference = row.get("action_inference", {})
                scores = inference.get("scores")
                if (inference.get("status") != "supported"
                        or set(scores or {}) != set(ACTION_LABELS)
                        or not row["action"].get("action_type")):
                    raise ValueError(f"{job_id}/{actor}: checkpoint-backed action inference failed")
                annotation = row.get("recorded_annotation_evaluation_only", {})
                if (annotation.get("event_id") != query["event_id"]
                        or annotation.get("annotation_index") != query["annotation_index"]
                        or annotation.get("mapped_action_label") != query["mapped_action_label"]
                        or not row.get("action_query_conditioning", "").startswith("event-conditioned")):
                    raise ValueError(f"{job_id}/{actor}: evaluation annotation is not separated/keyed correctly")
            actor_reports.append({
                "actor": actor,
                "rows": expected,
                "unique_frames": len(set(frame_indices)),
                "event_conditioned_queries": len(scheduled),
                "model_contract_ready_rows": sum(bool(row["belief"]["model_contract_met"]) for row in rows),
                "action_scores_are_uncalibrated": all(
                    row.get("action_inference", {}).get("score_semantics", "").startswith("Uncalibrated")
                    for frame, _ in scheduled
                    for row in rows if int(row["frame_index"]) == frame
                ),
            })
            total_actor_steps += expected
            total_queries += len(scheduled)
        expected_calls = len(job.get("action_queries", []))
        if metrics.get("action_model_forward_calls") != expected_calls:
            raise ValueError(f"{job_id}: action inference counter does not match scheduled queries")
        reports.append({
            "job_id": job_id,
            "status": result["status"],
            "frames_per_actor": int(job["samples"]),
            "actors": actor_reports,
            "action_forward_calls": metrics.get("action_model_forward_calls"),
            "world_model_forward_calls": metrics.get("world_model_forward_calls"),
            "completion_artifact_count": len(completion["artifacts"]),
            "dropout_before_memory": bool(job.get("observation", {}).get("dropout")),
            "output": str(job_output.resolve()),
        })

    selected_totals = selection_validation["totals"]
    if (total_queries != selected_totals["event_conditioned_queries"]
            or total_actor_steps != selected_totals["actor_steps"]):
        raise ValueError("Executed stress replay does not match the validated selection totals")
    experiment = ROOT / config["defaults"]["experiment"]
    research_status = json.loads((experiment / "progress_summary.json").read_text(encoding="utf-8"))
    if research_status.get("status") != "PAUSED_FOR_INTEGRATION":
        raise ValueError("Dynamic-GNN research queue is not paused for integration")
    if summary.get("action_model_sessions_loaded") != 1:
        raise ValueError("Action model reuse/load counter mismatch")
    report = {
        "schema_version": "football.backbone-v0.6-stress-validation.1",
        "status": "PASS",
        "batch_summary_sha256": file_digest(output / "batch_summary.json"),
        "selection_validation": {
            "status": selection_validation["status"],
            "action_classes": selection_validation["totals"]["action_classes"],
            "non_pass_action_classes": selection_validation["totals"]["non_pass_action_classes"],
            "event_conditioned_queries": selection_validation["totals"]["event_conditioned_queries"],
        },
        "job_count": len(reports),
        "jobs": reports,
        "unique_source_frames": sum(int(job["frames_per_actor"]) for job in reports),
        "actor_steps": total_actor_steps,
        "event_conditioned_queries": total_queries,
        "action_model_sessions_loaded": summary["action_model_sessions_loaded"],
        "research_queue": research_status["status"],
        "retrospective_only": all(job["intended_use"] == "retrospective_replay" for job in configured_jobs.values()),
        "training_performed": False,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("selection", "completed"), default="completed")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    if args.phase == "selection":
        result = validate_selection(config_path, report_path)
    else:
        result = validate_stress(config_path, output, report_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
