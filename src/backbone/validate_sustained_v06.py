"""Validate sustained-replay evidence directly from completed batch artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]


def validate(
    summary_path: Path,
    *,
    expected_job: str,
    expected_actors: list[str],
    expected_samples: int,
    expected_unique_frames: int,
    output_path: Path,
) -> dict:
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise ValueError(f"Batch is not complete: {summary.get('status')}")
    matches = [item for item in summary["jobs"] if item.get("id") == expected_job]
    if len(matches) != 1 or matches[0].get("status") not in ("COMPLETE", "SKIPPED_HASH_MATCH"):
        raise ValueError(f"Expected exactly one completed job {expected_job}")
    job = matches[0]
    folder = Path(job["output"])
    completion_path = folder / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE":
        raise ValueError("Per-job completion marker is not COMPLETE")
    for relative, expected in completion["artifacts"].items():
        artifact = folder / relative
        if not artifact.is_file() or file_digest(artifact) != expected:
            raise ValueError(f"Completed replay artifact is missing or changed: {artifact}")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    replay = json.loads((folder / "replay.json").read_text(encoding="utf-8"))
    if len(replay.get("sequences", [])) != 1:
        raise ValueError("Sustained replay must contain exactly one source sequence")
    sequence = replay["sequences"][0]
    actor_rows = sequence["actors"]
    if list(actor_rows) != expected_actors:
        raise ValueError(f"Actor order/coverage mismatch: expected {expected_actors}, got {list(actor_rows)}")

    per_actor = {}
    all_source_frames = set()
    backend_occurrences: Counter[str] = Counter()
    assessment_statuses: Counter[str] = Counter()
    queries = []
    not_queried = 0
    identity_resets = 0
    actor_missing = 0
    g1_estimate_frames = 0
    for actor, rows in actor_rows.items():
        if len(rows) != expected_samples:
            raise ValueError(f"{actor} has {len(rows)} frames; expected {expected_samples}")
        times = [float(row["timestamp_s"]) for row in rows]
        indices = [int(row["frame_index"]) for row in rows]
        if len(set(times)) != expected_samples or len(set(indices)) != expected_samples:
            raise ValueError(f"{actor} contains duplicate timestamps or source frames")
        gaps = [right - left for left, right in zip(times, times[1:])]
        max_gap = max(gaps, default=0.0)
        if any(abs(gap - 0.2) > 1e-4 for gap in gaps):
            raise ValueError(f"{actor} has missing or non-causal 0.2-second samples")
        for row in rows:
            if abs(float(row["timestamp_s"]) - int(row["frame_index"]) / 25.0) > 1e-5:
                raise ValueError(f"{actor} timestamp does not match native 25Hz source frame")
            all_source_frames.add((row["source_id"], int(row["frame_index"])))
            belief = row["belief"]
            if not belief["actor_available"]:
                actor_missing += 1
            identity_resets += sum("DISCONTINUITY" in warning for warning in belief["warnings"])
            hidden_g1 = False
            for entity in belief["entities"]:
                backend_occurrences[entity["backend"]] += 1
                hidden_g1 |= entity["backend"] == "G1" and not entity["visible"]
            g1_estimate_frames += int(hidden_g1)
            assessment = row.get("assessment")
            if assessment:
                for candidate in assessment.get("candidates", []):
                    assessment_statuses[candidate["status"]] += 1
            inference = row.get("action_inference")
            if inference is not None:
                if inference["status"] == "not_queried":
                    not_queried += 1
                else:
                    annotation = row.get("recorded_annotation_evaluation_only")
                    if annotation is None:
                        raise ValueError("A scored retrospective query lacks its separate evaluation-only annotation")
                    if "mapped_action_label" in annotation and "action" not in row:
                        raise ValueError("Annotation exists without the separate action-output record")
                    queries.append({
                        "actor": actor,
                        "frame_index": int(row["frame_index"]),
                        "timestamp_s": float(row["timestamp_s"]),
                        "inference_status": inference["status"],
                        "predicted_action": row["action"]["action_type"],
                        "annotation_evaluation_only": annotation.get("mapped_action_label"),
                        "scores": inference.get("scores"),
                        "score_semantics": inference.get("score_semantics"),
                    })
        per_actor[actor] = {
            "samples": len(rows),
            "first_timestamp_s": times[0],
            "last_timestamp_s": times[-1],
            "elapsed_span_s": times[-1] - times[0],
            "max_gap_s": max_gap,
            "first_frame": indices[0],
            "last_frame": indices[-1],
        }
    unique_frames = len(all_source_frames)
    if unique_frames != expected_unique_frames:
        raise ValueError(f"Unique source frames {unique_frames} != expected {expected_unique_frames}")
    actor_steps = sum(len(rows) for rows in actor_rows.values())
    if int(metrics["frames"]) != actor_steps:
        raise ValueError("Batch summary actor-step count does not reconcile to replay rows")
    if int(metrics["learned_hidden_estimates"]) != backend_occurrences["G1"]:
        raise ValueError("Learned hidden-estimate metric does not reconcile to actual G1-backed memory outputs")
    if int(metrics["fallbacks"]) != sum(
        count for backend, count in backend_occurrences.items()
        if backend in ("last_seen", "constant_velocity")
    ):
        raise ValueError("Fallback metric does not reconcile to emitted fallback-backed memory outputs")
    forward_calls = int(metrics["world_supported_frames"])
    action_calls = len([query for query in queries if query["inference_status"] == "supported"])
    if "world_model_forward_calls" in metrics:
        if metrics["world_model_forward_calls"] != forward_calls:
            raise ValueError("Instrumented world-model calls do not match supported per-step model inputs")
        if metrics["world_model_forward_batch_elements"] != forward_calls:
            raise ValueError("Instrumented world-model batch elements do not reconcile with call count")
    if "action_model_forward_calls" in metrics:
        if metrics["action_model_forward_calls"] != action_calls:
            raise ValueError("Instrumented action-model calls do not match supported scheduled queries")
        if metrics["action_model_forward_batch_elements"] != action_calls:
            raise ValueError("Instrumented action-model batch elements do not reconcile with call count")
    if int(metrics["available_assessments"]) + int(metrics["unavailable_assessments"]) == 0:
        raise ValueError("No geometric assessment outcomes were recorded")
    if not g1_estimate_frames:
        raise ValueError("G1 predictions were not used in hidden-entity memory downstream")
    status = "PASS"
    result = {
        "schema_version": "football.sustained-replay-validation.0.6",
        "status": status,
        "batch_summary": str(summary_path),
        "batch_summary_sha256": file_digest(summary_path),
        "job_id": expected_job,
        "job_fingerprint": job["fingerprint"],
        "replay_path": str((folder / "replay.json").resolve()),
        "replay_sha256": file_digest(folder / "replay.json"),
        "duration": {
            "nominal_samples_per_actor": expected_samples,
            "nominal_sampling_seconds": 0.2,
            "source_time_samples": unique_frames,
            "elapsed_span_per_actor_s": per_actor,
            "unique_source_frames": unique_frames,
            "actor_steps": actor_steps,
            "actor_steps_are_not_unique_footage": True,
        },
        "source_frame_keys": sorted(f"{source_id}:{frame_index}" for source_id, frame_index in all_source_frames),
        "per_actor": per_actor,
        "model_usage": {
            "g1_forward_calls": forward_calls,
            "g1_batch_elements": forward_calls,
            "g1_hidden_estimate_outputs_used_in_memory": backend_occurrences["G1"],
            "actor_steps_with_g1_hidden_estimate": g1_estimate_frames,
            "fallback_outputs": metrics["fallbacks"],
            "fallbacks_by_backend": {
                key: value for key, value in backend_occurrences.items()
                if key in ("last_seen", "constant_velocity")
            },
            "action_forward_calls": action_calls,
            "action_batch_elements": action_calls,
            "forward_counters_source": (
                "ModelSession/ActionModelSession execution counters"
                if "world_model_forward_calls" in metrics else
                "Validated inference-path model-contract counter in the pre-instrumentation pilot"
            ),
            "scheduled_queries": queries,
            "not_queried_actor_steps": not_queried,
            "geometric_assessment_candidate_statuses": dict(assessment_statuses),
            "actor_identity_resets": identity_resets,
            "actor_missing_steps": actor_missing,
        },
        "metrics": metrics,
        "decision": "PASS_AND_ADVANCE" if status == "PASS" else "REPAIR_AND_RETEST",
        "next_stage": "Complete the remaining eleven sustained sequences; then run external-observation parity, true mid-sequence restart, and timing/exclusion reconciliations.",
    }
    write_json(output_path.resolve(), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--actors", nargs="+", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--unique-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.summary,
        expected_job=args.job,
        expected_actors=args.actors,
        expected_samples=args.samples,
        expected_unique_frames=args.unique_frames,
        output_path=args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "job_id": result["job_id"],
        "unique_source_frames": result["duration"]["unique_source_frames"],
        "actor_steps": result["duration"]["actor_steps"],
        "g1_forward_calls": result["model_usage"]["g1_forward_calls"],
        "action_forward_calls": result["model_usage"]["action_forward_calls"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
