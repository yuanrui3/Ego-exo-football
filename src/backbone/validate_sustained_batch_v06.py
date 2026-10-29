"""Validate all twelve sustained replay sequences from their produced artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from src.backbone.validate_sustained_v06 import validate as validate_sequence
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MATCH_HALVES = {
    ("118577", "1st"), ("118577", "2nd"),
    ("128058", "1st"), ("128058", "2nd"),
}


def validate_batch(config_path: Path, summary_path: Path, output_path: Path) -> dict:
    config_path, summary_path = config_path.resolve(), summary_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise ValueError(f"Sustained batch not complete: {summary.get('status')}")
    specifications = {job["id"]: job for job in config["jobs"]}
    completed = [job for job in summary["jobs"] if job.get("status") == "COMPLETE"]
    failed = [job for job in summary["jobs"] if job.get("status") == "FAILED"]
    if failed or len(completed) != 12 or len(specifications) != 12:
        raise ValueError(f"Expected 12 completed jobs and no failures; got {len(completed)} completed, {len(failed)} failed")

    reports, source_frame_keys = [], set()
    jobs_by_half: Counter[tuple[str, str]] = Counter()
    classes: Counter[str] = Counter()
    for job in completed:
        spec = specifications.get(job["id"])
        if spec is None:
            raise ValueError(f"Completed job absent from config: {job['id']}")
        jobs_by_half[(str(spec["match"]), spec["half"])] += 1
        action_label = spec["action_queries"][0]["mapped_action_label"]
        classes[action_label] += 1
        report_path = output_path.parent / "per_sequence" / f"{job['id']}.json"
        report = validate_sequence(
            summary_path,
            expected_job=job["id"],
            expected_actors=spec["actors"],
            expected_samples=600,
            expected_unique_frames=600,
            output_path=report_path,
        )
        keys = set(report["source_frame_keys"])
        overlap = source_frame_keys & keys
        if overlap:
            raise ValueError(f"Source intervals overlap; repeated source footage: {sorted(overlap)[:3]}")
        source_frame_keys.update(keys)
        reports.append(report)
    if set(jobs_by_half) != EXPECTED_MATCH_HALVES or any(count != 3 for count in jobs_by_half.values()):
        raise ValueError(f"Expected three intervals per retrospective half: {dict(jobs_by_half)}")
    trained_classes = set(config["selection_provenance"]["selected_classes"])
    if set(classes) != trained_classes:
        raise ValueError(f"Selected annotations do not match declared available classes: {dict(classes)}")
    actor_steps = sum(report["duration"]["actor_steps"] for report in reports)
    if len(source_frame_keys) != 7200 or actor_steps != 14400:
        raise ValueError(
            f"Expected 7,200 unique time samples and 14,400 actor-steps; got {len(source_frame_keys)} and {actor_steps}"
        )
    total = summary["totals"]
    model_calls = sum(report["model_usage"]["g1_forward_calls"] for report in reports)
    action_calls = sum(report["model_usage"]["action_forward_calls"] for report in reports)
    if total.get("world_model_forward_calls") != model_calls:
        raise ValueError("Batch G1 forward counter does not reconcile with per-sequence inference rows")
    if total.get("action_model_forward_calls") != action_calls:
        raise ValueError("Batch action forward counter does not reconcile with scheduled action outputs")
    result = {
        "schema_version": "football.sustained-batch-validation.0.6",
        "status": "PASS",
        "config": str(config_path),
        "config_sha256": file_digest(config_path),
        "batch_summary": str(summary_path),
        "batch_summary_sha256": file_digest(summary_path),
        "jobs": len(reports),
        "jobs_per_half": {f"{match}_{half}": jobs_by_half[(match, half)] for match, half in sorted(jobs_by_half)},
        "unique_source_frames": len(source_frame_keys),
        "unique_source_minutes": len(source_frame_keys) / 5.0 / 60.0,
        "actor_steps": actor_steps,
        "actors_per_sequence": 2,
        "nominal_sequence_samples_per_actor": 600,
        "actual_elapsed_span_per_sequence_seconds": {
            report["job_id"]: report["duration"]["elapsed_span_per_actor_s"]
            for report in reports
        },
        "class_query_coverage": dict(classes),
        "model_usage": {
            "g1_forward_calls": model_calls,
            "g1_batch_elements": sum(report["model_usage"]["g1_batch_elements"] for report in reports),
            "g1_hidden_estimates_used_by_memory": sum(
                report["model_usage"]["g1_hidden_estimate_outputs_used_in_memory"] for report in reports
            ),
            "fallback_outputs": sum(report["model_usage"]["fallback_outputs"] for report in reports),
            "action_forward_calls": action_calls,
            "action_batch_elements": sum(report["model_usage"]["action_batch_elements"] for report in reports),
            "not_queried_actor_steps": sum(
                report["model_usage"]["not_queried_actor_steps"] for report in reports
            ),
            "geometric_candidate_statuses": {
                key: sum(report["model_usage"]["geometric_assessment_candidate_statuses"].get(key, 0)
                         for report in reports)
                for key in ("AVAILABLE", "UNAVAILABLE")
            },
        },
        "batch_execution_seconds": summary["execution_seconds"],
        "batch_output_bytes": total["output_bytes"],
        "per_sequence_validation": [
            {
                "job_id": report["job_id"],
                "path": str((output_path.parent / "per_sequence" / f"{report['job_id']}.json").resolve()),
                "first_timestamp_s": min(v["first_timestamp_s"] for v in report["per_actor"].values()),
                "last_timestamp_s": max(v["last_timestamp_s"] for v in report["per_actor"].values()),
                "max_actor_gap_s": max(v["max_gap_s"] for v in report["per_actor"].values()),
                "actor_missing_steps": report["model_usage"]["actor_missing_steps"],
                "identity_resets": report["model_usage"]["actor_identity_resets"],
            }
            for report in reports
        ],
        "decision": "PASS_AND_ADVANCE",
        "next_stage": "Run true interrupted mid-sequence recovery and external JSONL parity; then complete remaining Stage A timing and legacy-count reconciliation.",
    }
    write_json(output_path.resolve(), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_batch(args.config, args.summary, args.output)
    print(json.dumps({
        "status": result["status"],
        "jobs": result["jobs"],
        "unique_source_minutes": result["unique_source_minutes"],
        "actor_steps": result["actor_steps"],
        "g1_forward_calls": result["model_usage"]["g1_forward_calls"],
        "action_forward_calls": result["model_usage"]["action_forward_calls"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
