from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(corpus_root: Path) -> dict:
    index = json.loads((corpus_root / "training_index.json").read_text(encoding="utf-8"))
    totals: Counter = Counter()
    split_counts: dict[str, Counter] = defaultdict(Counter)
    causal_examples = 0
    shard_count = 0
    errors: list[str] = []

    for shard in index["shards"]:
        shard_count += 1
        folder = Path(shard["folder"])
        eligibility = _jsonl(folder / "event_eligibility.jsonl")
        students = _jsonl(folder / "student_metadata.jsonl")
        event_ids = [row["event_id"] for row in eligibility]
        if len(event_ids) != len(set(event_ids)):
            errors.append(f"Duplicate event IDs in {folder}")

        by_event = {row["event_id"]: row for row in eligibility}
        if len(students) != shard["samples"]:
            errors.append(f"Student row count mismatch in {folder}")

        for row in eligibility:
            split = row["split"]
            if split != shard["split"]:
                errors.append(f"Split mismatch for {row['event_id']}")
            totals["raw_events"] += 1
            split_counts[split]["raw_events"] += 1
            supported = row["mapped_action_label"] is not None
            totals["supported_labeled_events"] += supported
            split_counts[split]["supported_labeled_events"] += supported
            position = row.get("position_ms")
            if position is not None:
                floor_frame = int(
                    (Decimal(str(position)) * Decimal(25) / Decimal(1000))
                    .to_integral_value(rounding=ROUND_FLOOR)
                )
                legacy_reject = abs(int(row["stable_id_frame"]) - floor_frame) > 1
                totals["legacy_alignment_rejects_all"] += legacy_reject
                split_counts[split]["legacy_alignment_rejects_all"] += legacy_reject
                if supported:
                    totals["legacy_alignment_rejects_supported"] += legacy_reject
                    split_counts[split]["legacy_alignment_rejects_supported"] += legacy_reject
                else:
                    totals["legacy_alignment_rejects_unsupported"] += legacy_reject
                    split_counts[split]["legacy_alignment_rejects_unsupported"] += legacy_reject

        for student in students:
            event = by_event.get(student["event_id"])
            if event is None or event["eligibility"] != "ELIGIBLE":
                errors.append(f"Student row has no eligible event: {student['event_id']}")
                continue
            if student["split"] != shard["split"]:
                errors.append(f"Student split mismatch: {student['event_id']}")
            frames = student["input_frame_indices"]
            query = int(student["query_frame"])
            resolution = event.get("time_resolution", {})
            query_time_ms = Decimal(query) * Decimal(1000) / Decimal(25)
            position_ms = Decimal(str(resolution.get("position_ms", "NaN")))
            if (
                len(frames) != 11
                or frames[-1] != query
                or resolution.get("status") != "supported"
                or query != resolution.get("native_zero_based_frame")
                or query_time_ms > position_ms
                or any(int(frame) > query for frame in frames)
                or any(int(b) - int(a) != 5 for a, b in zip(frames, frames[1:]))
            ):
                errors.append(f"Non-causal, off-grid, or unsupported history: {student['event_id']}")
            causal_examples += 1

    checks = {
        "legacy_reject_population_identity": (
            totals["legacy_alignment_rejects_all"]
            == totals["legacy_alignment_rejects_supported"]
            + totals["legacy_alignment_rejects_unsupported"]
        ),
        "causal_history_examples_match_index": causal_examples
        == sum(shard["samples"] for shard in index["shards"]),
        "legacy_counts_reconcile_to_prior_acceptance": (
            totals["legacy_alignment_rejects_supported"] == 7433
            and totals["legacy_alignment_rejects_unsupported"] == 490
            and totals["legacy_alignment_rejects_all"] == 7923
        ),
    }
    return {
        "schema_version": "football.backbone-v0.6-timing-reconciliation.1",
        "status": "PASS" if all(checks.values()) and not errors else "FAIL",
        "corpus_root": str(corpus_root.resolve()),
        "shards": shard_count,
        "totals": dict(sorted(totals.items())),
        "by_split": {
            key: dict(sorted(value.items()))
            for key, value in sorted(split_counts.items())
        },
        "causal_history_examples": causal_examples,
        "checks": checks,
        "errors": errors,
        "interpretation": (
            "The 7,923 legacy frame-delta rejections count all raw events. "
            "The prior 7,433 acceptance count applied only after unsupported/context "
            "labels were excluded; 490 such events also meet the legacy rejection rule."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("outputs/backbone_v0_5_20261020/corpus"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/backbone_v0_6_20261021/stage_a/timing_validation.json"),
    )
    args = parser.parse_args()
    report = validate(args.corpus_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
