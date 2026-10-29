"""Build the event-time coverage waterfall from source annotations and corpus index."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path

from src.backbone.corpus import _class_for_label, _event_rows
from src.backbone.event_time import resolve_event_time
from src.backbone.source import configured_path
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_report(config_path: Path, index_path: Path, *, root: Path = ROOT) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("status") != "COMPLETE":
        raise ValueError("Timing diagnosis requires a complete indexed v0.5 corpus")
    paths = config.get("source_paths", {})
    event_root = configured_path(root, paths.get("event_data"), Path("data"))
    cache_root = configured_path(
        root, paths.get("tracking_cache"),
        Path("outputs") / "trajectory_forecasting_20261016" / "cache",
    )
    total = Counter()
    per_source: dict[str, dict] = {}
    per_split_class: dict[str, Counter] = defaultdict(Counter)
    for shard in index["shards"]:
        match, half, split = str(shard["match"]), shard["half"], shard["split"]
        source_key = f"{match}_{half}"
        folder = Path(shard["folder"])
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        cache_meta_path = cache_root / f"{source_key}.json"
        cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        bas_path = event_root / f"{match}_12_class_events.json"
        event_file = json.loads(bas_path.read_text(encoding="utf-8"))
        eligibility = {
            row["event_id"]: row for row in _read_jsonl(folder / "event_eligibility.jsonl")
        }
        grouped = {
            "raw_events": 0,
            "time_status": Counter(),
            "time_reason": Counter(),
            "new_stable_frame_delta_from_nearest": Counter(),
            "legacy_stable_frame_delta_from_floor": Counter(),
            "legacy_alignment_rejections": 0,
            "eligible": 0,
            "exclusions": Counter(),
            "classes": {},
        }
        class_groups: dict[str, Counter] = defaultdict(Counter)
        for item in _event_rows(event_file, match, half):
            event = item["event"]
            annotation_index = int(item["annotation_index"])
            stable_frame = int(event.get("frame", -1))
            event_id = f"{match}_p{1 if half == '1st' else 2}_f{stable_frame}_{annotation_index}"
            label = _class_for_label(event.get("label"))
            label_group = label or "UNSUPPORTED_OR_CONTEXT"
            result = resolve_event_time(
                event.get("position"),
                event.get("frame"),
                clock=event_file.get("clock"),
                gsr_info=cache_meta.get("gsr_info", {}),
                source_frame_count=int(summary["frames_available"]),
            )
            old_delta = None
            try:
                position = Decimal(str(event["position"]))
                old_floor = int((position * Decimal(25) / Decimal(1000)).to_integral_value(
                    rounding=ROUND_FLOOR,
                ))
                old_delta = stable_frame - old_floor
            except (KeyError, ArithmeticError, ValueError):
                pass
            eligibility_row = eligibility.get(event_id)
            if eligibility_row is None:
                raise ValueError(f"Corpus eligibility record missing stable event ID {event_id}")
            exclusion = eligibility_row.get("exclusion_reason")
            grouped["raw_events"] += 1
            grouped["time_status"][result.status] += 1
            grouped["time_reason"][result.reason or "supported"] += 1
            grouped["new_stable_frame_delta_from_nearest"][str(
                result.stable_frame_delta_from_nearest
            )] += 1
            if old_delta is not None:
                grouped["legacy_stable_frame_delta_from_floor"][str(old_delta)] += 1
                grouped["legacy_alignment_rejections"] += int(abs(old_delta) > 1)
            grouped["eligible"] += int(eligibility_row.get("eligibility") == "ELIGIBLE")
            grouped["exclusions"][exclusion or "eligible"] += int(exclusion is not None)
            class_groups[label_group]["raw_events"] += 1
            class_groups[label_group][f"time:{result.status}"] += 1
            class_groups[label_group][f"eligible:{split}"] += int(
                eligibility_row.get("eligibility") == "ELIGIBLE"
            )
            class_groups[label_group][f"excluded:{exclusion or 'eligible'}"] += int(
                exclusion is not None
            )
            total["raw_events"] += 1
            total[f"time:{result.status}"] += 1
            total[f"legacy_alignment_rejected:{split}"] += int(old_delta is not None and abs(old_delta) > 1)
            total[f"eligible:{split}"] += int(eligibility_row.get("eligibility") == "ELIGIBLE")
            per_split_class[f"{split}:{label_group}"]["raw_events"] += 1
            per_split_class[f"{split}:{label_group}"]["eligible"] += int(
                eligibility_row.get("eligibility") == "ELIGIBLE"
            )
        grouped["classes"] = {
            label: dict(counts) for label, counts in sorted(class_groups.items())
        }
        per_source[source_key] = {
            "split": split,
            "bas_sha256": file_digest(bas_path),
            "cache_metadata_sha256": file_digest(cache_meta_path),
            "summary": {
                "raw_events": grouped["raw_events"],
                "eligible_events": grouped["eligible"],
                "time_status": dict(grouped["time_status"]),
                "time_reason": dict(grouped["time_reason"]),
                "new_stable_frame_delta_from_nearest": dict(
                    grouped["new_stable_frame_delta_from_nearest"]
                ),
                "legacy_stable_frame_delta_from_floor": dict(
                    grouped["legacy_stable_frame_delta_from_floor"]
                ),
                "legacy_alignment_rejections": grouped["legacy_alignment_rejections"],
                "exclusions": dict(grouped["exclusions"]),
                "classes": grouped["classes"],
            },
        }
    return {
        "schema_version": "football.event-time-coverage.0.5",
        "status": "COMPLETE",
        "timing_policy": (
            "Causal query is floor(position_ms * native_fps / 1000); stable BAS frame independently "
            "checks nearest-rounded source time and is never used to select a future observation."
        ),
        "physical_contact_certified": False,
        "legacy_rule": "Excluded events when abs(stable BAS frame - floor(position_ms * 25 / 1000)) > 1.",
        "source_paths": {
            "corpus_index": str(index_path.resolve()),
            "corpus_index_sha256": file_digest(index_path),
            "config": str(config_path.resolve()),
            "config_sha256": file_digest(config_path),
        },
        "totals": {
            "raw_events": total["raw_events"],
            "time_supported": total["time:supported"],
            "time_unsupported": total["time:unsupported"],
            "eligible_fitting": total["eligible:fitting"],
            "eligible_development": total["eligible:development"],
            "eligible_retrospective_evaluation": total["eligible:retrospective_evaluation"],
            "legacy_frame_alignment_would_have_rejected": {
                split: total[f"legacy_alignment_rejected:{split}"]
                for split in ("fitting", "development", "retrospective_evaluation")
            },
        },
        "per_split_class": {
            key: dict(counts) for key, counts in sorted(per_split_class.items())
        },
        "per_match_half": per_source,
        "interpretation": (
            "BAS frame rounding is treated as an independent consistency check. A one-frame stable-ID "
            "offset is accepted only relative to nearest-rounded time; the query stays floor-rounded and causal."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/backbone_v0_5.json"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    index_path = args.index if args.index.is_absolute() else root / args.index
    output_path = args.output if args.output.is_absolute() else root / args.output
    report = build_report(config_path.resolve(), index_path.resolve(), root=root)
    write_json(output_path, report)
    print(json.dumps({"status": report["status"], "totals": report["totals"],
                      "output": str(output_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
