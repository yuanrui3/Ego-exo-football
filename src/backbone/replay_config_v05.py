"""Select a deterministic 12-segment retrospective batch from indexed labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from src.player_centric_dataset import ACTION_LABELS
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MATCHES = {"118577", "128058"}
HALF_ORDER = ("118577_1st", "118577_2nd", "128058_1st", "128058_2nd")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_config(index_path: Path, action_checkpoint: str) -> dict:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != "football.action-corpus.0.4" or index.get("status") != "COMPLETE":
        raise ValueError("Retrospective batch selection requires a complete indexed action corpus")
    shards = [
        shard for shard in index["shards"]
        if shard.get("split") == "retrospective_evaluation"
    ]
    if {str(shard["match"]) for shard in shards} != EXPECTED_MATCHES:
        raise ValueError("Retrospective shards must contain only matches 118577 and 128058")
    by_half: dict[str, list[dict]] = defaultdict(list)
    for shard in shards:
        key = f"{shard['match']}_{shard['half']}"
        if key not in HALF_ORDER:
            raise ValueError(f"Unexpected retrospective source half: {key}")
        folder = Path(shard["folder"])
        for name, expected in shard["artifact_hashes"].items():
            path = folder / name
            if not path.is_file() or file_digest(path) != expected:
                raise ValueError(f"Indexed retrospective artifact changed: {path}")
        metadata = {
            row["sample_id"]: row for row in _read_jsonl(folder / "student_metadata.jsonl")
        }
        targets = _read_jsonl(folder / "optional_targets.jsonl")
        if len(metadata) != shard["samples"] or len(targets) != shard["samples"]:
            raise ValueError(f"Keyed retrospective annotations incomplete in {key}")
        for target in targets:
            sample_id = target["sample_id"]
            row = metadata.get(sample_id)
            if row is None or row.get("split") != "retrospective_evaluation":
                raise ValueError(f"Target has no matching evaluation input metadata: {sample_id}")
            if target["source"]["match"] != str(shard["match"]):
                raise ValueError(f"Target match provenance mismatch: {sample_id}")
            label = target["mapped_action_label"]
            if label not in ACTION_LABELS:
                raise ValueError(f"Unsupported mapped action class in target: {label}")
            by_half[key].append({"target": target, "metadata": row})
        by_half[key].sort(key=lambda item: (
            int(item["metadata"]["query_frame"]),
            int(item["target"]["annotation_index"]),
            item["target"]["event_id"],
        ))
    if set(by_half) != set(HALF_ORDER):
        raise ValueError("Retrospective index must include both halves of both configured matches")

    selected: dict[str, list[dict]] = {key: [] for key in HALF_ORDER}
    selected_ids: set[str] = set()
    supported_classes = [
        label for label in ACTION_LABELS
        if any(row["target"]["mapped_action_label"] == label
               for rows in by_half.values() for row in rows)
    ]
    for label in supported_classes:
        candidates = {
            key: next((row for row in by_half[key]
                       if row["target"]["mapped_action_label"] == label
                       and row["target"]["sample_id"] not in selected_ids), None)
            for key in HALF_ORDER
        }
        available = [key for key in HALF_ORDER if candidates[key] is not None and len(selected[key]) < 3]
        if not available:
            continue
        key = min(available, key=lambda item: (len(selected[item]), HALF_ORDER.index(item)))
        row = candidates[key]
        selected[key].append(row)
        selected_ids.add(row["target"]["sample_id"])

    for key in HALF_ORDER:
        for row in by_half[key]:
            if len(selected[key]) >= 3:
                break
            sample_id = row["target"]["sample_id"]
            if sample_id not in selected_ids:
                selected[key].append(row)
                selected_ids.add(sample_id)
    if any(len(selected[key]) != 3 for key in HALF_ORDER):
        raise ValueError("Could not select exactly three eligible retrospective events per source half")

    jobs = []
    for key in HALF_ORDER:
        match, half = key.rsplit("_", 1)
        for choice_index, row in enumerate(sorted(selected[key], key=lambda item: (
            int(item["metadata"]["query_frame"]), int(item["target"]["annotation_index"])
        )), start=1):
            metadata, target = row["metadata"], row["target"]
            query_frame = int(metadata["query_frame"])
            actor = str(metadata["actor_track"])
            event_id = str(target["event_id"])
            jobs.append({
                "id": f"retro_{match}_{half}_{query_frame}_{choice_index}",
                "source_adapter": "soccertrack_cache",
                "match": match,
                "half": half,
                "start_frame": query_frame - 50,
                "samples": 11,
                "actors": [actor],
                "intended_use": "retrospective_replay",
                "backend": "G1",
                "exo_p0": False,
                "experiment": "outputs/dynamic_gnn_tactical_20261019",
                "action_checkpoint": action_checkpoint,
                "selection": (
                    "Deterministic label-availability, source-half coverage and chronological selection; "
                    "never conditioned on model predictions."
                ),
                "observation": {"seed": 42, "fov": 120, "dropout": False},
                "passing_assessment": True,
                "action_queries": [{
                    "frame_index": query_frame,
                    "actor_track": actor,
                    "event_id": event_id,
                    "annotation_index": int(target["annotation_index"]),
                    "original_bas_label": target["original_bas_label"],
                    "mapped_action_label": target["mapped_action_label"],
                    "mapping_provenance": target["mapping_provenance"],
                    "conditioning": "event-conditioned retrospective demonstration; actor and query time explicitly supplied",
                }],
            })
    counts = {
        key: sorted(row["target"]["mapped_action_label"] for row in selected[key])
        for key in HALF_ORDER
    }
    return {
        "schema_version": "backbone.batch.0.2",
        "defaults": {
            "source_adapter": "soccertrack_cache",
            "intended_use": "retrospective_replay",
            "backend": "G1",
            "exo_p0": False,
            "experiment": "outputs/dynamic_gnn_tactical_20261019",
            "observation": {"seed": 42, "fov": 120, "dropout": False},
            "source_paths": {
                "tracking_cache": "outputs/trajectory_forecasting_20261016/cache",
                "associations": "outputs/trajectory_audit_20261016",
            },
        },
        "selection_provenance": {
            "training_index": str(index_path.resolve()),
            "training_index_sha256": file_digest(index_path),
            "selected_segments": len(jobs),
            "segments_per_half": {key: 3 for key in HALF_ORDER},
            "selected_classes_by_half": counts,
            "all_available_action_classes": supported_classes,
            "selection_policy": "Deterministic label availability and chronological source coverage; no prediction-based selection.",
            "annotation_separation": "Stable event ID and annotation index are displayed evaluation-only and never passed to the observation encoder.",
        },
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.index if args.index.is_absolute() else ROOT / args.index
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    result = build_config(index_path.resolve(), args.action_checkpoint)
    write_json(output_path, result)
    print(json.dumps({"status": "READY", "segments": len(result["jobs"]),
                      "classes": result["selection_provenance"]["all_available_action_classes"],
                      "output": str(output_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
