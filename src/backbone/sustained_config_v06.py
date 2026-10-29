"""Choose fixed-coverage 120-second retrospective sequences deterministically."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from src.player_centric_dataset import ACTION_LABELS
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HALVES = (
    ("118577", "1st"),
    ("118577", "2nd"),
    ("128058", "1st"),
    ("128058", "2nd"),
)
TIME_BINS = ((0, 15), (15, 30), (30, 45))
CLASS_PRIORITY = ("HEADER", "CROSS", "SHOT", "DEFENSIVE_INTERVENTION", "HIGH_PASS", "DRIVE", "PASS")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _window_tracks(raw: dict[str, np.ndarray], start: int, samples: int) -> list[int]:
    indices = start + np.arange(samples, dtype=np.int64) * 5
    if indices[-1] >= len(raw["observed"]):
        return []
    valid_tracks = []
    for slot in range(raw["observed"].shape[1]):
        if not bool(raw["observed"][indices, slot].all()):
            continue
        segments = raw["segments"][indices, slot]
        if bool(np.all(segments == segments[0])):
            valid_tracks.append(slot)
    return valid_tracks


def _window_start(query_frame: int, frame_count: int, samples: int) -> int | None:
    start = query_frame - (samples // 2) * 5
    if start < 0:
        start = query_frame % 5
    last = start + (samples - 1) * 5
    if query_frame < start or (query_frame - start) % 5 or last >= frame_count:
        return None
    if not start <= query_frame <= last:
        return None
    return start


def build_config(
    index_path: Path,
    *,
    cache_root: Path,
    action_checkpoint: str,
    experiment: str = "outputs/dynamic_gnn_tactical_20261019",
    samples: int = 600,
) -> dict:
    if samples != 600:
        raise ValueError("The sustained integration gate is fixed to 600 samples at 5 Hz (120 nominal seconds)")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != "football.action-corpus.0.4" or index.get("status") != "COMPLETE":
        raise ValueError("Sustained selection requires the complete indexed v0.4 action corpus")
    shards = {
        (str(shard["match"]), shard["half"]): shard
        for shard in index["shards"]
        if shard.get("split") == "retrospective_evaluation"
    }
    if set(shards) != set(EXPECTED_HALVES):
        raise ValueError("Expected both halves of retrospective matches 118577 and 128058")

    cells: list[dict] = []
    for match, half in EXPECTED_HALVES:
        shard = shards[(match, half)]
        folder = Path(shard["folder"])
        for name, expected in shard["artifact_hashes"].items():
            artifact = folder / name
            if not artifact.is_file() or file_digest(artifact) != expected:
                raise ValueError(f"Indexed retrospective artifact changed: {artifact}")
        metadata = {row["sample_id"]: row for row in _read_jsonl(folder / "student_metadata.jsonl")}
        targets = _read_jsonl(folder / "optional_targets.jsonl")
        cache = cache_root / f"{match}_{half}.npz"
        with np.load(cache, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in ("observed", "segments")}
        for bin_start, bin_end in TIME_BINS:
            candidates_by_label: dict[str, list[dict]] = {label: [] for label in ACTION_LABELS}
            supported_count = 0
            for target in targets:
                label = target.get("mapped_action_label")
                if label not in candidates_by_label:
                    continue
                row = metadata.get(target["sample_id"])
                if row is None or row.get("split") != "retrospective_evaluation":
                    raise ValueError(f"Target has no indexed retrospective metadata: {target['sample_id']}")
                query_frame = int(row["query_frame"])
                minute = query_frame / 25.0 / 60.0
                if not bin_start <= minute < bin_end:
                    continue
                start = _window_start(query_frame, len(raw["observed"]), samples)
                if start is None:
                    continue
                covered = _window_tracks(raw, start, samples)
                actor = str(row["actor_track"])
                try:
                    actor_slot = int(actor.removeprefix("track_"))
                except ValueError:
                    continue
                if actor_slot not in covered or len(covered) < 2:
                    continue
                supported_count += 1
                candidates_by_label[label].append({
                    "target": target,
                    "metadata": row,
                    "query_frame": query_frame,
                    "start_frame": start,
                    "actor_slot": actor_slot,
                    "covered_slots": covered,
                })
            if not supported_count:
                raise ValueError(f"No fully covered event-conditioned candidate in {match} {half} {bin_start}-{bin_end}m")
            cells.append({
                "match": match,
                "half": half,
                "bin": [bin_start, bin_end],
                "candidates_by_label": candidates_by_label,
                "candidate_count": supported_count,
            })

    chosen: dict[int, dict] = {}
    class_cells: dict[str, int] = {}
    for label in CLASS_PRIORITY:
        choices = [
            (len(cell["candidates_by_label"][label]), position)
            for position, cell in enumerate(cells)
            if position not in chosen and cell["candidates_by_label"][label]
        ]
        if choices:
            _, position = min(choices)
            chosen[position] = cell_candidate(cells[position], label)
            class_cells[label] = position
    for position, cell in enumerate(cells):
        if position in chosen:
            continue
        labels = [label for label in CLASS_PRIORITY if cell["candidates_by_label"][label]]
        if not labels:
            raise ValueError(f"No labeled event available in {cell['match']} {cell['half']} {cell['bin']}")
        unused = [label for label in labels if label not in class_cells]
        label = unused[0] if unused else labels[0]
        chosen[position] = cell_candidate(cell, label)
        class_cells.setdefault(label, position)

    jobs = []
    selected_classes: Counter[str] = Counter()
    for position, cell in enumerate(cells):
        candidate = chosen[position]
        target, metadata = candidate["target"], candidate["metadata"]
        actor = str(metadata["actor_track"])
        actor_slot = candidate["actor_slot"]
        partner = next(slot for slot in candidate["covered_slots"] if slot != actor_slot)
        event_id = str(target["event_id"])
        identifier = f"sustained_{cell['match']}_{cell['half']}_{cell['bin'][0]:02d}m"
        label = str(target["mapped_action_label"])
        selected_classes[label] += 1
        jobs.append({
            "id": identifier,
            "source_adapter": "soccertrack_cache",
            "match": cell["match"],
            "half": cell["half"],
            "start_frame": candidate["start_frame"],
            "samples": samples,
            "actors": [actor, f"track_{partner}"],
            "intended_use": "retrospective_replay",
            "backend": "G1",
            "exo_p0": False,
            "experiment": experiment,
            "action_checkpoint": action_checkpoint,
            "device": "cuda",
            "selection": (
                f"Predeclared fixed temporal bin {cell['bin'][0]}-{cell['bin'][1]} minutes; "
                f"query actor and partner each have one stable observed cache segment at all {samples} "
                "sample times; deterministic rare-class/coverage tie-break; not selected by predictions."
            ),
            "observation": {"seed": 42, "fov": 120, "dropout": False},
            "passing_assessment": True,
            "mc_samples": 128,
            "action_queries": [{
                "frame_index": candidate["query_frame"],
                "actor_track": actor,
                "event_id": event_id,
                "annotation_index": int(target["annotation_index"]),
                "original_bas_label": target["original_bas_label"],
                "mapped_action_label": label,
                "mapping_provenance": target["mapping_provenance"],
                "conditioning": "event-conditioned sustained replay demonstration; actor and query time explicitly supplied",
            }],
        })
    if len(jobs) != 12:
        raise ValueError(f"Expected exactly twelve 120-second jobs, got {len(jobs)}")
    if set(selected_classes) != set(ACTION_LABELS):
        raise ValueError(f"Not every supported trained class was selected: {sorted(selected_classes)}")
    return {
        "schema_version": "backbone.batch.0.2",
        "viewer_job_ids": [jobs[0]["id"]],
        "defaults": {
            "source_adapter": "soccertrack_cache",
            "intended_use": "retrospective_replay",
            "backend": "G1",
            "exo_p0": False,
            "experiment": experiment,
            "action_checkpoint": action_checkpoint,
            "device": "cuda",
            "observation": {"seed": 42, "fov": 120, "dropout": False},
            "source_paths": {
                "tracking_cache": "outputs/trajectory_forecasting_20261016/cache",
                "associations": "outputs/trajectory_audit_20261016",
            },
            "passing_assessment": True,
            "mc_samples": 128,
        },
        "selection_provenance": {
            "source_index": str(index_path.resolve()),
            "source_index_sha256": file_digest(index_path),
            "retrospective_halves": [f"{match}_{half}" for match, half in EXPECTED_HALVES],
            "sequence_duration": "600 source samples at 5Hz; 119.8s between first and last inclusive sample",
            "temporal_bins_per_half_minutes": [list(value) for value in TIME_BINS],
            "candidate_counts_per_bin": [
                {"match": cell["match"], "half": cell["half"], "bin_minutes": cell["bin"],
                 "fully_covered_action_events": cell["candidate_count"]}
                for cell in cells
            ],
            "selected_classes": dict(selected_classes),
            "selection_policy": (
                "Coverage from observed source slots and stable segment IDs plus existing event-label availability; "
                "chronological source strata and deterministic rare-class coverage; never model prediction correctness."
            ),
            "actor_policy": "Event actor and second actor must both be observed at every selected source sample with unchanged source segment.",
        },
        "jobs": jobs,
    }


def cell_candidate(cell: dict, label: str) -> dict:
    candidates = cell["candidates_by_label"][label]
    return min(candidates, key=lambda item: (item["query_frame"], item["target"]["event_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/trajectory_forecasting_20261016/cache"))
    parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.index if args.index.is_absolute() else ROOT / args.index
    cache_root = args.cache_root if args.cache_root.is_absolute() else ROOT / args.cache_root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = build_config(
        index_path.resolve(),
        cache_root=cache_root.resolve(),
        action_checkpoint=args.action_checkpoint,
    )
    write_json(output.resolve(), result)
    print(json.dumps({
        "status": "READY",
        "jobs": len(result["jobs"]),
        "samples_per_job": result["jobs"][0]["samples"],
        "supported_classes": sorted(result["selection_provenance"]["selected_classes"]),
        "output": str(output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
