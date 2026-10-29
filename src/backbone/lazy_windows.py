"""Fitting-only, target-free actor/time indices referencing shared verified sources."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from src.backbone.corpus import _write_npz
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.source import SoccerTrackSource
from src.backbone.train_action import FIT_MATCHES
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "outputs" / "backbone_v0_5_20261020" / "corpus" / "training_index.json"


def select_windows(native_frames: int, slots: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    if native_frames <= 75 or slots < 1 or count < 1:
        raise ValueError("Positive windows require 3s causal source coverage and actor slots")
    queries = np.arange(75, native_frames, 25, dtype=np.int32)
    available = len(queries) * slots
    positions = np.linspace(0, available - 1, min(count, available), dtype=np.int64)
    return queries[positions // slots], (positions % slots).astype(np.int16)


def build(output: Path, limit: int = 100000) -> dict:
    if not 1 <= limit <= 100000:
        raise ValueError("Lazy index limit must be 1..100000")
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    shards = sorted((s for s in index["shards"] if s["split"] == "fitting"),
                    key=lambda s: (s["match"], s["half"]))
    if {s["match"] for s in shards} != FIT_MATCHES or len(shards) != 12:
        raise ValueError("Expected the twelve current indexed fitting halves")
    sources, all_query, all_actor, all_source, all_quality = [], [], [], [], []
    for number, shard in enumerate(shards):
        source = SoccerTrackSource(ROOT, shard["match"], shard["half"])
        allocation = limit // len(shards) + int(number < limit % len(shards))
        if not allocation:
            continue
        query, actor = select_windows(len(source.raw["xy"]), source.raw["xy"].shape[1], allocation)
        observed = source.raw["observed"][query, actor]
        team = source.raw["teams"][query, actor]
        role = source.raw["roles"][query, actor]
        quality = (observed.astype(np.uint8) | (np.isin(team, (1, 2)).astype(np.uint8) << 1)
                   | (np.isin(role, (0, 1)).astype(np.uint8) << 2))
        all_query.append(query)
        all_actor.append(actor)
        all_source.append(np.full(len(query), number, np.int16))
        all_quality.append(quality)
        sources.append({**source.provenance, "source_index": number, "native_frames": len(source.raw["xy"]),
                        "actor_slots": source.raw["xy"].shape[1], "windows": len(query),
                        "indexed_shard_fingerprint": shard["fingerprint"]})
    _write_npz(output / "windows.npz", source_index=np.concatenate(all_source),
               actor_slot=np.concatenate(all_actor), query_frame=np.concatenate(all_query),
               quality_bits=np.concatenate(all_quality),
               temporal_group=np.concatenate(all_query) // (25 * 60))
    manifest = {
        "schema_version": "football.lazy-unlabeled-windows.0.1", "status": "COMPLETE",
        "windows": sum(map(len, all_query)), "index_sha256": file_digest(INDEX),
        "windows_sha256": file_digest(output / "windows.npz"), "sources": sources,
        "permitted_matches": sorted(FIT_MATCHES), "semantic_labels": "ABSENT; no NO_ACTION or off-ball negatives",
        "selection": "Systematic chronological actor/time sampling, equal per indexed half, independent of events",
        "timing": "11 input samples every five native frames ending at query; five earlier causal heading warmups",
        "quality_bits": {"1": "actor observed at query", "2": "team known", "4": "role known"},
        "missingness_policy": "Missing actors and metadata retained; downstream inference must abstain/fallback",
        "temporal_grouping": "source_index plus temporal_group (one-minute source bin), NOT independent matches",
        "student_access": "LazyWindowDataset.observations: simulated partial observations only",
        "privileged_access": "LazyWindowDataset.privileged_window: explicit separate supervision-only accessor",
        "shared_arrays": "Immutable original verified caches, referenced by path/hash; no duplicated dense tensors",
    }
    write_json(output / "manifest.json", manifest)
    return validate(output)


class LazyWindowDataset:
    def __init__(self, output: Path):
        self.manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if file_digest(output / "windows.npz") != self.manifest["windows_sha256"]:
            raise ValueError("Lazy window index hash mismatch")
        with np.load(output / "windows.npz", allow_pickle=False) as saved:
            self.windows = {key: saved[key] for key in saved.files}
        self.sources = {row["source_index"]: row for row in self.manifest["sources"]}
        self.cache: OrderedDict[int, SoccerTrackSource] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows["query_frame"])

    def _window(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError("Lazy window outside index")
        source_id = int(self.windows["source_index"][index])
        if source_id not in self.cache:
            row = self.sources[source_id]
            source = SoccerTrackSource(ROOT, row["match"], row["half"])
            if source.provenance != {key: row[key] for key in source.provenance}:
                raise ValueError("Shared source provenance changed")
            self.cache[source_id] = source
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(source_id)
        return (self.cache[source_id], int(self.windows["query_frame"][index]),
                int(self.windows["actor_slot"][index]))

    def observations(self, index: int):
        source, query, slot = self._window(index)
        provider = SyntheticObservationProvider(f"track_{slot}", dropout=False)
        return tuple(provider.observe(source.frame(frame)) for frame in range(query - 75, query + 1, 5))

    def privileged_window(self, index: int) -> dict:
        source, query, _ = self._window(index)
        frames = np.arange(query - 50, query + 1, 5)
        return {"role": "PRIVILEGED_SUPERVISION_ONLY", "native_frames": frames,
                "xy": source.raw["xy"][frames].copy(), "observed": source.raw["observed"][frames].copy()}


def validate(output: Path) -> dict:
    dataset = LazyWindowDataset(output)
    windows = dataset.windows
    unique = set()
    counts = {}
    for source_id, source_row in dataset.sources.items():
        if source_row["match"] not in FIT_MATCHES:
            raise ValueError("Non-fitting lazy source")
        selected = windows["source_index"] == source_id
        query, actor = windows["query_frame"][selected], windows["actor_slot"][selected]
        if (np.any(query < 75) or np.any(query >= source_row["native_frames"])
                or np.any(actor < 0) or np.any(actor >= source_row["actor_slots"])
                or np.any((query - 75) % 25)):
            raise ValueError("Out-of-source or noncausal lazy window")
        keys = {(int(source_id), int(a), int(q)) for a, q in zip(actor, query)}
        if len(keys) != len(query) or unique & keys:
            raise ValueError("Duplicated lazy window")
        unique |= keys
        counts[f"{source_row['match']}_{source_row['half']}"] = len(keys)
        number = int(np.flatnonzero(selected)[len(query) // 2])
        observations = dataset.observations(number)
        teacher = dataset.privileged_window(number)
        if (observations[-1].index != windows["query_frame"][number]
                or teacher["native_frames"][-1] != observations[-1].index
                or len(observations) != 16 or len(dataset.cache) > 2):
            raise ValueError("Lazy materialization/causality failure")
    if len(unique) != dataset.manifest["windows"]:
        raise ValueError("Lazy index declared count mismatch")
    result = {"status": "PASS", "distinct_windows": len(unique), "per_half": counts,
              "no_semantic_targets": True, "dense_student_tensors_stored": False,
              "verified_causal_materialization_halves": len(counts),
              "index_bytes": (output / "windows.npz").stat().st_size,
              "maximum_cached_sources": 2,
              "unknown_or_missing_actor_metadata_windows": int(np.sum(windows["quality_bits"] != 7))}
    write_json(output / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build", "validate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.limit) if args.phase == "build" else validate(args.output), indent=2))


if __name__ == "__main__":
    main()
