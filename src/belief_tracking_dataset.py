"""Isolated continuous tracking caches; slots are assigned by first sight, not roster IDs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.belief_observation import observation_window
from src.player_centric_dataset import stable_seed
from src.trajectory_dataset import TRAIN_MATCHES, VALIDATION_MATCHES, check_deadline, file_digest, write_json


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def build_continuous(root, output, deadline, stride=25, views=2, limit=None):
    folder = output / "continuous"
    folder.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for match in (*TRAIN_MATCHES, *VALIDATION_MATCHES):
        for half in ("1st", "2nd"):
            check_deadline(deadline)
            name = f"{match}_{half}"
            destination = folder / name
            manifest = destination / "manifest.json"
            source = root / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{name}.npz"
            digest = file_digest(source)
            verified = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
            if verified["cache_sha256"] != digest:
                raise ValueError(f"Tracking cache checksum differs from its verified provenance: {name}")
            if manifest.exists():
                saved = json.loads(manifest.read_text(encoding="utf-8"))
                if saved["source_sha256"] != digest or saved["stride"] != stride or saved["views"] != views:
                    raise ValueError(f"Continuous cache provenance changed: {name}")
                summaries[name] = saved
                continue
            raw = load_npz(source)
            records, anchors, exclusions = [], [], Counter()
            for anchor in range(150, len(raw["xy"]) - 75, stride):
                if anchor % 250 == 0:
                    check_deadline(deadline)
                current = np.flatnonzero(raw["observed"][anchor])
                rng = np.random.default_rng(stable_seed(f"{name}:{anchor}", 42))
                actor_order = rng.permutation(current)
                successes = 0
                for actor in actor_order:
                    seed = stable_seed(f"{name}:{anchor}:{actor}", 42)
                    try:
                        item = observation_window(raw, anchor, int(actor), seed)
                    except ValueError as error:
                        exclusions[str(error)] += 1
                        continue
                    item.pop("heading_reliability")
                    records.append(item)
                    anchors.append(anchor)
                    successes += 1
                    if successes >= views:
                        break
                if limit and len(records) >= limit:
                    break
            if not records:
                raise RuntimeError(f"No causal continuous examples for {name}")
            destination.mkdir(parents=True, exist_ok=True)
            for key in records[0]:
                array = np.stack([r[key] for r in records])
                if array.dtype == np.float32:
                    array = array.astype(np.float16)
                np.save(destination / f"{key}.npy", array, allow_pickle=False)
            np.save(destination / "anchor.npy", np.asarray(anchors, np.int32), allow_pickle=False)
            summary = {
                "match": match, "half": half, "examples": len(records), "exclusions": dict(exclusions),
                "source_sha256": digest, "stride": stride, "views": views, "history_frames": 150,
                "horizons": [10, 25, 50, 75], "schema": 1,
                "features": "causal first-sight anonymous slots; never-seen identities only aggregate occupancy targets",
                "input_fields": ["x", "mask", "visible", "actor_xy", "last_xy"],
                "target_only_fields": ["occupancy", "hidden_count", "position", "position_mask", "future", "future_mask"],
            }
            write_json(manifest, summary)
            summaries[name] = summary
            del records, raw
    write_json(output / "continuous_manifest.json", summaries)
    return summaries


class ContinuousDataset(Dataset):
    def __init__(self, output: Path, matches, evaluation=False, short_history=False):
        self.sources, self.index, self.short_history = [], [], short_history
        for match in matches:
            for half in ("1st", "2nd"):
                directory = output / "continuous" / f"{match}_{half}"
                summary = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                arrays = {path.stem: np.load(path, mmap_mode="r", allow_pickle=False)
                          for path in directory.glob("*.npy")}
                source_index = len(self.sources)
                self.sources.append((str(match), arrays))
                seen_anchors = set()
                for row in range(summary["examples"]):
                    anchor = int(arrays["anchor"][row])
                    # One perspective per nine-second nonoverlapping history/future interval.
                    if evaluation and (anchor % 225 != 150 or anchor in seen_anchors):
                        continue
                    self.index.append((source_index, row))
                    seen_anchors.add(anchor)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        source, row = self.index[item]
        _, arrays = self.sources[source]
        result = {}
        for key, array in arrays.items():
            if key == "anchor":
                continue
            value = np.array(array[row], copy=True)
            result[key] = torch.from_numpy(value.astype(np.float32) if value.dtype == np.float16 else value)
        if self.short_history:
            result["x"] = result["x"][-11:].clone()
            result["mask"] = result["mask"][-11:].clone()
            result["visible"] = result["visible"][-11:].clone()
            seen = torch.zeros(22, dtype=torch.bool)
            times_seen = torch.zeros(22, dtype=torch.int32)
            for time_index in range(11):
                now = result["visible"][time_index]
                times_seen += now
                seen |= now
                result["mask"][time_index] &= seen
                result["x"][time_index, ~seen] = 0
                result["x"][time_index, times_seen < 2, 2:4] = 0
            result["future_mask"][~seen] = False
            result["position_mask"][~seen] = False
            result["last_xy"][~seen] = 0
            result["age"][~seen] = 0
            result["velocity"] = result["x"][-1, :, 2:4] * 12
        result["source_index"] = torch.tensor(source)
        return result
