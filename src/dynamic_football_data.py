"""Isolated causal datasets for the dynamic-GNN study."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.belief_observation import encode_observations, occupancy
from src.belief_tracking_dataset import load_npz
from src.player_centric_dataset import stable_seed
from src.player_centric_visibility import simulate_visibility
from src.receiver_forecast_adapter import asof_base
from src.trajectory_dataset import check_deadline, file_digest, write_json

FIT_MATCHES = ("117092", "117093", "118576", "118578", "128057", "132831")
DEVELOPMENT = ("132877",)
RETROSPECTIVE = ("118577", "128058")
ALL_MATCHES = (*FIT_MATCHES, *DEVELOPMENT, *RETROSPECTIVE)
WORLD_FIELDS = ("x", "mask", "visible", "last_xy", "actor_xy", "velocity", "age",
                "position", "position_mask", "occupancy", "anonymous_defender", "anchor",
                "target_association")
WORLD_INPUTS = ("x", "mask", "actor_xy", "last_xy")


def assert_partition(fitting, predicted=(), outer_excluded=()):
    fitting = set(map(str, fitting))
    if not fitting or not fitting <= set(FIT_MATCHES):
        raise ValueError("Only the six designated fitting matches may train a model")
    if fitting & (set(map(str, predicted)) | set(map(str, outer_excluded))):
        raise ValueError("Out-of-fold or outer-fold exclusion violated")


def make_visibility(raw, times, actor, seed, fov=120):
    xy, observed = raw["xy"][times], raw["observed"][times]
    visible = simulate_visibility(xy, observed, actor, fov_degrees=fov,
                                  max_range_m=45 if fov != 360 else 1000, seed=seed)[0]
    rng = np.random.default_rng(seed)
    for player in range(observed.shape[1]):
        if player != actor and rng.random() < .4:
            start = int(rng.integers(0, len(times)))
            length = int(rng.integers(3, len(times) + 1))
            visible[start:start + length, player] = False
    return visible


def encode_window(raw, times, actor, visible):
    """Reinitialize first-sight slots and velocities for EACH chosen history."""
    xy, observed = raw["xy"][times], raw["observed"][times]
    segments = raw["segments"][times]
    if not observed[:, actor].all() or np.any(segments[:, actor] != segments[-1, actor]):
        raise ValueError("Actor discontinuity in history")
    result = encode_observations(xy, observed, segments, raw["teams"][times],
                                 raw["roles"][times], actor, visible)
    mapping = result.pop("mapping")
    association = np.full(22, -1, np.int16)
    for player in np.flatnonzero(mapping >= 0):
        association[mapping[player]] = player
    hidden = observed[-1] & ~visible[-1]
    position = np.zeros((22, 2), np.float32)
    eligible = np.zeros(22, bool)
    for player in np.flatnonzero(mapping >= 0):
        slot = mapping[player]
        last_index = len(times) - 1 - int(round(float(result["age"][slot]) / .2))
        same = segments[last_index, player] == segments[-1, player]
        eligible[slot] = hidden[player] and same
        if observed[-1, player] and same:
            position[slot] = xy[-1, player]
    opponents = raw["teams"][times[-1]] != raw["teams"][times[-1], actor]
    result.update(position=position, position_mask=eligible, actor_xy=xy[-1, actor].copy(),
                  occupancy=occupancy(xy[-1], hidden),
                  anonymous_defender=occupancy(xy[-1], hidden & (mapping < 0) & opponents),
                  anchor=np.int32(times[-1]), target_association=association)
    return result


def states_to_graph(states):
    """The very same strictly as-of three snapshots feed P0 and graph controls."""
    state = np.asarray(states, np.float32)
    n, t = state.shape[:2]
    x = np.zeros((n, t, 22, 10), np.float32)
    valid = state[..., 2] > .5
    absolute = state[..., :2] * [52.5, 34.]
    x[..., :2] = (absolute - absolute[:, :, :1]) / [52.5, 34.]
    if t > 1:
        delta = np.diff(absolute, axis=1) / .2 / 12
        good = valid[:, 1:] & valid[:, :-1]
        x[:, 1:, :, 2:4] = np.clip(delta, -1, 1) * good[..., None]
    x[..., 4:6] = valid[..., None]
    x[..., 7] = np.where(state[..., 5] > .5, -1, 1)
    x[..., 8] = state[..., 6]
    x[:, :, 0, 9] = 1
    x[~valid] = 0
    return x, valid


def build_datasets(root, output, deadline):
    output.mkdir(parents=True, exist_ok=True)
    old = root / "outputs" / "video_belief_world_model_20261018" / "continuous"
    receiver_rows = []
    for split in ("train", "val"):
        with (root / "data" / f"{split}_pass_receiver.jsonl").open(encoding="utf-8") as stream:
            receiver_rows += [json.loads(line) for line in stream if line.strip()]
    receiver = output / "receiver_asof.npz"
    receiver_records, rejected, sources = [], [], {}
    saved_receiver = receiver.exists()
    for match in ALL_MATCHES:
        events_path = root / "data" / f"{match}_12_class_events.json"
        events = json.loads(events_path.read_text(encoding="utf-8"))
        if events["clock"] != "ms from the first frame of the half's released video":
            raise ValueError("BAS clock is unverified")
        events = events.get("actions") or events.get("annotations")
        for half_index, half in enumerate(("1st", "2nd"), 1):
            check_deadline(deadline)
            name = f"{match}_{half}"
            cache = root / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{name}.npz"
            association_path = root / "outputs" / "trajectory_audit_20261016" / f"associations_{name}.json"
            association = json.loads(association_path.read_text(encoding="utf-8"))
            meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
            digest = file_digest(cache)
            if digest != meta["cache_sha256"] or digest != association["cache_sha256"]:
                raise ValueError(f"Tracking provenance mismatch: {name}")
            sources[name] = {"cache": digest, "association": file_digest(association_path)}
            paths = [output / f"world_{seconds}s" / name for seconds in (2, 6)]
            ready = all((path / "manifest.json").exists() for path in paths)
            if ready:
                for path in paths:
                    if json.loads((path / "manifest.json").read_text())["cache_sha256"] != digest:
                        raise ValueError("World source changed since preparation")
            if ready and saved_receiver:
                continue
            raw = load_npz(cache)
            if not ready:
                anchors = np.load(old / name / "anchor.npy", mmap_mode="r")
                actor_xy = np.load(old / name / "actor_xy.npy", mmap_mode="r")
                records = {2: [], 6: []}
                excluded = Counter()
                print(f"Preparing matched histories: {name}, {len(anchors)} historical rows", flush=True)
                for row, anchor_value in enumerate(anchors):
                    if row % 200 == 0:
                        check_deadline(deadline)
                        write_json(output / "preparation_progress.json", {
                            "half": name, "row": row, "historical_rows": len(anchors),
                            "matched_rows": len(records[2]), "exclusions": dict(excluded)})
                    anchor = int(anchor_value)
                    # Historical actor_xy was float16; match its quantization exactly.
                    matches = np.flatnonzero(raw["observed"][anchor] & (
                        raw["xy"][anchor].astype(np.float16) == actor_xy[row]).all(-1))
                    if len(matches) != 1:
                        excluded["actor_quantization_ambiguous_or_missing"] += 1
                        continue
                    actor = int(matches[0])
                    seed = stable_seed(f"{name}:{anchor}:{actor}", 42)
                    times = np.arange(anchor - 150, anchor + 1, 5)
                    visibility = make_visibility(raw, times, actor, seed)
                    try:
                        long = encode_window(raw, times, actor, visibility)
                        short = encode_window(raw, times[-11:], actor, visibility[-11:])
                    except ValueError as error:
                        excluded[str(error)] += 1
                        continue
                    records[2].append(short)
                    records[6].append(long)
                if not records[2]:
                    raise ValueError(f"No matched histories for {name}")
                for seconds, path in zip((2, 6), paths):
                    path.mkdir(parents=True, exist_ok=True)
                    for field in WORLD_FIELDS:
                        array = np.stack([record[field] for record in records[seconds]])
                        if array.dtype == np.float32:
                            array = array.astype(np.float16)
                        np.save(path / f"{field}.npy", array, allow_pickle=False)
                    write_json(path / "manifest.json", {
                        "examples": len(records[seconds]), "history_seconds": seconds,
                        "cache_sha256": digest, "exclusions": dict(excluded), "fov": 120,
                        "input_fields": list(WORLD_INPUTS), "target_only_fields": [
                            "position", "position_mask", "occupancy", "anonymous_defender", "target_association"],
                        "matching": "identical endpoint visibility, shared six-second simulated mask; first-sight memory rebuilt",
                    })
                del records
            if not saved_receiver:
                for row in receiver_rows:
                    if str(row["match_id"]) != match or int(row["id"].split("_p")[1][0]) != half_index:
                        continue
                    event = events[int(row["id"].rsplit("_", 1)[1])]
                    anchor = int(np.floor(float(event["position"]) / 40))
                    if anchor * 40 > float(event["position"]):
                        raise AssertionError("Post-event receiver observation")
                    built = asof_base(raw, association["slots"], row, anchor)
                    if built is None or built[2] < 0:
                        rejected.append({"id": row["id"], "reason": "asof_actor/capacity/receiver_unavailable"})
                        continue
                    state, _, target = built
                    receiver_records.append({"states": state, "target": np.int64(target), "ids": row["id"],
                                             "matches": match, "anchor": np.int32(anchor)})
            del raw
    if not saved_receiver:
        packed = {key: np.stack([record[key] for record in receiver_records]) for key in receiver_records[0]}
        packed["x"], packed["mask"] = states_to_graph(packed["states"])
        np.savez_compressed(receiver, **packed)
        write_json(output / "receiver_manifest.json", {
            "examples": len(receiver_records), "excluded": rejected,
            "clock": "floor(BAS milliseconds/40); no future observations",
            "candidate_interface": "10 as-of distance-ranked teammates; target is next-same-team actor proxy",
            "history": "three snapshots over 0.4 seconds, IDENTICAL for fresh P0 and all receiver models",
            "limitation": "common full-observation receiver control, not proof of partial-observation utility",
        })
    write_json(output / "sources.json", sources)


class WorldDataset(Dataset):
    def __init__(self, output, matches, seconds=2, evaluation=False):
        self.matches = tuple(matches)
        self.sources, self.index = [], []
        for match in self.matches:
            for half in ("1st", "2nd"):
                directory = output / f"world_{seconds}s" / f"{match}_{half}"
                arrays = {field: np.load(directory / f"{field}.npy", mmap_mode="r") for field in WORLD_FIELDS}
                source = len(self.sources)
                self.sources.append(arrays)
                seen = set()
                for row, anchor in enumerate(arrays["anchor"]):
                    # Non-overlapping 9s evaluation spans, same rows across history lengths.
                    if evaluation and (int(anchor) % 225 != 150 or int(anchor) in seen):
                        continue
                    self.index.append((source, row))
                    seen.add(int(anchor))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        source, row = self.index[index]
        arrays = self.sources[source]
        return {key: torch.from_numpy(np.asarray(arrays[key][row]).copy().astype(
            np.float32 if arrays[key].dtype == np.float16 else arrays[key].dtype))
                for key in WORLD_FIELDS if key != "anchor"}


def risk_world_inputs(root, risk):
    """Only input fields from the frozen event cache cross the model boundary."""
    event = load_npz(root / "outputs" / "video_belief_world_model_20261018" / "events_v2.npz")
    rows = risk["event_index"]
    if not np.array_equal(event["ids"][rows], risk["ids"]):
        raise ValueError("Risk/event scene mismatch")
    result = {key: event[key][rows].astype(np.float32 if event[key].dtype == np.float16 else event[key].dtype)
              for key in ("x", "mask", "visible", "actor_xy", "last_xy", "velocity", "age")}
    result["x"][..., 6] *= 5 / 8
    return result
