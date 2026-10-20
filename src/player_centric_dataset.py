"""Build leakage-safe tracking-only global and synthetic egocentric examples."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.player_centric_visibility import HeadingMode, simulate_visibility
from src.recode_primary_actions import SOURCE_TO_PRIMARY_ACTION
from src.trajectory_dataset import TRAIN_MATCHES, VALIDATION_MATCHES, atomic_replace, file_digest

ACTION_LABELS = (
    "PASS", "HIGH_PASS", "DRIVE", "CROSS", "SHOT", "HEADER",
    "DEFENSIVE_INTERVENTION",
)
ACTION_ALIASES = {
    key.upper().replace(" ", "_"): value for key, value in SOURCE_TO_PRIMARY_ACTION.items()
}
FOV_WIDTHS = (90, 120, 180, 360)
FEATURE_COUNT = 10
GRID_X, GRID_Y, GRID_CELL_M = 21, 14, 5.0
ROW_ID = re.compile(r"^(?P<match>\d+)_p(?P<period>[12])_f(?P<frame>\d+)_(?P<index>\d+)$")


def stable_seed(value: str, seed: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _anonymous_slots(current_slots: list[int], slot_to_player: dict[int, str], key: str) -> dict[int, int]:
    ordered = sorted(current_slots, key=lambda slot: slot_to_player.get(slot, f"~{slot}"))
    rng = np.random.default_rng(stable_seed(key, 1741))
    anon_indices = rng.permutation(22)
    return {slot: int(anon_indices[i]) for i, slot in enumerate(ordered)}


def _frame_tensor(raw: dict[str, np.ndarray], anchor: int, actor_slot: int, period: str,
                  history_seconds: float, sample_stride: int):
    fps = 25
    span = int(round(history_seconds * fps))
    if span < 1 or span % sample_stride:
        raise ValueError("History span must be a positive multiple of sample_stride")
    offsets = np.arange(-span, 1, sample_stride, dtype=np.int64)
    if len(offsets) > 26:
        raise ValueError("History is capped at 5 seconds sampled every 0.2 seconds")
    if anchor < 0 or anchor >= len(raw["xy"]) or not raw["observed"][anchor, actor_slot]:
        raise ValueError("Actor is not observed at the BAS as-of frame")
    current_slots = np.flatnonzero(raw["observed"][anchor])
    if len(current_slots) > 22 or actor_slot not in current_slots:
        raise ValueError("Decision scene exceeds 22 current tracked players")
    segment = raw["segments"][anchor, current_slots]
    xy = np.zeros((len(offsets), len(current_slots), 2), np.float32)
    seen = np.zeros((len(offsets), len(current_slots)), bool)
    teams = raw["teams"][anchor, current_slots].astype(np.int64)
    roles = raw["roles"][anchor, current_slots].astype(np.int64)
    for ti, offset in enumerate(offsets):
        frame = anchor + int(offset)
        if frame < 0:
            continue
        valid = raw["observed"][frame, current_slots] & (raw["segments"][frame, current_slots] == segment)
        xy[ti, valid] = raw["xy"][frame, current_slots[valid]]
        seen[ti] = valid
    local_actor = int(np.flatnonzero(current_slots == actor_slot)[0])
    return offsets, current_slots, xy, seen, teams, roles, local_actor


def _entity_features(
    xy: np.ndarray, observed: np.ndarray, teams: np.ndarray, roles: np.ndarray,
    actor: int, visible: np.ndarray, anonymous_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode visible measurements and last-seen memory, never hidden coordinates."""
    times, players = observed.shape
    features = np.zeros((times, 22, FEATURE_COUNT), np.float32)
    entity_mask = np.zeros((times, 22), bool)
    visibility = np.zeros((times, 22), bool)
    last_xy = np.zeros((players, 2), np.float32)
    last_velocity = np.zeros((players, 2), np.float32)
    last_seen = np.full(players, -1, np.int32)
    prior_xy = np.zeros((players, 2), np.float32)
    prior_time = np.full(players, -1, np.int32)
    last_memory_xy = np.zeros((22, 2), np.float32)
    last_memory_age = np.full(22, 99.0, np.float32)
    actor_team = teams[actor]
    for t in range(times):
        if not visible[t, actor]:
            continue
        actor_xy = xy[t, actor]
        for j in range(players):
            if visible[t, j] and observed[t, j]:
                point = xy[t, j]
                if prior_time[j] >= 0 and t > prior_time[j]:
                    dt = (t - int(prior_time[j])) * .2
                    last_velocity[j] = (point - prior_xy[j]) / max(dt, 1e-6)
                prior_xy[j], prior_time[j] = point, t
                last_xy[j], last_seen[j] = point, t
            if last_seen[j] < 0:
                continue
            out = int(anonymous_map[j])
            if out < 0:
                continue
            now_visible = bool(visible[t, j] and observed[t, j])
            age = max(0.0, (t - int(last_seen[j])) * .2)
            delta = last_xy[j] - actor_xy
            role_goalkeeper = float(roles[j] == 1)
            team_relation = 1.0 if teams[j] == actor_team else -1.0
            features[t, out] = (
                delta[0] / 52.5, delta[1] / 34.0,
                np.clip(last_velocity[j, 0] / 12.0, -1, 1),
                np.clip(last_velocity[j, 1] / 12.0, -1, 1),
                float(now_visible), 1.0, min(age, 5.0) / 5.0,
                team_relation, role_goalkeeper, float(j == actor),
            )
            entity_mask[t, out] = True
            visibility[t, out] = now_visible
            if t == times - 1:
                last_memory_xy[out] = last_xy[j]
                last_memory_age[out] = age
    return features, entity_mask, visibility, last_memory_xy, last_memory_age


def _occupancy_target(xy: np.ndarray, observed: np.ndarray, visible: np.ndarray) -> np.ndarray:
    target = np.zeros((GRID_Y, GRID_X), np.float32)
    hidden = np.flatnonzero(observed & ~visible)
    for player in hidden:
        x, y = xy[player]
        gx = int(np.clip(math.floor((float(x) + 52.5) / GRID_CELL_M), 0, GRID_X - 1))
        gy = int(np.clip(math.floor((float(y) + 34.0) / GRID_CELL_M), 0, GRID_Y - 1))
        target[gy, gx] = 1.0
    return target.reshape(-1)


def _load_specs(data_root: Path):
    action_rows, receiver_rows = [], []
    for split in ("train", "val"):
        action_rows += [dict(r, split=split) for r in _read_jsonl(data_root / f"{split}_temporal_7class.jsonl")]
        receiver_rows += [dict(r, split=split) for r in _read_jsonl(data_root / f"{split}_pass_receiver.jsonl")]
    examples = {}
    for row in action_rows:
        examples.setdefault(row["id"], {}).update(action=row["response"], split=row["split"])
    for row in receiver_rows:
        key = row.get("decision_id") or row["id"]
        # Frozen receiver records currently use the BAS decision ID for their own ID.
        key = key if key in examples else row["id"]
        examples.setdefault(key, {}).update(
            receiver_id=str(row["receiver_id"]), actor_id=str(row["passer_id"]),
            split=row["split"], receiver_example=True,
        )
    train_matches = {
        ROW_ID.fullmatch(key).group("match") for key, row in examples.items() if row["split"] == "train"
    }
    val_matches = {
        ROW_ID.fullmatch(key).group("match") for key, row in examples.items() if row["split"] == "val"
    }
    if train_matches != set(TRAIN_MATCHES) or val_matches != set(VALIDATION_MATCHES):
        raise ValueError("Action/receiver files do not match the frozen seven/two match split")
    return examples


def build_player_centric_cache(
    *,
    data_root: str | Path = "data",
    trajectory_root: str | Path = "outputs/trajectory_forecasting_20261016",
    association_root: str | Path = "outputs/trajectory_audit_20261016",
    output: str | Path,
    fov_widths: tuple[int, ...] = FOV_WIDTHS,
    max_range_m: float = 45.0,
    history_seconds: float = 2.0,
    sample_stride: int = 5,
    heading_mode: HeadingMode = "motion_random",
    coordinate_noise_std: float = 0.0,
    random_hide_probability: float = 0.0,
    split_filter: str | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> dict:
    """Build compact supervised examples from existing caches and BAS event labels."""
    data_root, output = Path(data_root), Path(output)
    specs = _load_specs(data_root)
    if split_filter is not None:
        if split_filter not in ("train", "val"):
            raise ValueError("split_filter must be train or val")
        specs = {key: value for key, value in specs.items() if value["split"] == split_filter}
    if limit is not None:
        specs = dict(list(sorted(specs.items()))[:limit])
    if coordinate_noise_std < 0 or not 0 <= random_hide_probability < 1:
        raise ValueError("Invalid coordinate noise or random-hide probability")
    # Resolve and validate match/event metadata before the large cache pass.
    by_half: dict[tuple[str, str], list[tuple[str, dict, dict]]] = {}
    event_files: dict[str, dict] = {}
    for example_id, spec in specs.items():
        match = ROW_ID.fullmatch(example_id)
        if match is None:
            raise ValueError(f"Unsupported decision identifier: {example_id}")
        groups = match.groupdict()
        mid, period, event_index = groups["match"], int(groups["period"]), int(groups["index"])
        event_file = event_files.setdefault(
            mid, json.loads((data_root / f"{mid}_12_class_events.json").read_text(encoding="utf-8"))
        )
        events = event_file.get("actions") or event_file.get("annotations")
        if event_index >= len(events):
            raise ValueError(f"BAS event index out of range for {example_id}")
        event = events[event_index]
        if not event_file.get("clock", "").startswith("ms from the first frame"):
            raise ValueError(f"Unsupported BAS time contract in match {mid}")
        event_period = str(event.get("gameTime", "")).split("-", 1)[0].strip()
        if event_period and event_period != str(period):
            raise ValueError(f"Period mismatch for {example_id}: BAS says {event_period}")
        position_ms = float(event["position"])
        anchor = int(math.floor(position_ms / 40.0))
        event_actor = str(event.get("player_id", ""))
        if spec.get("actor_id") and event_actor and spec["actor_id"] != event_actor:
            raise ValueError(f"Passer/action actor mismatch for {example_id}")
        action = spec.get("action")
        raw_action = event.get("label")
        event_action = SOURCE_TO_PRIMARY_ACTION.get(
            raw_action, ACTION_ALIASES.get(str(raw_action).upper().replace(" ", "_"), raw_action)
        )
        if action is not None and event_action != action:
            raise ValueError(f"Action label mismatch for {example_id}: {event_action} vs {action}")
        spec.update(match_id=mid, period=period, anchor=anchor, actor_id=event_actor or spec.get("actor_id", ""))
        half = "1st" if period == 1 else "2nd"
        by_half.setdefault((mid, half), []).append((example_id, spec, event))

    fovs = tuple(int(v) for v in fov_widths)
    if not fovs or any(v not in (90, 120, 180, 360) for v in fovs):
        raise ValueError("FOV widths must be selected from 90, 120, 180, 360")
    records: list[dict] = []
    exclusions: Counter[str] = Counter()
    source_hashes = {}
    for (mid, half), group in sorted(by_half.items()):
        cache_path = Path(trajectory_root) / "cache" / f"{mid}_{half}.npz"
        sidecar = Path(association_root) / f"associations_{mid}_{half}.json"
        meta = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
        association = json.loads(sidecar.read_text(encoding="utf-8"))
        if file_digest(cache_path) != meta["cache_sha256"] or association["cache_sha256"] != meta["cache_sha256"]:
            raise ValueError(f"Cache/association checksum mismatch: {mid}_{half}")
        source_hashes[f"{mid}_{half}"] = {
            "cache": meta["cache_sha256"], "association": file_digest(sidecar),
        }
        id_to_slot = {str(pid): int(slot) for pid, slot in association["slots"].items()}
        slot_to_player = {slot: pid for pid, slot in id_to_slot.items()}
        with np.load(cache_path, allow_pickle=False) as blob:
            raw = {key: blob[key] for key in blob.files}
            for example_id, spec, _ in group:
                actor_slot = id_to_slot.get(str(spec["actor_id"]))
                if actor_slot is None:
                    exclusions["actor_identity_missing"] += 1
                    continue
                anchor = int(spec["anchor"])
                if anchor >= len(raw["xy"]) or not raw["observed"][anchor, actor_slot]:
                    exclusions["actor_missing_at_asof_frame"] += 1
                    continue
                try:
                    (offsets, current_slots, xy0, observed0, team0, role0, local_actor) = _frame_tensor(
                        raw, anchor, actor_slot, half, history_seconds, sample_stride,
                    )
                except ValueError as error:
                    exclusions[str(error)] += 1
                    continue
                # Reindex only by a per-sequence random permutation. IDs remain association-only.
                slot_map = _anonymous_slots(current_slots.tolist(), slot_to_player, example_id)
                anon_map = np.full(len(current_slots), -1, np.int64)
                for i, slot in enumerate(current_slots):
                    anon_map[i] = slot_map[int(slot)]
                actor_anon = int(anon_map[local_actor])
                global_visible = observed0.copy()
                gx, gm, gv, _, _ = _entity_features(
                    xy0, observed0, team0, role0, local_actor, global_visible, anon_map,
                )
                partial_x, partial_m, visibility, occupancy, memory_xy, memory_age = [], [], [], [], [], []
                reliability_by_fov = []
                for width in fovs:
                    max_range = 1000.0 if width == 360 else max_range_m
                    scenario_seed = stable_seed(example_id, seed)
                    scenario_rng = np.random.default_rng(scenario_seed + width)
                    reported_xy = xy0.copy()
                    if coordinate_noise_std:
                        reported_xy[observed0] += scenario_rng.normal(
                            0, coordinate_noise_std, size=(int(observed0.sum()), 2)
                        ).astype(np.float32)
                    vis, headings, reliable = simulate_visibility(
                        reported_xy, observed0, local_actor, fov_degrees=width,
                        max_range_m=max_range, seed=scenario_seed,
                        heading_mode=heading_mode,
                    )
                    if random_hide_probability:
                        drop = scenario_rng.random(vis.shape) < random_hide_probability
                        drop[:, local_actor] = False
                        vis &= ~drop
                    # Keep actor visible whenever it is tracked; no hidden coordinates pass this boundary.
                    vis[:, local_actor] = observed0[:, local_actor]
                    px, pm, pv, memxy, age = _entity_features(
                        reported_xy, observed0, team0, role0, local_actor, vis, anon_map,
                    )
                    visible_now = pv[-1]
                    hidden_now = observed0[-1] & ~vis[-1]
                    partial_x.append(px)
                    partial_m.append(pm)
                    visibility.append(pv)
                    occupancy.append(_occupancy_target(xy0[-1], observed0[-1], vis[-1]))
                    memory_xy.append(memxy)
                    memory_age.append(age)
                    reliability_by_fov.append(reliable)
                receiver_player = id_to_slot.get(str(spec.get("receiver_id", "")), -1)
                current_index = {int(slot): i for i, slot in enumerate(current_slots)}
                receiver_local = current_index.get(receiver_player, -1)
                receiver_anon_global = int(anon_map[receiver_local]) if receiver_local >= 0 else 22
                receiver_targets, receiver_visible, receiver_distance = [], [], []
                for fov_i in range(len(fovs)):
                    vis_now = visibility[fov_i][-1]
                    is_visible = receiver_local >= 0 and bool(vis_now[anon_map[receiver_local]])
                    receiver_visible.append(is_visible)
                    receiver_targets.append(
                        int(anon_map[receiver_local]) if is_visible else 22
                    )
                    if receiver_local >= 0:
                        receiver_distance.append(float(np.linalg.norm(xy0[-1, receiver_local] - xy0[-1, local_actor])))
                    else:
                        receiver_distance.append(float("nan"))
                action_target = ACTION_LABELS.index(spec["action"]) if spec.get("action") in ACTION_LABELS else -1
                receiver_target_global = receiver_anon_global
                records.append({
                    "id": example_id, "match_id": mid, "split": spec["split"],
                    "action": action_target, "receiver_example": bool(spec.get("receiver_example")),
                    "global_x": gx.astype(np.float16), "global_mask": gm,
                    "partial_x": np.asarray(partial_x, np.float16),
                    "partial_mask": np.asarray(partial_m, bool),
                    "visibility": np.asarray(visibility, bool),
                    "occupancy": np.asarray(occupancy, np.float16),
                    "memory_xy": np.asarray(memory_xy, np.float32),
                    "memory_age": np.asarray(memory_age, np.float32),
                    "heading_reliable": np.asarray(reliability_by_fov, bool),
                    "receiver_global": receiver_target_global,
                    "receiver_targets": np.asarray(receiver_targets, np.int64),
                    "receiver_visible": np.asarray(receiver_visible, bool),
                    "receiver_distance": np.asarray(receiver_distance, np.float32),
                })
    if not records:
        raise ValueError("No usable decision examples were constructed")
    records.sort(key=lambda r: (r["split"], r["match_id"], r["id"]))
    summary = {
        "examples": len(records), "action_examples": int(sum(r["action"] >= 0 for r in records)),
        "receiver_examples": int(sum(r["receiver_example"] for r in records)),
        "exclusions": dict(exclusions), "matches": sorted({r["match_id"] for r in records}),
        "fov_widths": list(fovs), "max_range_m": max_range_m,
        "history_seconds": history_seconds, "sample_stride_frames": sample_stride,
        "sample_times_per_example": int(round(history_seconds * 25 / sample_stride)) + 1,
        "heading_mode": heading_mode, "coordinate_noise_std_m": coordinate_noise_std,
        "random_visible_player_hide_probability": random_hide_probability,
        "split_filter": split_filter,
        "heading_reliability": {
            str(width): float(np.mean(np.stack([r["heading_reliable"][i] for r in records])))
            for i, width in enumerate(fovs)
        },
        "visibility": {
            str(width): {
                "mean_players": float(np.mean([r["visibility"][i, -1].sum() for r in records])),
                "mean_visible_teammates": float(np.mean([
                    ((r["partial_x"][i, -1, :, 7] > .5)
                     & (r["partial_x"][i, -1, :, 9] < .5)
                     & r["visibility"][i, -1]).sum()
                    for r in records
                ])),
            } for i, width in enumerate(fovs)
        },
        "source_hashes": source_hashes,
        "action_labels": list(ACTION_LABELS),
        "receiver_target_convention": "visible anonymous player slot or UNOBSERVED_TARGET index 22",
        "timing_contract": "BAS position milliseconds; anchor=floor(position_ms/40), inputs end at that frame",
        "physical_contact_sync": "unverified",
    }
    arrays = {
        "ids": np.asarray([r["id"] for r in records]),
        "match_ids": np.asarray([r["match_id"] for r in records]),
        "splits": np.asarray([r["split"] for r in records]),
        "action": np.asarray([r["action"] for r in records], np.int64),
        "receiver_example": np.asarray([r["receiver_example"] for r in records], bool),
        "global_receiver": np.asarray([r["receiver_global"] for r in records], np.int64),
        "receiver_targets": np.stack([r["receiver_targets"] for r in records]),
        "receiver_visible": np.stack([r["receiver_visible"] for r in records]),
        "receiver_distance": np.stack([r["receiver_distance"] for r in records]),
        "global_x": np.stack([r["global_x"] for r in records]),
        "global_mask": np.stack([r["global_mask"] for r in records]),
        "partial_x": np.stack([r["partial_x"] for r in records]),
        "partial_mask": np.stack([r["partial_mask"] for r in records]),
        "visibility": np.stack([r["visibility"] for r in records]),
        "occupancy": np.stack([r["occupancy"] for r in records]),
        "memory_xy": np.stack([r["memory_xy"] for r in records]),
        "memory_age": np.stack([r["memory_age"] for r in records]),
        "heading_reliable": np.stack([r["heading_reliable"] for r in records]),
        "fov_widths": np.asarray(fovs, np.int64),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    atomic_replace(temporary, output)
    summary["cache_path"] = str(output.resolve())
    summary["cache_sha256"] = file_digest(output)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
