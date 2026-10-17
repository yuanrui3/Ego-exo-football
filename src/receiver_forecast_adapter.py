"""Causal, identity-associated receiver inputs and metric forecast features."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.audit_trajectory_integration import reconstruct_history, receiver_mapping
from src.models.candidate_ranker_receiver import candidate_features
from src.receiver_tensor_dataset import ENTITIES, parse_receiver_prompt
from src.trajectory_dataset import TRAIN_MATCHES, file_digest, source_signature

TIMES = (.4, 1., 2., 3.)
MOVEMENT_NAMES = (
    "displacement_x", "displacement_y", "displacement_norm",
    "relative_x", "relative_y", "actor_distance",
    "motion_along_current_pass_direction", "motion_across_current_pass_direction",
)
PRESSURE_NAMES = (
    "nearest_observed_defender", "defenders_within_3m", "defenders_within_5m",
    "defenders_within_10m", "separation_change", "pressure5_change", "space_radius_proxy",
)
AVAILABILITY_NAMES = ("candidate_history_fraction", "observed_defender_fraction", "forecast_available")
FORECAST_NAMES = tuple(f"{name}_{h}" for h in TIMES for name in (*MOVEMENT_NAMES, *PRESSURE_NAMES)) + AVAILABILITY_NAMES
MOVEMENT_INDICES = tuple(i for i, name in enumerate(FORECAST_NAMES) if any(name.startswith(n + "_") for n in MOVEMENT_NAMES))
PRESSURE_INDICES = tuple(i for i, name in enumerate(FORECAST_NAMES) if any(name.startswith(n + "_") for n in PRESSURE_NAMES))


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def assert_out_of_fold(predicted_match, fitting_matches, selection_matches=()):
    fitting, selected = set(map(str, fitting_matches)), set(map(str, selection_matches))
    if not fitting <= set(TRAIN_MATCHES) or not selected <= fitting:
        raise ValueError("Upstream fold uses non-training matches or foreign selection matches")
    if str(predicted_match) in fitting | selected:
        raise ValueError(f"IN-SAMPLE FORECAST REJECTED: {predicted_match}")
    if not fitting:
        raise ValueError("Empty upstream fitting partition")


def build_history_inputs(raw, anchor):
    """Target-free boundary: truncate every array before any processing."""
    allowed = ("xy", "observed", "segments", "teams", "roles")
    prefix = {key: raw[key][:anchor + 1] for key in allowed}
    return reconstruct_history(prefix, anchor)


def anonymous_inputs(inputs, selected_slots, label_to_slot):
    local = {int(slot): i for i, slot in enumerate(selected_slots)}
    gather = np.array([local.get(label_to_slot.get(label, -1), -1) for label in ENTITIES])
    result = {}
    for key, value in inputs.items():
        result[key] = np.zeros_like(value)
        good = gather >= 0
        result[key][good] = value[gather[good]]
    return result


def asof_base(raw, slots, row, anchor):
    """Separate benchmark: identity association at as-of time; no legacy future ordering."""
    current = np.flatnonzero(raw["observed"][anchor])
    actor = slots.get(str(row["passer_id"]))
    receiver = slots.get(str(row["receiver_id"]))
    if actor not in current:
        return None
    team = raw["teams"][anchor, actor]
    xy = raw["xy"][anchor]
    same = [int(s) for s in current if s != actor and raw["teams"][anchor, s] == team]
    opponents = [int(s) for s in current if raw["teams"][anchor, s] != team]
    # Stable slot tie-breaker is a documented association rule, not a learned identity.
    key = lambda s: (float(np.linalg.norm(xy[s] - xy[actor])), s)
    same.sort(key=key)
    opponents.sort(key=key)
    if len(same) > 10 or len(opponents) > 11:
        return None
    mapping = {"ACTOR": int(actor)}
    mapping.update({f"TEAMMATE_{i+1}": slot for i, slot in enumerate(same)})
    mapping.update({f"OPPONENT_{i+1}": slot for i, slot in enumerate(opponents)})
    state = np.zeros((3, 22, 8), np.float32)
    state[:, 0, 3], state[:, 1:11, 4], state[:, 11:, 5] = 1, 1, 1
    for label, slot in mapping.items():
        index = ENTITIES.index(label)
        state[:, index, 6 if raw["roles"][anchor, slot] else 7] = 1
        for time, offset in enumerate((-10, -5, 0)):
            frame = anchor + offset
            if frame >= 0 and raw["observed"][frame, slot] and (
                raw["segments"][frame, slot] == raw["segments"][anchor, slot]
            ):
                state[time, index, :2] = raw["xy"][frame, slot] / [52.5, 34.]
                state[time, index, 2] = 1
    target = same.index(receiver) if receiver in same else -1
    return state, mapping, target


def prepare_receiver_inputs(rows, trajectory_root, association_root, timing="legacy"):
    if timing not in ("legacy", "asof"):
        raise ValueError("Unknown timing contract")
    n = len(rows)
    inputs = {
        "history": np.zeros((n, 22, 51, 5), np.float32),
        "current": np.zeros((n, 22, 2), np.float32),
        "velocity": np.zeros((n, 22, 2), np.float32),
        "team": np.zeros((n, 22), np.int64), "role": np.zeros((n, 22), np.int64),
        "player_mask": np.zeros((n, 22), bool),
    }
    states = np.zeros((n, 3, 22, 8), np.float32)
    targets = np.full(n, -1, np.int64)
    covered, base_valid = np.zeros(n, bool), np.zeros(n, bool)
    reasons, provenance, signatures = [], [], {}
    groups = {}
    for index, row in enumerate(rows):
        half = "1st" if row["id"].split("_p")[1][0] == "1" else "2nd"
        groups.setdefault((str(row["match_id"]), half), []).append(index)
    for (match, half), indices in groups.items():
        name = f"{match}_{half}"
        path = Path(trajectory_root) / "cache" / f"{name}.npz"
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        association_path = Path(association_root) / f"associations_{name}.json"
        association = json.loads(association_path.read_text(encoding="utf-8"))
        if file_digest(path) != meta["cache_sha256"] or association["cache_sha256"] != meta["cache_sha256"]:
            raise ValueError("Trajectory cache/association checksum mismatch")
        if association["source"] != meta["source"] or source_signature(meta["source"]["path"]) != meta["source"]:
            raise ValueError("Raw association source changed")
        signatures[name] = {"cache": meta["cache_sha256"], "association": file_digest(association_path)}
        with np.load(path, allow_pickle=False) as blob:
            raw = {key: blob[key] for key in blob.files}
        states_by_frame = {int(k): value for k, value in association["states"].items()}
        events = json.loads((Path("data") / f"{match}_12_class_events.json").read_text(encoding="utf-8"))
        if events["clock"] != "ms from the first frame of the half's released video":
            raise ValueError("Unverified BAS time reference")
        events = events.get("actions") or events.get("annotations")
        for index in indices:
            row = rows[index]
            legacy_frame = int(row["id"].split("_f")[1].split("_")[0])
            event = events[int(row["id"].rsplit("_", 1)[1])]
            event_ms = float(event["position"])
            anchor = legacy_frame if timing == "legacy" else int(np.floor(event_ms / 40))
            if timing == "legacy":
                mapping, matches = receiver_mapping(row, anchor, states_by_frame, association["slots"])
                if not matches:
                    raise ValueError(f"Frozen candidate order mismatch: {row['id']}")
                states[index] = parse_receiver_prompt(row["prompt"]).numpy()
                targets[index] = int(row["response"].split("_")[1]) - 1
                base_valid[index] = True
            else:
                if anchor * 40 > event_ms:
                    raise AssertionError("As-of anchor is after BAS time")
                built = asof_base(raw, association["slots"], row, anchor)
                if built is None:
                    reasons.append({"id": row["id"], "reason": "asof_actor_missing_or_over_capacity"})
                    continue
                states[index], mapping, targets[index] = built
                base_valid[index] = targets[index] >= 0
            provenance.append({
                "id": row["id"], "match": match, "anchor_frame": anchor,
                "anchor_minus_bas_ms": anchor * 40 - event_ms,
                "timing": timing, "latest_feature_frame": anchor,
            })
            if anchor < 50:
                reasons.append({"id": row["id"], "reason": "less_than_2s_history"})
                continue
            present = int(raw["observed"][anchor].sum())
            if not 1 <= present <= 22:
                reasons.append({"id": row["id"], "reason": "player_capacity"})
                continue
            history, selected, _ = build_history_inputs(raw, anchor)
            ordered = anonymous_inputs(history, selected, mapping)
            all_ten = ordered["player_mask"][1:11].all()
            # Strict common-subset policy: all ten histories complete, known current actor.
            complete = ordered["history"][1:11, :, 4].all()
            covered[index] = bool(all_ten and complete and ordered["player_mask"][0] and base_valid[index])
            if not covered[index]:
                reasons.append({"id": row["id"], "reason": "missing_candidate_or_candidate_history_or_target"})
                continue
            for key in inputs:
                inputs[key][index] = ordered[key]
    return {
        "inputs": inputs, "states": states, "targets": targets, "covered": covered,
        "base_valid": base_valid, "ids": [r["id"] for r in rows],
        "matches": [str(r["match_id"]) for r in rows], "reasons": reasons,
        "provenance": provenance, "source_signatures": signatures, "timing": timing,
    }


def forecast_features(current, future, observed, history_fraction, available):
    """Inputs use anonymous actor/candidate/opponent order; all future values are predictions."""
    candidates, actor, defenders = future[:, 1:11], future[:, :1], future[:, 11:]
    candidate_seen = observed[:, 1:11]
    defender_seen = observed[:, 11:]
    has_defender = defender_seen.any(-1)[:, None].expand(-1, 10)
    now_dist = torch.cdist(current[:, 1:11], current[:, 11:])
    now_nearest = now_dist.masked_fill(~defender_seen[:, None], float("inf")).amin(-1)
    now_pressure5 = ((now_dist <= 5) & defender_seen[:, None]).sum(-1).float()
    direction = current[:, 1:11] - current[:, :1]
    norm = torch.linalg.vector_norm(direction, dim=-1)
    unit = direction / norm.clamp_min(1e-6)[..., None]
    values, validity = [], []
    base_good = candidate_seen & observed[:, :1] & available[:, None]
    pressure_good = base_good & has_defender
    for horizon in range(4):
        candidate = candidates[:, :, horizon]
        relative = candidate - actor[:, :, horizon]
        movement = candidate - current[:, 1:11]
        displacement = torch.linalg.vector_norm(movement, dim=-1)
        distance = torch.linalg.vector_norm(relative, dim=-1)
        along = (movement * unit).sum(-1)
        across = movement[..., 0] * unit[..., 1] - movement[..., 1] * unit[..., 0]
        d = torch.cdist(candidate, defenders[:, :, horizon])
        nearest = d.masked_fill(~defender_seen[:, None], float("inf")).amin(-1)
        pressures = [((d <= radius) & defender_seen[:, None]).sum(-1).float() for radius in (3, 5, 10)]
        movement_values = (movement[..., 0], movement[..., 1], displacement,
                           relative[..., 0], relative[..., 1], distance, along, across)
        pressure_values = (nearest, *pressures, nearest - now_nearest,
                           pressures[1] - now_pressure5, nearest.clamp(max=20))
        values.extend((*movement_values, *pressure_values))
        validity.extend([base_good] * 6 + [base_good & (norm > 1e-6)] * 2
                        + [pressure_good] * len(pressure_values))
    values.extend((history_fraction[:, 1:11], defender_seen.float().sum(-1)[:, None].expand(-1, 10) / 11,
                   available[:, None].expand(-1, 10).float()))
    validity.extend((base_good, base_good, candidate_seen))
    valid = torch.stack(validity, -1)
    features = torch.stack(values, -1).masked_fill(~valid, 0)
    if not torch.isfinite(features).all():
        raise FloatingPointError("Nonfinite forecast feature")
    return features, valid


def base_engineered(states):
    with torch.no_grad():
        return candidate_features(torch.as_tensor(states, dtype=torch.float32))


def select_forecast_group(values, valid, variant):
    mask = torch.ones(values.shape[-1], dtype=torch.bool, device=values.device)
    if variant == "E":
        mask[list(PRESSURE_INDICES)] = False
    elif variant == "F":
        mask[list(MOVEMENT_INDICES)] = False
    return values[..., mask], valid[..., mask]
