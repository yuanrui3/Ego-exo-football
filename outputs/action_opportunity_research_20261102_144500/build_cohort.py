from __future__ import annotations

import bisect
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
COHORT_DIR = RUN / "cohort_shards_sharedmask"
sys.path.insert(0, str(ROOT))
from src.backbone.event_time import resolve_event_time  # noqa: E402

FIT = {"117092", "117093", "118576", "118578", "128057", "132831"}
DEV = {"132877"}
RETROSPECTIVE = {"118577", "128058"}
MATCH_SPLIT = {
    **{match: "fit" for match in FIT},
    **{match: "development" for match in DEV},
    **{match: "previously_exposed_retrospective" for match in RETROSPECTIVE},
}
LABEL_MAP = {
    "PASS": 1, "HIGH PASS": 2, "DRIVE": 3, "CROSS": 4,
    "SHOT": 5, "HEADER": 6,
    "BALL PLAYER BLOCK": 7, "PLAYER SUCCESSFUL TACKLE": 7,
}
CLASS_NAMES = (
    "NO_ANNOTATED_EVENT_WITHIN_HORIZON", "PASS", "HIGH_PASS", "DRIVE",
    "CROSS", "SHOT", "HEADER", "DEFENSIVE_INTERVENTION",
)
HORIZON_SECONDS = 1.0
HORIZON_FRAMES = 25
CADENCE_FRAMES = 25
HISTORY_FRAMES = 50
HISTORY_STRIDE = 5
OBSERVATION_RANGE_M = 25.0
MAX_CONTEXT_PLAYERS_PER_SIDE = 10
TOKEN_DIM = 9
ACTOR_DIM = 16
FEATURE_DIM = ACTOR_DIM + 2 * MAX_CONTEXT_PLAYERS_PER_SIDE * TOKEN_DIM


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _half_for_event(event: dict) -> str | None:
    value = str(event.get("gameTime", ""))
    if value.startswith("1 - "):
        return "1st"
    if value.startswith("2 - "):
        return "2nd"
    return None


def _event_rows(match: str, half: str, gsr_info: dict, frame_count: int,
                id_to_slot: dict[str, int], unique_slots: set[int],
                raw: dict[str, np.ndarray]) -> list[dict]:
    path = ROOT / "data" / f"{match}_12_class_events.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for event in doc["actions"]:
        if _half_for_event(event) != half:
            continue
        mapped = LABEL_MAP.get(str(event.get("label", "")).strip().upper())
        resolution = resolve_event_time(
            event.get("position"), event.get("frame"),
            clock=doc["clock"], gsr_info=gsr_info,
            source_frame_count=frame_count,
        )
        if resolution.status != "supported":
            raise ValueError(f"Unresolved event clock in {match}_{half}: {resolution.reason}")
        if mapped is None:
            continue
        frame = int(resolution.native_zero_based_frame)
        player_id = str(event["player_id"]) if event.get("player_id") is not None else None
        slot = id_to_slot.get(player_id) if player_id is not None else None
        segment = None
        if slot is not None and slot in unique_slots and raw["observed"][frame, slot]:
            segment = int(raw["segments"][frame, slot])
        result.append({
            "frame": frame, "label": mapped, "slot": slot, "segment": segment,
            "known_actor": slot is not None and slot in unique_slots and segment is not None,
        })
    return sorted(result, key=lambda row: row["frame"])


def _velocity(raw: dict[str, np.ndarray], frame: int, slot: int,
              segment: int) -> tuple[np.ndarray, np.ndarray]:
    current = raw["xy"][frame, slot].astype(np.float32)
    first = frame - 25
    second = frame - 50
    velocity = np.zeros(2, dtype=np.float32)
    older_velocity = np.zeros(2, dtype=np.float32)
    if first >= 0 and raw["observed"][first, slot] and int(raw["segments"][first, slot]) == segment:
        velocity = (current - raw["xy"][first, slot]) / 1.0
        if second >= 0 and raw["observed"][second, slot] \
                and int(raw["segments"][second, slot]) == segment:
            older_velocity = (raw["xy"][first, slot] - raw["xy"][second, slot]) / 1.0
        else:
            older_velocity = velocity.copy()
    return velocity, older_velocity


def _features_from_estimates(
    actor_slot: int,
    actor_xy: np.ndarray,
    actor_velocity: np.ndarray,
    actor_old_velocity: np.ndarray,
    actor_segment: int,
    active_slots: np.ndarray,
    candidate_xy: np.ndarray,
    candidate_velocity: np.ndarray,
    candidate_age: np.ndarray,
    candidate_sigma: np.ndarray,
    candidate_known: np.ndarray,
    candidate_teams: np.ndarray,
    candidate_roles: np.ndarray,
    actor_team: int,
    actor_role: int,
    order_reference_xy: np.ndarray | None = None,
) -> np.ndarray:
    actor_speed = float(np.linalg.norm(actor_velocity))
    old_speed = float(np.linalg.norm(actor_old_velocity))
    actor_count = int(np.sum(candidate_known))
    actor_features = np.asarray([
        actor_xy[0] / 52.5, actor_xy[1] / 34.0,
        actor_velocity[0] / 12.0, actor_velocity[1] / 12.0,
        min(actor_speed, 12.0) / 12.0,
        min(float(np.linalg.norm(actor_velocity - actor_old_velocity)), 12.0) / 12.0,
        actor_old_velocity[0] / 12.0, actor_old_velocity[1] / 12.0,
        float(actor_role == 1), float(actor_team in (1, 2)),
        min(actor_count, 22) / 22.0,
        min(len(active_slots), 22) / 22.0,
        float(np.mean(candidate_age[candidate_known])) / 2.0 if candidate_known.any() else 1.0,
        float(np.mean(candidate_sigma[candidate_known])) / 10.0 if candidate_known.any() else 1.0,
        old_speed / 12.0,
        float(np.mean(~candidate_known)),
    ], dtype=np.float32)
    tokens = []
    for slot in active_slots:
        if int(slot) == actor_slot:
            continue
        known = bool(candidate_known[slot])
        relation = (
            1.0 if known and actor_team in (1, 2) and candidate_teams[slot] == actor_team
            else -1.0 if known and actor_team in (1, 2) and candidate_teams[slot] in (1, 2)
            else 0.0
        )
        role = float(candidate_roles[slot] == 1) if known else 0.0
        delta = candidate_xy[slot] - actor_xy if known else np.zeros(2, np.float32)
        order_delta = (
            order_reference_xy[slot] - actor_xy
            if known and order_reference_xy is not None
            else delta
        )
        velocity = candidate_velocity[slot] if known else np.zeros(2, np.float32)
        distance = float(np.linalg.norm(delta)) if known else float("inf")
        order_distance = float(np.linalg.norm(order_delta)) if known else float("inf")
        token = np.asarray([
            delta[0] / 105.0, delta[1] / 68.0,
            velocity[0] / 12.0, velocity[1] / 12.0,
            float(candidate_age[slot]) / 2.0 if known else 1.0,
            float(candidate_sigma[slot]) / 10.0 if known else 1.0,
            float(known), relation, role,
        ], dtype=np.float32)
        order_dx, order_dy = (order_delta if order_reference_xy is not None else delta)
        tokens.append((
            not known, order_distance if order_reference_xy is not None else distance,
            float(order_dx), float(order_dy), token,
        ))
    tokens.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    packed = np.zeros((2 * MAX_CONTEXT_PLAYERS_PER_SIDE, TOKEN_DIM), dtype=np.float32)
    for index, (_, _, _, _, token) in enumerate(tokens[:len(packed)]):
        packed[index] = token
    result = np.concatenate((actor_features, packed.reshape(-1)))
    if result.shape != (FEATURE_DIM,) or not np.isfinite(result).all():
        raise FloatingPointError("Invalid actor-time feature row")
    return result


def _partial_feature_pair(
    raw: dict[str, np.ndarray], frame: int, active_slots: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    history_frames = np.arange(
        frame - HISTORY_FRAMES, frame + 1, HISTORY_STRIDE, dtype=np.int64
    )
    positions = raw["xy"][history_frames].astype(np.float32)
    observed = raw["observed"][history_frames].astype(bool)
    segments = raw["segments"][history_frames]
    current_segments = raw["segments"][frame]
    current_roster = np.zeros(len(current_segments), dtype=bool)
    current_roster[active_slots] = True
    same_candidate_segment = segments == current_segments[None, :]
    eligible = observed & same_candidate_segment
    same_observer_segment = eligible.copy()
    deltas = positions[:, None, :, :] - positions[:, :, None, :]
    distance = np.sqrt(np.sum(deltas * deltas, axis=-1))
    valid_pair = (
        same_observer_segment[:, :, None]
        & eligible[:, None, :]
        & (distance <= OBSERVATION_RANGE_M)
    )
    time_axis = np.arange(len(history_frames), dtype=np.int16)[:, None, None]
    last_index = np.where(valid_pair, time_axis, -1).max(axis=0)
    prior_pair = valid_pair & (time_axis < last_index[None, :, :])
    prior_index = np.where(prior_pair, time_axis, -1).max(axis=0)
    asof_map, observer_map, uncertainty_map = {}, {}, {}
    teams_at_history = raw["teams"][history_frames]
    roles_at_history = raw["roles"][history_frames]
    for actor_slot in active_slots:
        actor_slot = int(actor_slot)
        actor_segment = int(current_segments[actor_slot])
        actor_xy = raw["xy"][frame, actor_slot].astype(np.float32)
        actor_velocity, actor_old_velocity = _velocity(raw, frame, actor_slot, actor_segment)
        last = last_index[actor_slot]
        known = (last >= 0) & current_roster
        safe_last = np.maximum(last, 0)
        candidate_index = np.arange(len(current_segments), dtype=np.int64)
        seen_xy = positions[safe_last, candidate_index].copy()
        seen_xy[~known] = 0.0
        seen_age = np.where(known, (frame - history_frames[safe_last]) / 25.0, 2.0).astype(np.float32)
        prior = prior_index[actor_slot]
        has_prior = prior >= 0
        safe_prior = np.maximum(prior, 0)
        elapsed = (history_frames[safe_last] - history_frames[safe_prior]).astype(np.float32) / 25.0
        candidate_velocity = np.zeros((len(current_segments), 2), dtype=np.float32)
        good_velocity = known & has_prior & (elapsed > 0)
        if np.any(good_velocity):
            candidate_velocity[good_velocity] = (
                seen_xy[good_velocity] - positions[safe_prior[good_velocity], candidate_index[good_velocity]]
            ) / elapsed[good_velocity, None]
            speeds = np.linalg.norm(candidate_velocity, axis=1)
            scale = np.minimum(1.0, 12.0 / np.maximum(speeds, 1e-6))
            candidate_velocity *= scale[:, None]
        candidate_velocity[~known] = 0.0
        seen_teams = teams_at_history[safe_last, candidate_index].copy()
        seen_roles = roles_at_history[safe_last, candidate_index].copy()
        seen_teams[~known] = 0
        seen_roles[~known] = -1

        asof_xy = raw["xy"][frame].astype(np.float32).copy()
        asof_xy[~known] = 0.0
        asof_velocity = np.zeros((len(current_segments), 2), dtype=np.float32)
        for slot in active_slots:
            slot = int(slot)
            if known[slot]:
                asof_velocity[slot], _ = _velocity(
                    raw, frame, slot, int(current_segments[slot])
                )
        asof_features = _features_from_estimates(
            actor_slot, actor_xy, actor_velocity, actor_old_velocity, actor_segment,
            active_slots, asof_xy, asof_velocity, np.zeros_like(seen_age),
            np.zeros_like(seen_age), known, seen_teams, seen_roles,
            int(raw["teams"][frame, actor_slot]), int(raw["roles"][frame, actor_slot]),
            order_reference_xy=seen_xy,
        )
        last_features = _features_from_estimates(
            actor_slot, actor_xy, actor_velocity, actor_old_velocity, actor_segment,
            active_slots, seen_xy, candidate_velocity, seen_age,
            np.zeros_like(seen_age), known, seen_teams, seen_roles,
            int(raw["teams"][frame, actor_slot]), int(raw["roles"][frame, actor_slot]),
            order_reference_xy=seen_xy,
        )
        predicted_xy = seen_xy + candidate_velocity * seen_age[:, None]
        predicted_xy[~known] = 0.0
        uncertainty = np.clip(
            0.5 + 0.8 * seen_age + 0.15 * np.linalg.norm(candidate_velocity, axis=1) * seen_age,
            0.5, 10.0,
        ).astype(np.float32)
        uncertainty[~known] = 10.0
        uncertainty_features = _features_from_estimates(
            actor_slot, actor_xy, actor_velocity, actor_old_velocity, actor_segment,
            active_slots, predicted_xy, candidate_velocity, seen_age,
            uncertainty, known, seen_teams, seen_roles,
            int(raw["teams"][frame, actor_slot]), int(raw["roles"][frame, actor_slot]),
            order_reference_xy=seen_xy,
        )
        asof_map[actor_slot] = asof_features
        observer_map[actor_slot] = last_features
        uncertainty_map[actor_slot] = uncertainty_features
    return asof_map, observer_map, uncertainty_map


def _labels_for_query(
    query: int, future_rows: list[dict], active_slots: np.ndarray, segments: np.ndarray,
) -> dict[int, tuple[int, float]]:
    by_actor: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in future_rows:
        if row["known_actor"]:
            by_actor[(int(row["slot"]), int(row["segment"]))].append(row)
    labels = {}
    for slot in active_slots:
        key = (int(slot), int(segments[slot]))
        actor_rows = by_actor.get(key, [])
        if not actor_rows:
            labels[int(slot)] = (0, 1.0)
            continue
        earliest = min(row["frame"] for row in actor_rows)
        at_earliest = [row for row in actor_rows if row["frame"] == earliest]
        types = {int(row["label"]) for row in at_earliest}
        if len(types) != 1:
            labels[int(slot)] = (-1, float("nan"))
        else:
            labels[int(slot)] = (
                next(iter(types)), max(0.0, (earliest - query) / 25.0)
            )
    return labels


def _future_event_rows(events: list[dict], event_frames: list[int], query: int) -> list[dict]:
    return events[
        bisect.bisect_right(event_frames, query):
        bisect.bisect_right(event_frames, query + HORIZON_FRAMES)
    ]


def _future_window_has_unmapped_actor(future_rows: list[dict]) -> bool:
    return any(not row["known_actor"] for row in future_rows)


def _build_half(match: str, half: str) -> dict:
    key = f"{match}_{half}"
    cache_path = ROOT / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{key}.npz"
    meta_path = cache_path.with_suffix(".json")
    assoc_path = ROOT / "outputs" / "trajectory_audit_20261016" / f"associations_{key}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assoc = json.loads(assoc_path.read_text(encoding="utf-8"))
    cache_hash = sha256_file(cache_path)
    if cache_hash != meta["cache_sha256"] or cache_hash != assoc["cache_sha256"]:
        raise ValueError(f"Source tracking provenance mismatch: {key}")
    with np.load(cache_path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in ("xy", "observed", "segments", "teams", "roles")}
    slots_by_id: dict[str, int] = {str(pid): int(slot) for pid, slot in assoc["slots"].items()}
    ids_by_slot: dict[int, list[str]] = defaultdict(list)
    for player_id, slot in slots_by_id.items():
        ids_by_slot[slot].append(player_id)
    unique_slots = {slot for slot, ids in ids_by_slot.items() if len(ids) == 1}
    id_to_slot = slots_by_id
    events = _event_rows(
        match, half, meta["gsr_info"], len(raw["observed"]),
        id_to_slot, unique_slots, raw,
    )
    event_frames = [row["frame"] for row in events]
    split = MATCH_SPLIT[match]
    query_rows: list[tuple[int, int, int, int, int, float]] = []
    feature_rows = {name: [] for name in (
        "full_tracking", "causal_last_seen", "causal_uncertainty"
    )}
    counters = Counter()
    scene_id = 0
    h = HORIZON_FRAMES
    for query in range(HISTORY_FRAMES, len(raw["observed"]) - h, CADENCE_FRAMES):
        active_all = np.flatnonzero(raw["observed"][query])
        counters["scheduled_actor_time_rows"] += len(active_all)
        active = active_all.astype(np.int64)
        counters["unique_identity_actor_time_rows"] += len(active)
        label_active = np.asarray(
            [slot for slot in active if int(slot) in unique_slots], dtype=np.int64
        )
        counters["identity_unavailable_actor_rows"] += len(active) - len(label_active)
        future = _future_event_rows(events, event_frames, query)
        if _future_window_has_unmapped_actor(future):
            counters["censored_unknown_actor_rows"] += len(active)
            scene_id += 1
            continue
        labels = _labels_for_query(query, future, label_active, raw["segments"][query])
        valid_active = np.asarray(
            [slot for slot in label_active if labels[int(slot)][0] >= 0], dtype=np.int64
        )
        counters["censored_ambiguous_label_rows"] += len(label_active) - len(valid_active)
        if not len(valid_active):
            scene_id += 1
            continue
        asof_features, partial_last, partial_unc = _partial_feature_pair(raw, query, active)
        for actor_slot in valid_active:
            actor_slot = int(actor_slot)
            target, lead = labels[actor_slot]
            if target == 0:
                counters["NO_ANNOTATED_EVENT_WITHIN_HORIZON"] += 1
            else:
                counters[f"positive:{CLASS_NAMES[target]}"] += 1
            feature_rows["full_tracking"].append(asof_features[actor_slot])
            feature_rows["causal_last_seen"].append(partial_last[actor_slot])
            feature_rows["causal_uncertainty"].append(partial_unc[actor_slot])
            query_rows.append((
                target, 0 if split == "fit" else 1 if split == "development" else 2,
                int(match), 1 if half == "1st" else 2, query, lead,
                actor_slot, int(raw["segments"][query, actor_slot]), scene_id,
            ))
        scene_id += 1

    if not query_rows:
        raise ValueError(f"No actor-time samples built for {key}")
    metadata_rows = np.asarray(query_rows, dtype=np.float64)
    arrays = {
        f"x_{name}": np.asarray(values, dtype=np.float16)
        for name, values in feature_rows.items()
    }
    arrays.update({
        "target": metadata_rows[:, 0].astype(np.int8),
        "split_code": metadata_rows[:, 1].astype(np.int8),
        "match": metadata_rows[:, 2].astype("U6"),
        "half_code": metadata_rows[:, 3].astype(np.int8),
        "query_frame": metadata_rows[:, 4].astype(np.int32),
        "lead_seconds": metadata_rows[:, 5].astype(np.float32),
        "actor_slot_local_only": metadata_rows[:, 6].astype(np.int16),
        "segment_local_only": metadata_rows[:, 7].astype(np.int16),
        "scene_id_local_only": metadata_rows[:, 8].astype(np.int32),
    })
    destination = COHORT_DIR / f"{key}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return {
        "match": match, "half": half, "split": split,
        "source_cache_sha256": cache_hash,
        "association_sha256": sha256_file(assoc_path),
        "shard_path": str(destination.relative_to(ROOT)),
        "shard_sha256": sha256_file(destination),
        "shard_size_bytes": destination.stat().st_size,
        "frame_count": len(raw["observed"]),
        "scheduled_query_frames": scene_id,
        "sample_rows": len(query_rows),
        "sample_class_counts": dict(sorted(Counter(int(row[0]) for row in query_rows).items())),
        "build_counters": dict(counters),
        "feature_dim": FEATURE_DIM,
        "history_frames": list(range(-HISTORY_FRAMES, 1, HISTORY_STRIDE)),
        "observation_mask": {
            "kind": "synthetic_orientation_agnostic_radial",
            "radius_metres": OBSERVATION_RANGE_M,
            "history_seconds": HISTORY_FRAMES / 25,
            "cadence_seconds": HISTORY_STRIDE / 25,
            "actor_visible_by_definition": True,
            "shared_candidate_mask_across_all_tracking_arms": True,
            "candidate_order": "shared causal last-seen geometry; identity slots never exposed",
            "human_perception_claim": False,
        },
        "feature_arm_semantics": {
            "full_tracking": "current as-of positions and velocities under the shared candidate mask",
            "causal_last_seen": "last visible position and causal finite-difference velocity",
            "causal_uncertainty": "last-seen constant-velocity extrapolation plus uncertainty proxy",
            "shared": "actor state, roster, candidate mask, team/role fields, and candidate token order",
        },
    }


def build() -> dict:
    started = time.time()
    shards = []
    for match in sorted(MATCH_SPLIT):
        for half in ("1st", "2nd"):
            shard = _build_half(match, half)
            shards.append(shard)
            print(f"built {match}_{half}: {shard['sample_rows']:,} rows")
    manifest = {
        "schema": "football.action-opportunity-cohort.2",
        "status": "COMPLETE",
        "built_unix_seconds": time.time(),
        "elapsed_seconds": time.time() - started,
        "primary_horizon_seconds": HORIZON_SECONDS,
        "query_cadence_seconds": CADENCE_FRAMES / 25,
        "actor_sampling": "currently observed tracking slots with unique target-only identity mappings; unmapped query slots are counted and excluded from supervised scoring",
        "label": "earliest supported BAS event by query actor in (t,t+1.0s]; class 0 means no annotated supported event in window, not real no action",
        "class_names": list(CLASS_NAMES),
        "split_names": ["fit", "development", "previously_exposed_retrospective"],
        "data_assumption": "KNOWN_ROSTER_PROXY for current active track candidates; no ball or source ID features",
        "shards": shards,
        "totals": {
            "samples": sum(item["sample_rows"] for item in shards),
            "fit": sum(item["sample_rows"] for item in shards if item["split"] == "fit"),
            "development": sum(item["sample_rows"] for item in shards if item["split"] == "development"),
            "previously_exposed_retrospective": sum(item["sample_rows"] for item in shards if item["split"] == "previously_exposed_retrospective"),
        },
        "private_local_only_fields": [
            "actor_slot_local_only", "segment_local_only", "scene_id_local_only",
            "query_frame", "match", "half_code",
        ],
    }
    (RUN / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    counts = Counter()
    split_counts: dict[str, Counter] = defaultdict(Counter)
    match_counts: dict[str, Counter] = defaultdict(Counter)
    for shard in shards:
        path = ROOT / shard["shard_path"]
        with np.load(path, allow_pickle=False) as data:
            target = data["target"]
            for class_index, count in zip(*np.unique(target, return_counts=True), strict=True):
                counts[int(class_index)] += int(count)
                split_counts[shard["split"]][int(class_index)] += int(count)
                match_counts[shard["match"]][int(class_index)] += int(count)
    waterfall = {
        "schema": "football.actor-time-waterfall.1",
        "supported_annotations_all_matches": 20078,
        "supported_annotations_cached_matches": 18115,
        "supported_annotations_excluded_match_118575": 1963,
        "candidates": "currently observed actors with unique target-only identity mappings on a fixed 1 s grid after 2 s history",
        "rows_by_split": {
            split: {
                "rows": int(sum(split_counts[split].values())),
                "class_counts": {
                    CLASS_NAMES[i]: int(split_counts[split][i])
                    for i in range(len(CLASS_NAMES))
                },
                "positive_rows": int(sum(
                    split_counts[split][i] for i in range(1, len(CLASS_NAMES))
                )),
                "positive_prevalence": (
                    float(sum(split_counts[split][i] for i in range(1, len(CLASS_NAMES)))
                          / max(1, sum(split_counts[split].values())))
                ),
            } for split in ("fit", "development", "previously_exposed_retrospective")
        },
        "overall_class_counts": {CLASS_NAMES[i]: int(counts[i]) for i in range(len(CLASS_NAMES))},
        "overall_rows": int(sum(counts.values())),
        "overall_positive_prevalence": float(sum(counts[i] for i in range(1, len(CLASS_NAMES))) / max(1, sum(counts.values()))),
        "by_match": {
            match: {CLASS_NAMES[i]: int(match_counts[match][i]) for i in range(len(CLASS_NAMES))}
            for match in sorted(match_counts)
        },
        "label_caveat": "The background class means no mapped supported annotation within H, not true inactivity; unknown actor windows were censored.",
    }
    (RUN / "event_waterfall.json").write_text(
        json.dumps(waterfall, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = build()
    print(json.dumps({"status": result["status"], "totals": result["totals"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
