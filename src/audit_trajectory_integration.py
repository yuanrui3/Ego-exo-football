"""Read-only trajectory causality, selection and receiver-association audit."""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ijson
import numpy as np
import torch

from src.make_temporal_sft_dataset import prepare_entities, serialize_temporal_state
from src.models.trajectory_forecasting import TrajectoryForecaster
from src.receiver_tensor_dataset import read_receiver_rows, validate_match_split
from src.trajectory_dataset import (
    DEVELOPMENT_MATCH, TRAIN_MATCHES, VALIDATION_MATCHES,
    file_digest, make_window, source_signature, write_json,
)

MATCHES = (*TRAIN_MATCHES, *VALIDATION_MATCHES)
OFFSETS = np.array([10, 25, 50, 75])
INPUT_KEYS = ("history", "current", "velocity", "team", "role", "player_mask")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Audit budget reached; completed half evidence retained for resume.")


def reconstruct_history(raw, anchor):
    """Independent explicit-time reconstruction. Does not read any future frame."""
    selected = np.flatnonzero(raw["observed"][anchor])
    if anchor < 50 or not 1 <= len(selected) <= 22:
        raise ValueError("Insufficient frame span or invalid present player count")
    n = len(selected)
    history = np.zeros((22, 51, 5), np.float32)
    current = np.zeros((22, 2), np.float32)
    velocity = np.zeros((22, 2), np.float32)
    seen = np.zeros((22, 51), bool)
    current[:n] = raw["xy"][anchor, selected]
    for player, slot in enumerate(selected):
        segment = int(raw["segments"][anchor, slot])
        points = np.zeros((51, 2), np.float32)
        for index, frame in enumerate(range(anchor - 50, anchor + 1)):
            valid = bool(raw["observed"][frame, slot]) and int(raw["segments"][frame, slot]) == segment
            seen[player, index] = valid
            if valid:
                points[index] = raw["xy"][frame, slot]
                history[player, index, :2] = (points[index] - current[player]) / 10
                history[player, index, 4] = 1
            if index and seen[player, index - 1] and valid:
                delta = (points[index] - points[index - 1]) * 25
                speed = np.linalg.norm(delta)
                history[player, index, 2:4] = delta / max(1., float(speed) / 12) / 12
        times = np.arange(11, dtype=np.float32) / 25
        valid = seen[player, -11:]
        if valid.sum() >= 6:
            center = (times * valid).sum() / valid.sum()
            x = (times - center) * valid
            estimate = (points[-11:] * x[:, None]).sum(0) / max(float((x * x).sum()), 1e-8)
            velocity[player] = estimate / max(1., float(np.linalg.norm(estimate)) / 12)
    team, role = np.zeros(22, np.int64), np.zeros(22, np.int64)
    team[:n], role[:n] = raw["teams"][anchor, selected], raw["roles"][anchor, selected]
    return {
        "history": history, "current": current, "velocity": velocity, "team": team, "role": role,
        "player_mask": np.arange(22) < n,
    }, selected, seen


def reconstruct_sample(raw, anchor):
    inputs, selected, seen = reconstruct_history(raw, anchor)
    target = np.zeros((22, 4, 2), np.float32)
    valid = np.zeros((22, 4), bool)
    for player, slot in enumerate(selected):
        for horizon, offset in enumerate((10, 25, 50, 75)):
            frame = anchor + offset
            good = bool(raw["observed"][frame, slot]) and (
                raw["segments"][frame, slot] == raw["segments"][anchor, slot]
            )
            valid[player, horizon] = good
            if good:
                target[player, horizon] = raw["xy"][frame, slot] - raw["xy"][anchor, slot]
    return {**inputs, "target": target, "target_mask": valid}, selected, seen


def selection_arrays(raw):
    frames = np.arange(50, len(raw["xy"]) - 75, 13)
    present = raw["observed"][frames]
    counts = present.sum(1)
    count_rule = (counts >= 18) & (counts <= 22)
    team_over = np.zeros(len(frames), bool)
    for team in (1, 2):
        team_over |= (present & (raw["teams"][frames] == team)).sum(1) > 11
    future_good = present & raw["observed"][frames + 75] & (
        raw["segments"][frames + 75] == raw["segments"][frames]
    )
    future_count = future_good.sum(1)
    future_rule = future_count >= 12
    accepted = count_rule & ~team_over & future_rule
    history_count = np.zeros(len(frames), np.int32)
    any_missing_history = np.zeros(len(frames), bool)
    for offset in range(-50, 1):
        good = present & raw["observed"][frames + offset] & (
            raw["segments"][frames + offset] == raw["segments"][frames]
        )
        history_count += good.sum(1)
        any_missing_history |= (present & ~good).any(1)
    history_fraction = history_count / np.maximum(counts * 51, 1)
    return {
        "frames": frames, "counts": counts, "count_rule": count_rule,
        "team_over": team_over, "future_count": future_count,
        "future_rule": future_rule, "accepted": accepted,
        "history_fraction": history_fraction, "any_missing_history": any_missing_history,
    }


def quality_subset(arrays, flag):
    count = int(flag.sum())
    if not count:
        return {"windows": 0}
    return {
        "windows": count,
        "mean_observed_players": float(arrays["counts"][flag].mean()),
        "mean_history_observation_fraction": float(arrays["history_fraction"][flag].mean()),
        "any_missing_history_fraction": float(arrays["any_missing_history"][flag].mean()),
        "mean_valid_three_second_players": float(arrays["future_count"][flag].mean()),
    }


def selection_summary(raw):
    a = selection_arrays(raw)
    n = len(a["frames"])
    fail_count = ~a["count_rule"]
    fail_team = a["count_rule"] & a["team_over"]
    fail_future = a["count_rule"] & ~a["team_over"] & ~a["future_rule"]
    result = {
        "frames": len(raw["xy"]), "candidate_windows": n,
        "eligible": int(a["accepted"].sum()),
        "excluded_player_count_sequential": int(fail_count.sum()),
        "excluded_team_count_sequential": int(fail_team.sum()),
        "excluded_future_continuity_sequential": int(fail_future.sum()),
        "fails_player_count_independent": int(fail_count.sum()),
        "fails_team_count_independent": int(a["team_over"].sum()),
        "fails_future_continuity_independent": int((~a["future_rule"]).sum()),
        "observed_count_distribution": {str(k): int(v) for k, v in zip(*np.unique(a["counts"], return_counts=True))},
        "included": quality_subset(a, a["accepted"]),
        "excluded": quality_subset(a, ~a["accepted"]),
        "difficulty_groups": {},
    }
    for label, flag in {
        "current_players_under_22": a["counts"] < 22,
        "any_missing_history": a["any_missing_history"],
        "fewer_than_22_valid_future_players": a["future_count"] < 22,
        "fewer_than_12_valid_future_players": a["future_count"] < 12,
    }.items():
        count = int(flag.sum())
        result["difficulty_groups"][label] = {
            "candidates": count, "excluded": int((flag & ~a["accepted"]).sum()),
            "exclusion_fraction": float((flag & ~a["accepted"]).sum() / count) if count else None,
        }
    assert result["eligible"] + int(fail_count.sum() + fail_team.sum() + fail_future.sum()) == n
    return result, a


def raw_associations(path, wanted_frames, deadline):
    """One raw stream per half recovers identity order and sparse verification records."""
    slot_by_player, states = {}, defaultdict(list)
    with Path(path).open("rb") as stream:
        for index, ann in enumerate(ijson.items(stream, "annotations.item", use_float=True)):
            if index % 10000 == 0:
                deadline_check(deadline)
            attrs = ann.get("attributes") or {}
            if ann.get("supercategory") != "object" or attrs.get("role") not in ("player", "goalkeeper"):
                continue
            player, track = attrs.get("player_id"), ann.get("track_id")
            if player in (None, "", -1, "-1") or track is None:
                continue
            player = str(player)
            slot_by_player.setdefault(player, len(slot_by_player))
            text = str(ann["image_id"])
            if not text.isdigit() or len(text) < 7:
                raise ValueError(f"Unexpected raw image ID: {text}")
            frame = int(text[-6:]) - 1
            if frame not in wanted_frames:
                continue
            pitch = ann.get("bbox_pitch") or {}
            x, y = pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")
            if x is None or y is None:
                continue
            x, y = float(x), float(y)
            if not np.isfinite([x, y]).all():
                continue
            states[frame].append({
                "player_id": player, "track_id": str(track), "team": attrs.get("team"),
                "role": attrs["role"], "x": x, "y": y,
            })
    return slot_by_player, dict(states)


def verify_sparse_raw(raw, slots, states):
    compared, max_error = 0, 0.
    for frame, records in states.items():
        groups = defaultdict(list)
        for record in records:
            groups[record["player_id"]].append(record)
        for player, records_for_player in groups.items():
            slot = slots[player]
            if len(records_for_player) != 1 or not raw["observed"][frame, slot]:
                continue
            record = records_for_player[0]
            xy = np.array([record["x"], record["y"]], np.float32)
            error = float(np.abs(xy - raw["xy"][frame, slot]).max())
            max_error = max(error, max_error)
            if error > 1e-5:
                raise AssertionError(f"Raw/cache coordinate mismatch at {frame}/{player}: {error}")
            if raw["teams"][frame, slot] != {"left": 1, "right": 2}[record["team"]]:
                raise AssertionError("Raw/cache team mismatch")
            if raw["roles"][frame, slot] != int(record["role"] == "goalkeeper"):
                raise AssertionError("Raw/cache role mismatch")
            compared += 1
    return {"raw_player_frames_checked": compared, "max_coordinate_difference_m": max_error}


def receiver_mapping(row, frame, states, slots):
    snapshots = {offset: states.get(frame + offset, []) for offset in (-10, -5, 0)}
    record = {"actor_id": row["passer_id"], "period": int(row["id"].split("_p")[1][0])}
    actor_team, entities = prepare_entities(record, snapshots[0])
    state = serialize_temporal_state(record["period"], actor_team, entities, snapshots)
    frozen_state = row["prompt"].split("<|user|>\n", 1)[1].split("<|assistant|>", 1)[0].strip()
    matches = state.strip() == frozen_state
    mapping = {entity["label"]: slots[entity["key"]] for entity in entities}
    return mapping, matches


def receiver_diagnostic(row, raw, slots, states, bas_event):
    frame = int(row["id"].split("_f")[1].split("_")[0])
    result = {
        "id": row["id"], "match": row["match_id"], "frame": frame,
        "span_51_available": frame >= 50,
        "count_now": int(raw["observed"][frame].sum()),
        "mapping_matches_frozen_prompt": False, "masked_inputs_usable": False,
        "all_ten_candidates_now": False, "all_defenders_now": False,
        "all_ten_candidate_history_complete": False, "all_player_history_complete": False,
        "all_candidate_velocities_supported": False, "all_defender_velocities_supported": False,
        "future_benchmark_eligible": False, "failure_reason": "",
        "source_frame_seconds": frame / 25,
        "bas_seconds": float(bas_event["position"]) / 1000,
        "source_minus_bas_ms": round(frame * 40 - float(bas_event["position"]), 6),
    }
    mapping, matches = receiver_mapping(row, frame, states, slots)
    result["mapping_matches_frozen_prompt"] = matches
    candidates = [mapping[f"TEAMMATE_{i}"] for i in range(1, 11) if f"TEAMMATE_{i}" in mapping]
    defenders = [slot for label, slot in mapping.items() if label.startswith("OPPONENT_")]
    result["all_ten_candidates_now"] = len(candidates) == 10 and bool(raw["observed"][frame, candidates].all())
    result["defender_count"] = len(defenders)
    result["all_defenders_now"] = len(defenders) == 11 and bool(raw["observed"][frame, defenders].all())
    if not matches:
        result["failure_reason"] = "raw_scene_does_not_reproduce_frozen_prompt"
        return result
    if not result["span_51_available"]:
        result["failure_reason"] = "half_start_less_than_50_frames"
        return result
    if result["count_now"] > 22 or not result["all_ten_candidates_now"]:
        result["failure_reason"] = "missing_candidates_or_over_capacity"
        return result
    history_frames = np.arange(frame - 50, frame + 1)
    current_slots = np.flatnonzero(raw["observed"][frame])
    observed = raw["observed"][history_frames[:, None], current_slots]
    continuous = observed & (
        raw["segments"][history_frames[:, None], current_slots] == raw["segments"][frame, current_slots]
    )
    local_slot = {int(slot): index for index, slot in enumerate(current_slots)}
    c = [local_slot[slot] for slot in candidates]
    d = [local_slot[slot] for slot in defenders if slot in local_slot]
    result["masked_inputs_usable"] = True
    result["all_ten_candidate_history_complete"] = bool(continuous[:, c].all())
    result["all_player_history_complete"] = bool(continuous.all())
    result["all_candidate_velocities_supported"] = bool((continuous[-11:, c].sum(0) >= 6).all())
    result["all_defender_velocities_supported"] = len(d) == 11 and bool((continuous[-11:, d].sum(0) >= 6).all())
    result["candidate_history_fraction"] = float(continuous[:, c].mean())
    result["all_player_history_fraction"] = float(continuous.mean())
    result["actor_history_complete"] = bool(continuous[:, local_slot[mapping["ACTOR"]]].all())
    if frame + 75 < len(raw["xy"]):
        future_count = int((
            raw["observed"][frame + 75, current_slots]
            & (raw["segments"][frame + 75, current_slots] == raw["segments"][frame, current_slots])
        ).sum())
        team_ok = all((raw["teams"][frame, current_slots] == team).sum() <= 11 for team in (1, 2))
        result["future_benchmark_eligible"] = 18 <= len(current_slots) <= 22 and team_ok and future_count >= 12
    return result


def summarize_receivers(rows):
    boolean_fields = (
        "span_51_available", "mapping_matches_frozen_prompt", "masked_inputs_usable",
        "all_ten_candidates_now", "all_defenders_now", "all_ten_candidate_history_complete",
        "all_player_history_complete", "all_candidate_velocities_supported",
        "all_defender_velocities_supported", "future_benchmark_eligible",
    )
    n = len(rows)
    return {
        "examples": n,
        **{key: {"count": sum(bool(r[key]) for r in rows),
                 "fraction": sum(bool(r[key]) for r in rows) / n} for key in boolean_fields},
        "failure_reasons": dict(Counter(r["failure_reason"] for r in rows if r["failure_reason"])),
        "source_minus_bas_ms_distribution": dict(sorted(Counter(str(r["source_minus_bas_ms"]) for r in rows).items())),
        "usable_despite_future_benchmark_ineligibility": sum(
            r["masked_inputs_usable"] and not r["future_benchmark_eligible"] for r in rows
        ),
    }


def audit_saved_selection(experiment):
    progress = read_json(experiment / "progress.json")
    selected = read_json(experiment / "selected_configurations.json")
    checked = []
    for name, recorded_hash in progress["code_sha256"].items():
        if file_digest(Path(name)) != recorded_hash:
            raise AssertionError(f"Completed-experiment code differs from launch: {name}")
    for family, selection in selected.items():
        path = Path(selection["directory"])
        history = read_json(path / "history.json")
        config = read_json(path / "config.json")
        best, epoch, stale = float("inf"), 0, 0
        for row in history:
            if row["development_ade_m"] < best - config["min_delta_m"]:
                best, epoch, stale = row["development_ade_m"], row["epoch"], 0
            else:
                stale += 1
        assert math.isclose(best, selection["fit"]["best_development_ade_m"])
        assert epoch == selection["epochs"]
        candidates = []
        for directory in sorted((experiment / "selection").glob(f"{family}_*")):
            fit = read_json(directory / "fit_result.json")
            candidates.append((fit["best_development_ade_m"], fit["parameters"], directory))
        assert Path(min(candidates, key=lambda value: value[:2])[2]) == path
        for seed in (42, 7, 123):
            final = read_json(experiment / "fits" / f"{family}_seed{seed}" / "config.json")
            assert final["epochs"] == epoch
            assert all(final[k] == selection["candidate"][k] for k in selection["candidate"])
        checked.append({"family": family, "selected_epoch": epoch, "dev_ade_m": best})
    best_probabilistic = read_json(Path(selected["probabilistic"]["directory"]) / "best_development_metrics.json")
    radius = best_probabilistic["uncertainty"]["mean_squared_standardized_radius"]
    temperature = max(.05, math.sqrt(np.mean(radius) / 2))
    assert math.isclose(float(temperature), selected["probabilistic"]["temperature"], rel_tol=1e-6)
    return {
        "launch_code_and_config_hashes_match": True, "families": checked,
        "development_calibration_temperature": float(temperature),
        "calibration_matches_development_checkpoint_metrics": True,
        "selection_fitting_matches": [m for m in TRAIN_MATCHES if m != DEVELOPMENT_MATCH],
        "development_matches": [DEVELOPMENT_MATCH], "heldout_matches": list(VALIDATION_MATCHES),
        "caution": "Calibration transferred from six-match development model to seven-match refits; not "
                   "a guarantee of held-out or per-seed calibration.",
    }


def checkpoint_causality(experiment, sample):
    selected = read_json(experiment / "selected_configurations.json")["social"]
    model = TrajectoryForecaster(
        width=selected["candidate"]["width"], layers=selected["candidate"]["layers"]
    ).eval()
    model.load_state_dict(torch.load(experiment / "fits/social_seed42/final.pt", map_location="cpu", weights_only=True))
    batch = {key: torch.from_numpy(value)[None] for key, value in sample.items()}
    other = {key: value.clone() for key, value in batch.items()}
    other["target"][:] = 1e6
    other["target_mask"][:] = False
    with torch.no_grad():
        a, _ = model(batch)
        b, _ = model(other)
        delta = float((a - b).abs().max())
    assert delta == 0
    return {"actual_social_seed42_target_and_target_mask_mutation_max_forecast_delta": delta}


def receiver_inference_checks(experiment, output, diagnostic_rows, receiver_rows):
    """Inference only: truncate raw arrays at the decision and never construct targets."""
    eligible = [r for r in diagnostic_rows if r["masked_inputs_usable"]]
    rng = np.random.default_rng(20261016)
    chosen = [eligible[int(i)] for i in rng.choice(len(eligible), 12, replace=False)]
    # Also exercise fewer-than-ten candidates and missing-defender cases explicitly.
    for predicate in (
        lambda r: r["span_51_available"] and not r["all_ten_candidates_now"],
        lambda r: r["masked_inputs_usable"] and not r["all_defenders_now"],
    ):
        found = next((r for r in diagnostic_rows if predicate(r)), None)
        if found and found not in chosen:
            chosen.append(found)
    lookup = {row["id"]: row for row in receiver_rows}
    selected = read_json(experiment / "selected_configurations.json")["social"]["candidate"]
    model = TrajectoryForecaster(width=selected["width"], layers=selected["layers"]).eval()
    model.load_state_dict(torch.load(
        experiment / "fits/social_seed42/final.pt", map_location="cpu", weights_only=True
    ))
    evidence = []
    for diagnostic in chosen:
        row, anchor = lookup[diagnostic["id"]], diagnostic["frame"]
        half = "1st" if row["id"].split("_p")[1][0] == "1" else "2nd"
        name = f"{row['match_id']}_{half}"
        with np.load(experiment / "cache" / f"{name}.npz") as blob:
            # Physically no post-decision entries are accessible to the input adapter.
            raw = {key: blob[key][:anchor + 1] for key in blob.files}
        association = read_json(output / f"associations_{name}.json")
        states = {int(k): value for k, value in association["states"].items()}
        mapping, matches = receiver_mapping(row, anchor, states, association["slots"])
        assert matches
        inputs, slots, _ = reconstruct_history(raw, anchor)
        assert set(inputs) == set(INPUT_KEYS)
        batch = {key: torch.from_numpy(value)[None] for key, value in inputs.items()}
        with torch.no_grad():
            prediction, _ = model(batch)
        assert tuple(prediction.shape) == (1, 22, 4, 2) and torch.isfinite(prediction).all()
        forecast = inputs["current"][:, None] + prediction[0].numpy()
        local = {int(slot): index for index, slot in enumerate(slots)}
        defenders = [local[slot] for label, slot in mapping.items()
                     if label.startswith("OPPONENT_") and slot in local]
        candidates = [local[mapping[f"TEAMMATE_{i}"]] for i in range(1, 11)
                      if f"TEAMMATE_{i}" in mapping and mapping[f"TEAMMATE_{i}"] in local]
        distances = np.linalg.norm(
            forecast[candidates, None] - forecast[None, defenders], axis=-1
        ).min(axis=1)
        assert distances.shape == (len(candidates), 4) and np.isfinite(distances).all()
        evidence.append({
            "id": row["id"], "anchor": anchor, "latest_available_frame": anchor,
            "candidates_forecast": len(candidates), "observed_defenders_forecast": len(defenders),
            "forecast_and_nearest_observed_defender_features_finite": True,
            "future_positions_or_targets_supplied_to_model": False,
        })
    result = {
        "status": "PASS", "checkpoint": "social_seed42/final.pt", "examples": evidence,
        "count": len(evidence),
        "caution": "In-sample checkpoint used ONLY to test interface/finite outputs, not receiver accuracy "
                   "or feature usefulness. OOF forecasting is still required for training.",
    }
    write_json(output / "receiver_inference_checks.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("outputs/trajectory_forecasting_20261016"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trajectory_audit_20261016"))
    parser.add_argument("--minutes", type=float, default=85)
    args = parser.parse_args()
    if not 0 < args.minutes <= 100:
        parser.error("Audit execution budget must be >0 and <=100 minutes")
    started = time.monotonic()
    deadline = started + args.minutes * 60
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows_by_half = defaultdict(list)
    train_rows = read_receiver_rows(Path("data/train_pass_receiver.jsonl"))
    val_rows = read_receiver_rows(Path("data/val_pass_receiver.jsonl"))
    validate_match_split(train_rows, val_rows)
    assert len(train_rows) == 4799 and len(val_rows) == 1224
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        for row in rows:
            period = row["id"].split("_p")[1][0]
            rows_by_half[(str(row["match_id"]), "1st" if period == "1" else "2nd")].append((split, row))
    protected = [
        Path("data/train_pass_receiver.jsonl"), Path("data/val_pass_receiver.jsonl"),
        Path("results/trajectory_forecasting_metrics.json"), Path("results/trajectory_forecasting_report.md"),
    ]
    before = {str(p): file_digest(p) for p in protected}
    weights_before = {str(p): source_signature(p) for p in args.experiment.glob("**/*.pt")}
    selection = audit_saved_selection(args.experiment)
    write_json(args.output / "saved_selection_verification.json", selection)
    all_candidates, selection_rows, sample_by_half = [], {}, defaultdict(list)
    for match in MATCHES:
        for half in ("1st", "2nd"):
            with np.load(args.experiment / "cache" / f"{match}_{half}.npz", allow_pickle=False) as blob:
                raw = {k: blob[k] for k in blob.files}
            stats, arrays = selection_summary(raw)
            selection_rows[f"{match}_{half}"] = stats
            all_candidates.extend((match, half, int(f)) for f in arrays["frames"][arrays["accepted"]])
    rng = np.random.default_rng(20261016)
    for index in rng.choice(len(all_candidates), 20, replace=False):
        match, half, frame = all_candidates[int(index)]
        sample_by_half[(match, half)].append(frame)
    write_json(args.output / "random_sample_plan.json", {
        "seed": 20261016, "population": len(all_candidates),
        "by_half": {f"{m}_{h}": f for (m, h), f in sample_by_half.items()},
    })
    receiver_rows, samples, raw_checks = [], [], {}
    last_sample = None
    for match in MATCHES:
        bas = read_json(Path("data") / f"{match}_12_class_events.json")
        assert bas["clock"] == "ms from the first frame of the half's released video"
        events = bas.get("actions") or bas.get("annotations")
        for half in ("1st", "2nd"):
            deadline_check(deadline)
            name = f"{match}_{half}"
            print(f"AUDIT {name}", flush=True)
            path = args.experiment / "cache" / f"{name}.npz"
            meta = read_json(path.with_suffix(".json"))
            assert file_digest(path) == meta["cache_sha256"]
            assert source_signature(meta["source"]["path"]) == meta["source"]
            with np.load(path, allow_pickle=False) as blob:
                raw = {k: blob[k] for k in blob.files}
            frame_set = set()
            for split, row in rows_by_half[(match, half)]:
                frame = int(row["id"].split("_f")[1].split("_")[0])
                frame_set.update((frame - 10, frame - 5, frame))
            for frame in sample_by_half[(match, half)]:
                frame_set.update(range(frame - 50, frame + 1))
                frame_set.update(int(frame + offset) for offset in OFFSETS)
            association_path = args.output / f"associations_{name}.json"
            if association_path.exists():
                cached = read_json(association_path)
                assert cached["source"] == meta["source"] and cached["cache_sha256"] == meta["cache_sha256"]
                assert set(cached["wanted_frames"]) == frame_set
                slots = cached["slots"]
                states = {int(k): value for k, value in cached["states"].items()}
            else:
                slots, states = raw_associations(meta["source"]["path"], frame_set, deadline)
                write_json(association_path, {
                    "source": meta["source"], "cache_sha256": meta["cache_sha256"],
                    "wanted_frames": sorted(frame_set), "slots": slots, "states": states,
                })
            assert len(slots) == raw["xy"].shape[1]
            raw_checks[name] = verify_sparse_raw(raw, slots, states)
            for frame in sample_by_half[(match, half)]:
                independent, selected, seen = reconstruct_sample(raw, frame)
                original = make_window(raw, frame)
                errors = {}
                for key in original:
                    errors[key] = float(np.abs(
                        original[key].astype(np.float64) - independent[key].astype(np.float64)
                    ).max())
                    if key in ("history", "velocity"):
                        np.testing.assert_allclose(original[key], independent[key], atol=1e-4, rtol=1e-4)
                    else:
                        np.testing.assert_array_equal(original[key], independent[key])
                # Counterfactual future wipe must not change any model input, even if eligibility changes.
                changed = {k: v.copy() for k, v in raw.items()}
                changed["xy"][frame + 1:] = 12345
                changed["observed"][frame + 1:] = False
                changed["segments"][frame + 1:] = 0
                other = make_window(changed, frame)
                for key in INPUT_KEYS:
                    np.testing.assert_array_equal(original[key], other[key])
                samples.append({
                    "match": match, "half": half, "anchor": frame, "anchor_seconds": frame / 25,
                    "history_first_frame": frame - 50, "history_last_frame": frame,
                    "history_points": 51, "target_frames": (frame + OFFSETS).tolist(),
                    "target_seconds": ((frame + OFFSETS) / 25).tolist(),
                    "valid_targets": original["target_mask"].sum(0).tolist(),
                    "max_absolute_tensor_differences": errors, "future_input_invariance": True,
                    "raw_identity_mapping_checked": True,
                })
                last_sample = original
            for split, row in rows_by_half[(match, half)]:
                index = int(row["id"].rsplit("_", 1)[1])
                event = events[index]
                assert int(event["frame"]) == int(row["id"].split("_f")[1].split("_")[0])
                diagnostic = receiver_diagnostic(row, raw, slots, states, event)
                diagnostic["split"] = split
                receiver_rows.append(diagnostic)
            write_json(args.output / "progress.json", {
                "last_half": name, "receiver_examples": len(receiver_rows),
                "samples_checked": len(samples), "elapsed_seconds": time.monotonic() - started,
            })
    assert len(samples) == 20 and len(receiver_rows) == 6023
    causality = checkpoint_causality(args.experiment, last_sample)
    inference = receiver_inference_checks(
        args.experiment, args.output, receiver_rows, [*train_rows, *val_rows]
    )
    # Compare independently selected windows against the actual completed evaluation population.
    heldout_expected = sum(v["eligible"] for k, v in selection_rows.items() if k[:6] in VALIDATION_MATCHES)
    stored_windows = read_json(args.experiment / "evaluation/social_seed42/metrics.json")["coverage"]["windows"]
    assert heldout_expected == stored_windows
    with np.load(args.experiment / "evaluation/social_seed42/window_errors.npz") as errors:
        checked_rows = errors["rows"]
    expected_order = []
    for source, (match, half) in enumerate((m, h) for m in VALIDATION_MATCHES for h in ("1st", "2nd")):
        with np.load(args.experiment / "cache" / f"{match}_{half}.npz") as blob:
            arrays = selection_arrays({k: blob[k] for k in blob.files})
        expected_order.extend((source, int(f)) for f in arrays["frames"][arrays["accepted"]])
    np.testing.assert_array_equal(checked_rows[:, :2], np.asarray(expected_order))
    receiver_summary = {
        split: summarize_receivers([r for r in receiver_rows if r["split"] == split])
        for split in ("train", "validation")
    }
    by_match = {m: summarize_receivers([r for r in receiver_rows if r["match"] == m]) for m in MATCHES}
    protected_unchanged = before == {str(p): file_digest(p) for p in protected}
    weights_unchanged = weights_before == {str(p): source_signature(p) for p in args.experiment.glob("**/*.pt")}
    assert protected_unchanged and weights_unchanged
    write_json(args.output / "selection_coverage.json", selection_rows)
    write_json(args.output / "sample_reconstruction.json", {"samples": samples, **causality})
    write_json(args.output / "raw_cache_verification.json", raw_checks)
    write_csv(args.output / "receiver_coverage.csv", receiver_rows)
    result = {
        "status": "COMPLETE", "runtime_seconds": time.monotonic() - started,
        "saved_selection": selection, "selection": selection_rows,
        "receiver_summary": receiver_summary, "receiver_by_match": by_match,
        "independent_samples": len(samples), "raw_checks": raw_checks, "causality": causality,
        "receiver_inference_checks": inference,
        "completed_heldout_windows_match_independent_eligibility": True,
        "protected_files_unchanged": protected_unchanged, "weights_unchanged": weights_unchanged,
        "protected_sha256": before,
    }
    write_json(args.output / "audit_summary.json", result)
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"],
                      "receiver_summary": receiver_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
