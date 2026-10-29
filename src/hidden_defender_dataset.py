"""Exact as-of geometric labels with isolated, match-excluded learned belief inputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.belief_observation import GRID, SCALE
from src.belief_tracking_dataset import load_npz
from src.hidden_defender_geometry import geometric_targets
from src.player_centric_dataset import ROW_ID, _anonymous_slots, stable_seed
from src.trajectory_dataset import TRAIN_MATCHES, VALIDATION_MATCHES, check_deadline, file_digest, write_json
from train.belief_world_training import model_from

DEV = "132877"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def input_scene(x, mask, visible, actor_xy, last_xy):
    """No full-state arrays accepted across this student-input boundary."""
    x = x.astype(np.float32)
    known = mask[-1].astype(bool)
    current = visible[-1].astype(bool)
    hidden_defenders = known & ~current & (x[-1, :, 7] < -.5)
    visible_defenders = current & (x[-1, :, 7] < -.5)
    # Historical encoding is restricted to players visible now; hidden memory is explicit.
    sequence_mask = mask & current[None]
    sequence = np.where(sequence_mask[..., None], x, 0)
    age = np.where(known, x[-1, :, 6]*5, 0)
    return {
        "x": sequence.astype(np.float16), "mask": sequence_mask, "visible": visible & current[None],
        "actor_xy": actor_xy.astype(np.float32), "last_xy": last_xy.astype(np.float32),
        "age": age.astype(np.float32), "velocity": x[-1, :, 2:4]*12,
        "hidden_defender": hidden_defenders, "visible_defender": visible_defenders,
    }


def build_risk_dataset(root, output, deadline):
    destination = output / "dataset_v2.npz"
    if destination.exists():
        return load_npz(destination)
    old = root / "outputs" / "video_belief_world_model_20261018"
    event = load_npz(old / "events_v2.npz")
    selected = np.flatnonzero(event["action"] == 0)
    # PASS is an event-time subset, not a receiver/success label.
    exact = {}
    sources = {}
    for match in (*TRAIN_MATCHES, *VALIDATION_MATCHES):
        events = read(root / "data" / f"{match}_12_class_events.json")["actions"]
        for half in ("1st", "2nd"):
            check_deadline(deadline)
            path = root / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{match}_{half}.npz"
            sidecar = root / "outputs" / "trajectory_audit_20261016" / f"associations_{match}_{half}.json"
            association = read(sidecar)
            digest = file_digest(path)
            if digest != association["cache_sha256"] or digest != read(path.with_suffix(".json"))["cache_sha256"]:
                raise ValueError("Raw GSR cache provenance mismatch")
            sources[f"{match}_{half}"] = {"cache": digest, "association": file_digest(sidecar)}
            raw = load_npz(path)
            inverse = {int(slot): str(pid) for pid, slot in association["slots"].items()}
            for index in selected[event["matches"][selected] == match]:
                identifier = str(event["ids"][index])
                groups = ROW_ID.fullmatch(identifier).groupdict()
                if int(groups["period"]) != (1 if half == "1st" else 2):
                    continue
                annotation = events[int(groups["index"])]
                anchor = int(np.floor(float(annotation["position"])/40.))
                actor = int(association["slots"][str(annotation["player_id"])])
                current = np.flatnonzero(raw["observed"][anchor])
                if len(current) > 22 or actor not in current:
                    raise ValueError("Previously eligible event no longer matches raw current frame")
                mapping = _anonymous_slots(current.tolist(), inverse, identifier)
                xy = np.zeros((22, 2), np.float32)
                opponent = np.zeros(22, bool)
                valid = np.zeros(22, bool)
                for player, slot in mapping.items():
                    xy[slot] = raw["xy"][anchor, player]
                    valid[slot] = True
                    opponent[slot] = raw["teams"][anchor, player] != raw["teams"][anchor, actor]
                np.testing.assert_allclose(raw["xy"][anchor, actor], event["actor_xy"][index], atol=1e-6)
                exact[index] = (xy, opponent & valid, anchor)
            del raw
    scene_rows, candidate_rows, labels, provenance = [], [], [], []
    exclusions = Counter()
    for index in selected:
        check_deadline(deadline)
        x, mask, visible = event["x"][index], event["mask"][index], event["visible"][index]
        scene = input_scene(x, mask, visible, event["actor_xy"][index], event["last_xy"][index])
        xy, opponents, anchor = exact[index]
        candidates = np.flatnonzero(visible[-1] & (x[-1, :, 7] > .5) & (x[-1, :, 9] < .5))
        eligible = np.linalg.norm(xy[candidates]-event["actor_xy"][index], axis=-1) > 4
        exclusions["segment_too_short_after_endpoint_exclusion"] += int((~eligible).sum())
        candidates = candidates[eligible]
        if not opponents.any():
            exclusions["no_observed_defenders_for_ground_truth"] += 1
            continue
        if not len(candidates):
            exclusions["no_eligible_visible_candidate"] += 1
            continue
        scene_index = len(scene_rows)
        scene_rows.append(scene)
        provenance.append({"event_index": int(index), "id": str(event["ids"][index]), "match": str(event["matches"][index]),
                           "anchor": anchor, "candidates": len(candidates), "observed_defenders": int(opponents.sum())})
        for slot in candidates:
            actor, candidate = event["actor_xy"][index], xy[slot]
            target = geometric_targets(actor, candidate, xy[opponents], visible[-1, opponents],
                                       mask[-1, opponents], scene["age"][opponents])
            perturbations = []
            rng = np.random.default_rng(stable_seed(f"{event['ids'][index]}:{slot}", 919))
            for _ in range(32):
                # Every coordinate, including endpoints, receives independent bounded noise.
                dp, dq = rng.uniform(-1, 1, (2, 2))
                dd = rng.uniform(-1, 1, xy[opponents].shape)
                perturbed = geometric_targets(actor+dp, candidate+dq, xy[opponents]+dd,
                                               visible[-1, opponents], mask[-1, opponents], scene["age"][opponents])
                perturbations.append(np.r_[perturbed["lane"], perturbed["hidden"], perturbed["pressure"]])
            candidate_rows.append({
                "scene": scene_index, "slot": int(slot), "candidate_xy": candidate,
                "distance": np.float32(np.linalg.norm(candidate-actor)), "candidate_count": len(candidates),
            })
            labels.append({**target, "perturbed_mean": np.mean(perturbations, axis=0).astype(np.float32)})
    if not candidate_rows:
        raise ValueError("No eligible candidate lanes")
    data = {key: np.stack([row[key] for row in scene_rows]) for key in scene_rows[0]}
    data.update({key: np.asarray([row[key] for row in candidate_rows]) for key in candidate_rows[0]})
    data.update({f"target_{key}": np.stack([row[key] for row in labels]) for key in labels[0]})
    data["event_index"] = np.array([p["event_index"] for p in provenance])
    data["ids"] = np.array([p["id"] for p in provenance])
    data["matches"] = np.array([p["match"] for p in provenance])
    data["candidate_match"] = data["matches"][data["scene"]]
    # Full coordinates are target-only. Separate file prevents accidental use by the batch builder.
    truth = {"xy": np.stack([exact[p["event_index"]][0] for p in provenance]),
             "opponents": np.stack([exact[p["event_index"]][1] for p in provenance])}
    np.savez_compressed(output / "target_only_global_v2.npz", **truth)
    np.savez_compressed(destination, **data)
    write_json(output / "dataset_manifest.json", {
        "scenes": len(scene_rows), "candidate_lanes": len(candidate_rows),
        "PASS_event_subset": "exact old action==0 PASS class; no receiver pseudo-label used",
        "selection_exclusions": dict(exclusions), "sources": sources,
        "historical_event_sha256": file_digest(old / "events_v2.npz"),
        "primary_protocol": "original 120 degrees, 2 seconds/11 historical samples; all currently visible teammate candidates",
        "endpoint_exclusion_m": 2, "thresholds_m": [2, 3, 5],
        "truth": "raw float32 verified GSR at floor(BAS milliseconds/40); only observed defenders constitute labeled full state",
        "limitations": ["Eligibility inherits frozen event cache current-roster selection and segment filtering; causal at decision time, not a validated streaming tracker.",
                        "Untracked defenders cannot be labeled; complete observed GSR is not guaranteed physical full state."],
        "provenance": provenance,
    })
    return data


@torch.no_grad()
def infer_positions(root, output, data, phase, deadline, override_events=None, suffix=""):
    destination = output / f"{phase}_positions_v2{suffix}.npz"
    if destination.exists():
        saved = load_npz(destination)
        if not np.array_equal(saved["ids"], data["ids"]):
            raise ValueError("Position cache scene identities changed")
        for fold in read(destination.with_suffix(".json")):
            if file_digest(Path(fold["checkpoint"])) != fold["sha256"]:
                raise ValueError("Frozen world checkpoint changed after position inference")
        return saved
    old = root / "outputs" / "video_belief_world_model_20261018"
    events = override_events if override_events is not None else load_npz(old / "events_v2.npz")
    config = read(old / "world_selection.json")["selected"]["config"]
    n = len(data["ids"])
    means, stds, occupancy = np.zeros((n, 22, 2), np.float32), np.zeros((n, 22, 2), np.float32), np.zeros((n, 294), np.float32)
    covered, folds = np.zeros(n, bool), []
    for target in (*TRAIN_MATCHES, "retrospective"):
        indices = np.flatnonzero(np.isin(data["matches"], VALIDATION_MATCHES) if target == "retrospective" else data["matches"] == target)
        if not len(indices) or (target == "retrospective" and phase == "development"):
            continue
        if target == "retrospective":
            directory = old / "worlds" / "final_seed42"
        elif target == DEV:
            directory = old / "worlds" / config["name"]
        else:
            directory = old / "worlds" / f"{phase}_oof_{target}"
        spec = read(directory / "config.json")
        fitting = set(spec["training_sources"])
        predicted = set(data["matches"][indices])
        if fitting & predicted or fitting & set(VALIDATION_MATCHES) or (phase == "development" and DEV in fitting):
            raise ValueError("Position teacher violates match exclusion")
        model = model_from(config).cuda()
        checkpoint = directory / "final.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
        model.eval()
        for start in range(0, len(indices), 64):
            check_deadline(deadline)
            ix = indices[start:start+64]
            rows = data["event_index"][ix]
            x = events["x"][rows].astype(np.float32)
            x[..., 6] *= 5/8
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(torch.from_numpy(x).cuda(), torch.from_numpy(events["mask"][rows]).cuda(),
                               torch.from_numpy(events["actor_xy"][rows]).cuda(), torch.from_numpy(events["last_xy"][rows]).cuda())
            means[ix] = result["position_mean"].float().cpu().numpy()
            stds[ix] = result["position_log_std"].float().exp().cpu().numpy()
            occupancy[ix] = result["occupancy"].float().sigmoid().cpu().numpy()
            covered[ix] = True
        folds.append({"fitting_matches": sorted(fitting), "predicted_matches": sorted(predicted),
                      "checkpoint": str(checkpoint), "sha256": file_digest(checkpoint)})
        del model
        torch.cuda.empty_cache()
    result = {"mean": means, "std": stds, "occupancy": occupancy, "covered": covered, "ids": data["ids"]}
    write_json(destination.with_suffix(".json"), folds)
    np.savez_compressed(destination, **result)
    return result


def anonymous_occupancy(data, learned):
    """Residual all-player occupancy, then an explicit .5 opponent mixture assumption."""
    residual = np.zeros_like(learned["occupancy"])
    for i in range(len(residual)):
        known_hidden = data["age"][i] > 0
        # Use observed-memory masks, never a true hidden team count or hidden identity.
        known_hidden &= ~data["visible"][i, -1]
        means = learned["mean"][i, known_hidden]
        stds = np.maximum(learned["std"][i, known_hidden], .25)
        if len(means):
            delta = (GRID[None]-means[:, None])/stds[:, None]
            intensity = (np.exp(-.5*(delta**2).sum(-1))*25/(2*np.pi*stds.prod(-1)[:, None])).sum(0)
        else:
            intensity = np.zeros(294)
        total = -np.log1p(-np.clip(learned["occupancy"][i], 0, .999))
        residual[i] = -np.expm1(-.5*np.maximum(total-intensity, 0))
    return residual
