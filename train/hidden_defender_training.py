"""Resume-safe candidate risk fitting, matched geometry baselines and evaluation."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

from src.belief_observation import GRID, SCALE
from src.hidden_defender_dataset import anonymous_occupancy
from src.hidden_defender_geometry import probability_from_samples
from src.models.hidden_defender_risk import HiddenDefenderRisk
from src.trajectory_dataset import check_deadline, write_json
from train.run_trajectory_forecasting import restore_rng, rng_state, save_torch, seed_all

VARIANTS = {"R0": "visible", "R1": "last_seen", "R2": "constant_velocity", "R3": "learned",
            "R4": "occupancy", "R5": "shuffled", "R6": "empty"}


def prepare_belief(data, learned):
    return {**learned, "anonymous": anonymous_occupancy(data, learned)}


def hidden_state(data, belief, scenes, kind, permutation=None):
    ix = scenes if kind != "shuffled" else permutation[scenes]
    n = len(ix)
    hidden = np.zeros((n, 22, 7), np.float32)
    mask = data["hidden_defender"][ix].copy()
    occupancy = np.zeros((n, 294), np.float32)
    if kind in ("visible", "empty"):
        mask[:] = False
        return hidden, mask, occupancy
    centre = data["last_xy"][ix].copy()
    std = np.broadcast_to((1.5+1.1*data["age"][ix])[..., None], centre.shape).copy()
    if kind == "constant_velocity":
        centre += data["velocity"][ix]*data["age"][ix, :, None]
    elif kind in ("learned", "occupancy", "shuffled"):
        if not belief["covered"][ix].all():
            raise ValueError("Uncovered learned belief would enter risk inputs")
        centre, std = belief["mean"][ix], belief["std"][ix]
    hidden[..., :2] = (centre-data["actor_xy"][scenes, None])/SCALE
    hidden[..., 2:4] = std/10
    hidden[..., 4] = data["age"][ix]/5
    hidden[..., 5] = 1
    hidden[..., 6] = np.linalg.norm(centre-data["actor_xy"][scenes, None], axis=-1)/105
    hidden[~mask] = 0
    if kind in ("occupancy", "shuffled"):
        occupancy = belief["anonymous"][ix]
    return hidden, mask, occupancy


def permutation_by_split(data, seed):
    mapping = np.arange(len(data["ids"]))
    # Within-match shuffling preserves split and match geometry population.
    for match in sorted(set(data["matches"])):
        indices = np.flatnonzero(data["matches"] == match)
        ordered = np.random.default_rng(seed).permutation(indices)
        if len(ordered) > 1:
            mapping[ordered] = np.roll(ordered, 1)
    return mapping


def make_batch(data, belief, rows, kind, permutation=None, perturb=False):
    rows = np.asarray(rows)
    scenes = data["scene"][rows]
    hidden, hidden_mask, occ = hidden_state(data, belief, scenes, kind, permutation)
    x, mask = data["x"][scenes].astype(np.float32), data["mask"][scenes].copy()
    if kind == "visible":
        x[:, :-1] = 0
        mask[:, :-1] = False
    actor = data["actor_xy"][scenes]
    candidate = data["candidate_xy"][rows]
    relative = candidate-actor
    geometry = np.c_[relative/SCALE, np.linalg.norm(relative, axis=-1)/105,
                     actor/SCALE, data["candidate_count"][rows]/10].astype(np.float32)
    if perturb:
        noise = np.stack([np.random.default_rng(772+int(row)).uniform(-1, 1, (22, 2)) for row in rows]).astype(np.float32)
        actor_slot = data["x"][scenes, -1, :, 9].argmax(-1)
        actor_noise = noise[np.arange(len(rows)), actor_slot]
        candidate_noise = noise[np.arange(len(rows)), data["slot"][rows]]
        x[..., :2] += (noise-actor_noise[:, None])[:, None]/SCALE
        hidden[..., :2] += (noise-actor_noise[:, None])/SCALE
        relative = relative+candidate_noise-actor_noise
        geometry[:, :2] = relative/SCALE
        geometry[:, 2] = np.linalg.norm(relative, axis=-1)/105
        geometry[:, 3:5] = (actor+actor_noise)/SCALE
        x[~mask] = 0
        hidden[~hidden_mask] = 0
    cpu = {
        "x": x, "mask": mask, "candidate_slot": data["slot"][rows],
        "geometry": geometry, "hidden": hidden, "hidden_mask": hidden_mask, "occupancy": occ,
    }
    return {key: torch.from_numpy(value).cuda() for key, value in cpu.items()}


@torch.no_grad()
def infer(model, data, belief, indices, kind, deadline, permutation=None, perturb=False):
    predictions = []
    model.eval()
    for start in range(0, len(indices), 256):
        check_deadline(deadline)
        batch = make_batch(data, belief, indices[start:start+256], kind, permutation, perturb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(**batch)
        predictions.append(np.c_[result["logits"].sigmoid().cpu().numpy(), result["pressure"].cpu().numpy()])
    return np.concatenate(predictions)


def targets(data, rows):
    return np.c_[data["target_lane"][rows], data["target_hidden"][rows], data["target_pressure"][rows]]


def fit(directory, data, belief, fitting, development, variant, config, deadline, epochs=None, progress=None):
    directory.mkdir(parents=True, exist_ok=True)
    fitting = np.asarray(fitting)
    development = np.asarray(development) if development is not None else None
    spec = {"config": config, "variant": variant, "epochs": epochs,
            "fitting_matches": sorted(set(data["candidate_match"][fitting])),
            "development_matches": sorted(set(data["candidate_match"][development])) if development is not None else []}
    if set(spec["fitting_matches"]) & set(spec["development_matches"]):
        raise ValueError("Risk model match-split overlap")
    if (directory / "config.json").exists() and json.loads((directory / "config.json").read_text()) != spec:
        raise ValueError("Risk resume configuration mismatch")
    write_json(directory / "config.json", spec)
    if (directory / "result.json").exists():
        if not (directory / "final.pt").exists():
            raise FileNotFoundError("Completed risk checkpoint missing")
        return json.loads((directory / "result.json").read_text())
    seed_all(config["seed"])
    model = HiddenDefenderRisk(config["width"], config["layers"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=.0001)
    state = {"epoch": 0, "offset": 0, "steps": 0, "best": 1e20, "best_epoch": 0, "stale": 0, "history": []}
    path = directory / "last.pt"
    if path.exists():
        saved = torch.load(path, weights_only=False, map_location="cpu")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        restore_rng(saved["rng"])
    permutation = permutation_by_split(data, config["seed"])
    kind = VARIANTS[variant]
    last_save = time.monotonic()
    def save():
        save_torch(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "state": state, "rng": rng_state()})
        write_json(directory / "live.json", {k: state[k] for k in ("epoch", "offset", "steps", "best_epoch", "best")})
    try:
        maximum = epochs if epochs is not None else config["max_epochs"]
        while state["epoch"] < maximum and (epochs is not None or state["stale"] < config["patience"]):
            order = np.random.default_rng(config["seed"]+state["epoch"]).permutation(fitting)
            model.train()
            for start in range(state["offset"], len(order), config["batch_size"]):
                check_deadline(deadline)
                rows = order[start:start+config["batch_size"]]
                batch = make_batch(data, belief, rows, kind, permutation)
                truth = torch.from_numpy(targets(data, rows).astype(np.float32)).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**batch)
                    loss = F.binary_cross_entropy_with_logits(result["logits"], truth[:, :6])
                    loss += .2*F.smooth_l1_loss(result["pressure"]/10, truth[:, 6]/10)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite risk training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                state["offset"] = start+len(rows)
                state["steps"] += 1
                if time.monotonic()-last_save >= 90:
                    save()
                    if progress:
                        progress(fit=str(directory), epoch=state["epoch"], steps=state["steps"])
                    last_save = time.monotonic()
            score = 0.
            if development is not None:
                prediction = infer(model, data, belief, development, kind, deadline, permutation)
                truth = targets(data, development)
                score = float(((prediction[:, [1, 4]]-truth[:, [1, 4]])**2).mean())
            state["epoch"] += 1
            state["offset"] = 0
            state["history"].append({"epoch": state["epoch"], "development_primary_brier": score})
            if development is not None:
                if score < state["best"]-1e-5:
                    state["best"], state["best_epoch"], state["stale"] = score, state["epoch"], 0
                    save_torch(directory / "best.pt", model.state_dict())
                else:
                    state["stale"] += 1
            save()
            if progress:
                progress(fit=str(directory), epoch=state["epoch"], development_primary_brier=score)
        if development is not None:
            model.load_state_dict(torch.load(directory / "best.pt", weights_only=True))
        else:
            state["best_epoch"] = state["epoch"]
        save_torch(directory / "final.pt", model.state_dict())
        result = {"status": "COMPLETE", "best_epoch": state["best_epoch"], "best_development_brier": state["best"],
                  "steps": state["steps"], "parameters": sum(p.numel() for p in model.parameters())}
        write_json(directory / "result.json", result)
        return result
    except (Exception, KeyboardInterrupt):
        save()
        raise
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def calibrate(predictions, truth):
    parameters = []
    for column in range(6):
        x = np.log(np.clip(predictions[:, column], 1e-6, 1-1e-6)/np.clip(1-predictions[:, column], 1e-6, 1))
        y = truth[:, column]
        def objective(beta):
            z = x*beta[0]+beta[1]
            return float((np.logaddexp(0, z)-y*z).mean())
        result = minimize(objective, [1., 0.], bounds=[(.05, 5.), (-8, 8)], method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(f"Risk calibration failed: {result.message}")
        parameters.append(result.x.tolist())
    return {"fitting_match": "132877", "parameters": parameters,
            "note": "Fitted on out-of-match development predictions; transferred frozen to final models, not refitted in-sample or retrospectively."}


def apply_calibration(predictions, calibration):
    if calibration["fitting_match"] != "132877":
        raise ValueError("Calibration match violation")
    p = predictions.copy()
    for i, (slope, intercept) in enumerate(calibration["parameters"]):
        x = np.clip(p[:, i], 1e-6, 1-1e-6)
        p[:, i] = 1/(1+np.exp(-np.clip(slope*np.log(x/(1-x))+intercept, -40, 40)))
    return p


def binary_metrics(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    if not len(y):
        return {"examples": 0, "status": "EMPTY_SUBSET"}
    predicted = p >= .5
    tp = int((predicted & y).sum())
    bins, ece = [], 0.
    for lower in np.arange(0, 1, .1):
        keep = (p >= lower) & ((p < lower+.1) if lower < .9 else (p <= 1))
        if keep.any():
            confidence, frequency = float(p[keep].mean()), float(y[keep].mean())
            ece += keep.mean()*abs(confidence-frequency)
            bins.append({"count": int(keep.sum()), "confidence": confidence, "frequency": frequency})
    return {
        "examples": len(y), "positive": int(y.sum()), "brier": float(((p-y)**2).mean()),
        "pr_auc": float(average_precision_score(y, p)) if y.any() else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auc_status": "DEFINED" if len(np.unique(y)) == 2 else "SINGLE_CLASS",
        "precision_at_0_5": tp/max(1, predicted.sum()), "recall_at_0_5": tp/max(1, y.sum()),
        "ece": float(ece), "calibration_bins": bins,
        "fixed_operating_points": {
            str(threshold): {
                "precision": float(((p >= threshold) & y).sum()/max(1, (p >= threshold).sum())),
                "recall": float(((p >= threshold) & y).sum()/max(1, y.sum())),
                "predicted_positive": int((p >= threshold).sum()),
            } for threshold in (.2, .5, .8)
        },
    }


def evaluate(data, indices, prediction):
    truth = targets(data, indices)
    def subset(keep):
        rows = indices[keep]
        if not len(rows):
            return {"examples": 0}
        p, y = prediction[keep], truth[keep]
        ranking = []
        for scene in np.unique(data["scene"][rows]):
            local = np.flatnonzero(data["scene"][rows] == scene)
            if len(local) < 2 or len(np.unique(y[local, 1])) < 2:
                continue
            good = np.flatnonzero(y[local, 1] == y[local, 1].min())
            chosen = np.flatnonzero(p[local, 1] == p[local, 1].min())
            ranking.append(float(np.isin(chosen, good).mean()))
        return {
            "examples": len(rows), "scenes": len(np.unique(data["scene"][rows])),
            "lane": {str(t): binary_metrics(p[:, i], y[:, i]) for i, t in enumerate((2, 3, 5))},
            "hidden_threat": {str(t): binary_metrics(p[:, i+3], y[:, i+3]) for i, t in enumerate((2, 3, 5))},
            "nearest_defender_mae_m": float(np.abs(p[:, 6]-y[:, 6]).mean()),
            "pressure_category_accuracy": float((np.digitize(p[:, 6], [3, 5, 10]) == np.digitize(y[:, 6], [3, 5, 10])).mean()),
            "candidate_ranking_success": float(np.mean(ranking)) if ranking else None,
            "ranking_eligible_scenes": len(ranking),
            "perturbed_target_expected_brier": float((p[:, 1]**2-2*p[:, 1]*data["target_perturbed_mean"][rows, 1]+data["target_perturbed_mean"][rows, 1]).mean()),
        }
    all_rows = np.ones(len(indices), bool)
    result = {"overall": subset(all_rows), "by_match": {}, "subsets": {}}
    for match in sorted(set(data["candidate_match"][indices])):
        result["by_match"][str(match)] = subset(data["candidate_match"][indices] == match)
    for name, keep in (
        ("unobstructed", data["target_source"][indices, 1] == 0),
        ("visible_only", data["target_source"][indices, 1] == 1),
        ("hidden_only", data["target_source"][indices, 1] == 2),
        ("both", data["target_source"][indices, 1] == 3),
        ("never_seen_threat", data["target_never_seen_threat"][indices]),
        ("boundary_within_1m", data["target_boundary"][indices, 1]),
        ("away_from_boundary", ~data["target_boundary"][indices, 1]),
        ("candidate_under15m", data["distance"][indices] < 15),
        ("candidate_15to30m", (data["distance"][indices] >= 15) & (data["distance"][indices] < 30)),
        ("candidate_30mplus", data["distance"][indices] >= 30),
        ("candidates_1to3", data["candidate_count"][indices] <= 3),
        ("candidates_4plus", data["candidate_count"][indices] > 3),
    ):
        result["subsets"][name] = subset(keep)
    for low, high in ((0, .5), (.5, 1), (1, 2.1)):
        age = data["target_responsible_age"][indices]
        result["subsets"][f"responsible_memory_{low}_{high}s"] = subset((age >= low) & (age < high))
    return result


def geometry_predictions(data, belief, truth_only, deadline, samples=128):
    outputs = {k: [] for k in ("visible", "last_seen", "constant_velocity", "learned_point", "learned_gaussian", "learned_occupancy", "oracle")}
    for row, scene in enumerate(data["scene"]):
        if row % 100 == 0:
            check_deadline(deadline)
        actor, candidate = data["actor_xy"][scene], data["candidate_xy"][row]
        visible, hidden = data["visible_defender"][scene], data["hidden_defender"][scene]
        current = data["last_xy"][scene]
        rng = np.random.default_rng(1000+int(scene))
        noise = rng.standard_normal((samples, 22, 2)).astype(np.float32)
        for name in outputs:
            if name == "oracle":
                points, valid, hidden_mask = truth_only["xy"][scene], truth_only["opponents"][scene], truth_only["opponents"][scene] & ~data["visible"][scene, -1]
                draws = np.broadcast_to(points, (samples, 22, 2))
            else:
                if name.startswith("learned") and not belief["covered"][scene]:
                    # Development cache intentionally has no retrospective forecasts; never use these rows.
                    outputs[name].append(np.full(7, np.nan, np.float32))
                    continue
                valid = visible if name == "visible" else visible | hidden
                hidden_mask = hidden
                points = current.copy()
                if name == "constant_velocity":
                    points[hidden] += data["velocity"][scene, hidden]*data["age"][scene, hidden, None]
                if name.startswith("learned"):
                    points[hidden] = belief["mean"][scene, hidden]
                draws = np.broadcast_to(points, (samples, 22, 2)).copy()
                if name in ("learned_gaussian", "learned_occupancy"):
                    draws[:, hidden] += noise[:, hidden]*belief["std"][scene, hidden]
            occ = None
            if name == "learned_occupancy":
                # Independent cells and independent seen-player Gaussians; resolution uncertainty retained.
                jitter = rng.uniform(-1, 1, (samples, 294, 2))*np.array([2.5, 68/28])
                positions = GRID[None]+jitter
                selected = rng.random((samples, 294)) < belief["anonymous"][scene]
                occ = (positions, selected)
            outputs[name].append(probability_from_samples(actor, candidate, draws, valid, hidden_mask, occ))
    return {key: np.stack(value) for key, value in outputs.items()}
