"""Original-prevalence metrics and explicit synthetic-observation evidence levels."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.belief_observation import GRID, memory_belief
from src.hidden_defender_geometry import probability_from_samples
from train.belief_world_training import occupancy_metrics
from train.hidden_defender_training import binary_metrics, evaluate


def binary_with_operating_points(probability, target, development_probability=None, development_target=None):
    p, y = np.asarray(probability, float), np.asarray(target, bool)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid binary probabilities")
    result = binary_metrics(p, y)
    result.update(negative=int((~y).sum()), prevalence=float(y.mean()) if len(y) else None)
    result["recall_at_test_roc_fpr"] = {}
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        result["recall_at_test_roc_fpr"] = {
            str(limit): float(tpr[fpr <= limit].max()) for limit in (.01, .05, .1)}
    if development_probability is not None:
        dp, dy = np.asarray(development_probability), np.asarray(development_target, bool)
        negatives = dp[~dy]
        result["development_selected_fpr_thresholds"] = {}
        for limit in (.01, .05, .1):
            if not len(negatives):
                result["development_selected_fpr_thresholds"][str(limit)] = {"status": "NO_DEV_NEGATIVES"}
                continue
            # Strictly above the boundary; ties cannot inflate the selected dev FPR.
            rank = min(len(negatives) - 1, int(np.floor((1 - limit) * len(negatives))))
            threshold = float(np.nextafter(np.sort(negatives)[rank], np.inf))
            predicted = p >= threshold
            tp, fp = int((predicted & y).sum()), int((predicted & ~y).sum())
            result["development_selected_fpr_thresholds"][str(limit)] = {
                "threshold": threshold, "development_fpr": float((negatives >= threshold).mean()),
                "test_fpr": fp / max(1, (~y).sum()),
                "precision": tp / max(1, predicted.sum()), "recall": tp / max(1, y.sum()),
                "predicted_positive": int(predicted.sum()),
            }
    return result


def risk_metrics(data, truth, rows, prediction, development_rows=None, development_prediction=None):
    rows = np.asarray(rows)
    result = evaluate(data, rows, prediction)
    scene = data["scene"][rows]
    # Inclusion depends ONLY on hidden-defender presence, never on obstruction.
    hidden_defender = truth["opponents"] & ~data["visible"][:, -1]
    eligible = hidden_defender[scene].any(-1)
    corrected = {}
    for column, threshold in enumerate((2, 3, 5)):
        dp, dy = None, None
        if development_rows is not None:
            dev_eligible = hidden_defender[data["scene"][development_rows]].any(-1)
            dp = development_prediction[dev_eligible, column + 3]
            dy = data["target_hidden"][development_rows[dev_eligible], column]
        corrected[str(threshold)] = binary_with_operating_points(
            prediction[eligible, column + 3], data["target_hidden"][rows[eligible], column], dp, dy)
    result["hidden_threat_original_prevalence"] = corrected
    result["hidden_threat_population"] = "All original lanes with >=1 currently hidden tracked defender, including negatives"
    return result


def collect_world_targets(dataset):
    keys = ("position", "position_mask", "age", "visible", "last_xy", "velocity", "mask",
            "occupancy", "anonymous_defender", "target_association")
    values = {key: [] for key in keys}
    counts = {"visible_teammates": [], "visible_opponents": []}
    for batch in DataLoader(dataset, batch_size=256, num_workers=0):
        for key in keys:
            values[key].append(batch[key].numpy())
        current = batch["x"][:, -1].numpy()
        visible = batch["visible"][:, -1].numpy()
        counts["visible_teammates"].append((visible & (current[..., 7] > .5) & (current[..., 9] < .5)).sum(-1))
        counts["visible_opponents"].append((visible & (current[..., 7] < -.5)).sum(-1))
    return {key: np.concatenate(value) for key, value in {**values, **counts}.items()}


def world_metrics(target, prediction):
    selected = target["position_mask"].astype(bool)
    truth = target["position"]
    mean, std = prediction["mean"], prediction["std"]
    if np.any(std <= 0) or not all(np.isfinite(value).all() for value in prediction.values()):
        raise ValueError("Invalid world predictions")
    errors = np.linalg.norm(mean - truth, axis=-1)
    square_radius = (((mean - truth) / std) ** 2).sum(-1)
    nll = (.5 * ((mean - truth) / std) ** 2 + np.log(std) + .5 * math.log(2 * math.pi)).sum(-1)
    def localization(keep):
        if not keep.any():
            return {"count": 0, "status": "NO_HIDDEN_PREVIOUSLY_SEEN_TARGETS"}
        return {"count": int(keep.sum()), "localization_m": float(errors[keep].mean()),
                "position_nll": float(nll[keep].mean()),
                "coverage_68": float((square_radius[keep] <= 2.278868566).mean()),
                "coverage_95": float((square_radius[keep] <= 5.991464547).mean()),
                "mean_std_m": float(std[keep].mean())}
    result = {**localization(selected), "occupancy": occupancy_metrics(prediction["occupancy"], target["occupancy"]),
              "anonymous_never_seen_defender_occupancy": occupancy_metrics(
                  prediction["anonymous"], target["anonymous_defender"]),
              "by_age": {}, "by_visible_players": {}}
    for low, high in ((0, .5), (.5, 1), (1, 2.01), (2.01, 4), (4, 6.01)):
        result["by_age"][f"{low}-{high}"] = localization(selected & (target["age"] >= low) & (target["age"] < high))
    visible_count = target["visible"][:, -1].sum(-1)
    for low, high in ((0, 5), (5, 10), (10, 15), (15, 23)):
        result["by_visible_players"][f"{low}-{high}"] = localization(
            selected & ((visible_count >= low) & (visible_count < high))[:, None])
    for key in ("visible_teammates", "visible_opponents"):
        result[f"by_{key}"] = {}
        for low, high in ((0, 3), (3, 6), (6, 9), (9, 12)):
            result[f"by_{key}"][f"{low}-{high}"] = localization(
                selected & ((target[key] >= low) & (target[key] < high))[:, None])
    return result


def world_baselines(target):
    output = {}
    for name in ("last_seen", "constant_velocity"):
        mean = target["last_xy"].copy()
        if name == "constant_velocity":
            mean += target["velocity"] * target["age"][..., None]
        std = np.broadcast_to((1.5 + 1.1 * target["age"])[..., None], mean.shape).copy()
        oc = np.stack([memory_belief(target["last_xy"][row], target["velocity"][row],
                                     target["age"][row], target["mask"][row, -1],
                                     target["visible"][row, -1], name == "constant_velocity")
                       for row in range(len(mean))])
        output[name] = {"mean": mean, "std": std, "occupancy": oc,
                        "anonymous": np.zeros_like(oc)}
    return output


def receiver_metrics(prediction, target, matches):
    rank = np.argsort(-prediction, axis=-1, kind="stable")
    position = np.argmax(rank == np.asarray(target)[:, None], axis=-1) + 1
    def score(keep):
        return {"examples": int(keep.sum()), "top1": float((position[keep] == 1).mean()),
                "top3": float((position[keep] <= 3).mean()), "mrr": float((1 / position[keep]).mean())}
    return {"overall": score(np.ones(len(target), bool)),
            "by_match": {str(match): score(matches == match) for match in np.unique(matches)}}


def direct_geometry(data, belief, rows, deadline, kind="gaussian", truth=None, samples=128):
    from src.trajectory_dataset import check_deadline
    output = []
    for index, row in enumerate(rows):
        if index % 100 == 0:
            check_deadline(deadline)
        scene = data["scene"][row]
        actor, candidate = data["actor_xy"][scene], data["candidate_xy"][row]
        visible = data["visible_defender"][scene]
        hidden = data["hidden_defender"][scene]
        points = data["last_xy"][scene].copy()
        valid = visible | hidden
        if kind == "oracle":
            if truth is None:
                raise ValueError("Oracle requires explicitly separated evaluation labels")
            points, valid = truth["xy"][scene], truth["opponents"][scene]
            hidden = valid & ~data["visible"][scene, -1]
        elif kind == "constant_velocity":
            points[hidden] += data["velocity"][scene, hidden] * data["age"][scene, hidden, None]
        elif kind in ("gaussian", "point", "anonymous"):
            if not belief["covered"][scene]:
                raise ValueError("Direct geometry missing held-out belief")
            points[hidden] = belief["mean"][scene, hidden]
        elif kind == "visible":
            valid = visible
        draws = np.broadcast_to(points, (samples, 22, 2)).copy()
        rng = np.random.default_rng(1000 + int(scene))
        if kind in ("gaussian", "anonymous"):
            noise = rng.standard_normal(draws.shape).astype(np.float32)
            draws[:, hidden] += noise[:, hidden] * belief["std"][scene, hidden]
        occ = None
        if kind == "anonymous":
            positions = GRID[None] + rng.uniform(-1, 1, (samples, 294, 2)) * [2.5, 68 / 28]
            selected = rng.random((samples, 294)) < belief["anonymous"][scene]
            occ = positions, selected
        output.append(probability_from_samples(actor, candidate, draws, valid, hidden, occ))
    return np.stack(output)


def paired_match_effects(values, baseline):
    """Descriptive six-match uncertainty, not a seed-as-match significance claim."""
    common = sorted(set(values) & set(baseline))
    delta = np.asarray([values[key] - baseline[key] for key in common])
    if not len(delta):
        return {"matches": 0, "status": "NO_PAIRED_MATCHES"}
    rng = np.random.default_rng(405)
    means = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(-1)
    return {"matches": len(delta), "per_match_difference": dict(zip(common, delta.tolist())),
            "mean_difference": float(delta.mean()), "median_difference": float(np.median(delta)),
            "bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist(),
            "warning": "Exploratory interval across a small set of matches; seeds are not independent matches"}
