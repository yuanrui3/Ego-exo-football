"""Checkpointable world-model fits and matched reconstruction/forecast evaluation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
from torch.utils.data import DataLoader, Subset

from src.belief_observation import GRID, memory_belief
from src.models.belief_world_model import BeliefWorldModel, world_loss
from src.trajectory_dataset import check_deadline, write_json
from train.run_trajectory_forecasting import restore_rng, rng_state, save_torch, seed_all


def tensor_batch(batch, device="cuda"):
    return {key: value.to(device) for key, value in batch.items() if key != "source_index"}


def predict(model, batch):
    return model(batch["x"], batch["mask"], batch["actor_xy"], batch["last_xy"])


def loader(dataset, batch_size=32, order=None, start=0):
    if order is not None:
        dataset = Subset(dataset, order[start:])
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)


def model_from(config):
    return BeliefWorldModel(
        width=config["width"], layers=config["layers"], heads=config["heads"],
        social=config.get("social", True), dropout=config.get("dropout", .1),
    )


def fit_world(directory, config, training, development, deadline, epochs=None, progress=None):
    directory.mkdir(parents=True, exist_ok=True)
    specification = {
        "config": config, "epochs": epochs,
        "training_sources": [item[0] for item in training.sources],
        "development_sources": [item[0] for item in development.sources] if development is not None else [],
        "training_examples": len(training),
    }
    if development is not None and set(specification["training_sources"]) & set(specification["development_sources"]):
        raise ValueError("World fitting/development match overlap")
    if (directory / "config.json").exists() and json.loads((directory / "config.json").read_text()) != specification:
        raise ValueError(f"Resume specification mismatch: {directory}")
    write_json(directory / "config.json", specification)
    if (directory / "result.json").exists():
        if not (directory / "final.pt").exists():
            raise FileNotFoundError(directory / "final.pt")
        return json.loads((directory / "result.json").read_text())
    seed_all(config["seed"])
    model = model_from(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=.0001)
    state = {"epoch": 0, "offset": 0, "best": float(1e20), "best_epoch": 0,
             "stale": 0, "steps": 0, "history": [], "loss_sum": 0., "batches": 0}
    path = directory / "last.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        restore_rng(saved["rng"])
    started = time.monotonic()
    last_save = started

    def save():
        save_torch(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "state": state, "rng": rng_state()})
        write_json(directory / "live.json", {key: state[key] for key in ("epoch", "offset", "steps", "best", "best_epoch")})

    maximum = epochs if epochs is not None else config["max_epochs"]
    try:
        while state["epoch"] < maximum and (epochs is not None or state["stale"] < config["patience"]):
            order = np.random.default_rng(config["seed"] + state["epoch"]).permutation(len(training))
            model.train()
            for cpu in loader(training, config["batch_size"], order, state["offset"]):
                check_deadline(deadline)
                batch = tensor_batch(cpu)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = predict(model, batch)
                    loss = world_loss(output, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite world loss: {directory}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                state["steps"] += 1
                state["offset"] += len(cpu["x"])
                state["loss_sum"] += float(loss.detach())
                state["batches"] += 1
                if time.monotonic() - last_save >= 120:
                    save()
                    if progress:
                        progress({"fit": str(directory), "epoch": state["epoch"],
                                  "steps": state["steps"], "examples": len(training)})
                    last_save = time.monotonic()
            score = validation_loss(model, development, config["batch_size"], deadline) if development is not None else 0.
            state["epoch"] += 1
            state["offset"] = 0
            row = {"epoch": state["epoch"], "training_loss": state["loss_sum"] / max(1, state["batches"]),
                   "development_loss": score}
            state["history"].append(row)
            state["loss_sum"], state["batches"] = 0., 0
            if development is not None:
                if score < state["best"] - 1e-5:
                    state["best"], state["best_epoch"], state["stale"] = score, state["epoch"], 0
                    save_torch(directory / "best.pt", model.state_dict())
                else:
                    state["stale"] += 1
            save()
            write_json(directory / "history.json", state["history"])
            if progress:
                progress({"fit": str(directory), **row, "steps": state["steps"]})
        if development is not None:
            model.load_state_dict(torch.load(directory / "best.pt", weights_only=True))
        else:
            state["best_epoch"] = state["epoch"]
        save_torch(directory / "final.pt", model.state_dict())
        result = {
            "status": "COMPLETE", "best_epoch": state["best_epoch"], "development_loss": state["best"],
            "steps": state["steps"], "runtime_this_invocation_seconds": time.monotonic() - started,
            "parameters": sum(p.numel() for p in model.parameters()), "config": config,
        }
        write_json(directory / "result.json", result)
        return result
    except (Exception, KeyboardInterrupt):
        save()
        raise
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


@torch.no_grad()
def validation_loss(model, dataset, batch_size, deadline):
    model.eval()
    total, count = 0., 0
    for cpu in loader(dataset, batch_size):
        check_deadline(deadline)
        batch = tensor_batch(cpu)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = world_loss(predict(model, batch), batch)
        total += float(loss) * len(cpu["x"])
        count += len(cpu["x"])
    if not count:
        raise ValueError("Empty world-model evaluation set")
    return total / count


def occupancy_metrics(probability, target):
    p, y = np.asarray(probability, np.float64), np.asarray(target, np.float64)
    flat_p, flat_y = p.ravel(), y.ravel()
    positive = flat_y > .5
    ece = 0.
    for lower in np.linspace(0, .9, 10):
        selected = (flat_p >= lower) & (flat_p < lower + .1 if lower < .9 else flat_p <= 1)
        if selected.any():
            ece += float(selected.mean() * abs(flat_p[selected].mean() - flat_y[selected].mean()))
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    prediction = p >= .5
    tp = float((prediction & (y > .5)).sum())
    return {
        "brier": float(((p-y)**2).mean()),
        "bernoulli_nll": float(-(y*np.log(clipped) + (1-y)*np.log(1-clipped)).mean()),
        "pr_auc_average_precision": float(average_precision_score(flat_y, flat_p)) if positive.any() else None,
        "positive_cell_brier": float(((flat_p[positive]-1)**2).mean()) if positive.any() else None,
        "positive_cell_probability_mean": float(flat_p[positive].mean()) if positive.any() else None,
        "precision_at_0_5": tp / max(1., float(prediction.sum())),
        "recall_at_0_5": tp / max(1., float(positive.sum())),
        "occupied_cell_count_mae": float(np.abs(p.sum(-1)-y.sum(-1)).mean()),
        "ece": ece, "positive_cells": int(positive.sum()), "windows": len(p),
        "count_definition": "occupied 5m grid cells, not exact player count when players share a cell",
    }


def calibrate_probability(probability, parameters):
    p = np.clip(np.asarray(probability, np.float64), 1e-6, 1-1e-6)
    score = parameters["slope"] * np.log(p/(1-p)) + parameters["intercept"]
    return (1 / (1+np.exp(-np.clip(score, -40, 40)))).astype(np.float32)


def development_calibration(directory):
    """Fit probability scale/offset on development predictions only, with matched baseline treatment."""
    archive = directory / "occupancy_predictions.npz"
    with np.load(archive, allow_pickle=False) as data:
        y = data["target"].astype(float).ravel()
        result = {"fitting_match": "132877", "method": "bounded positive-slope logistic recalibration"}
        for name in ("learned", "last_seen", "constant_velocity"):
            p = np.clip(data[name].astype(float).ravel(), 1e-6, 1-1e-6)
            x = np.log(p/(1-p))
            def objective(parameters):
                logits = parameters[0]*x+parameters[1]
                return float((np.logaddexp(0, logits)-y*logits).mean())
            fit = minimize(objective, [1., 0.], bounds=[(.05, 5), (-8, 8)], method="L-BFGS-B")
            if not fit.success:
                raise RuntimeError(f"Development probability calibration failed: {fit.message}")
            result[name] = {"slope": float(fit.x[0]), "intercept": float(fit.x[1])}
    write_json(directory / "calibration.json", result)
    return result


@torch.no_grad()
def evaluate_world(model, dataset, config, deadline, directory):
    path = directory / "evaluation.json"
    if path.exists():
        return json.loads(path.read_text())
    model.eval()
    predictions = {"learned": [], "last_seen": [], "constant_velocity": [], "empty": []}
    targets, errors, ages, counts = [], {name: [] for name in ("learned", "last_seen", "constant_velocity")}, [], []
    count_errors = {"learned": [], "last_seen": [], "constant_velocity": []}
    forecast_sum = {name: np.zeros(4) for name in errors}
    forecast_count = np.zeros(4)
    future_nll_sum, future_nll_count = 0., 0
    forecast_coverage = np.zeros((2, 4))
    standardized_square_sum = np.zeros(4)
    positive_spatial = []
    trajectory_examples = []
    for cpu in loader(dataset, config["batch_size"]):
        check_deadline(deadline)
        batch = tensor_batch(cpu)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = predict(model, batch)
        pred = output["occupancy"].sigmoid().cpu().numpy()
        predictions["learned"].append(pred)
        targets.append(cpu["occupancy"].numpy())
        last, cv = [], []
        for i in range(len(cpu["x"])):
            common = (
                cpu["last_xy"][i].numpy(), cpu["velocity"][i].numpy(),
                cpu["age"][i].numpy(), cpu["mask"][i, -1].numpy(), cpu["visible"][i, -1].numpy(),
            )
            last.append(memory_belief(*common))
            cv.append(memory_belief(*common, constant_velocity=True))
        predictions["last_seen"].append(np.asarray(last))
        predictions["constant_velocity"].append(np.asarray(cv))
        predictions["empty"].append(np.zeros_like(pred))
        pmask = cpu["position_mask"].numpy().astype(bool)
        true_counts = cpu["hidden_count"].numpy()
        count_errors["learned"].extend(np.abs(output["hidden_count"].cpu().numpy()-true_counts).tolist())
        last_counts = (cpu["mask"][:, -1] & ~cpu["visible"][:, -1]).sum(-1).numpy()
        for name in ("last_seen", "constant_velocity"):
            count_errors[name].extend(np.abs(last_counts-true_counts).tolist())
        mean = output["position_mean"].cpu().numpy()
        position = cpu["position"].numpy()
        baselines = {
            "learned": mean, "last_seen": cpu["last_xy"].numpy(),
            "constant_velocity": cpu["last_xy"].numpy() + cpu["velocity"].numpy() * cpu["age"].numpy()[..., None],
        }
        for name, points in baselines.items():
            errors[name].append(np.linalg.norm(points-position, axis=-1)[pmask])
        ages.append(cpu["age"].numpy()[pmask])
        counts.extend(cpu["position_mask"].sum(-1).tolist())
        future = cpu["future"].numpy()
        fmask = cpu["future_mask"].numpy().astype(bool)
        predicted_future = output["future_mean"].cpu().numpy()
        if len(trajectory_examples) < 4:
            for row in range(len(cpu["x"])):
                candidates = np.flatnonzero(fmask[row].all(-1))
                if len(candidates):
                    player = int(candidates[0])
                    trajectory_examples.append({
                        "last_observed": cpu["last_xy"][row, player].tolist(),
                        "actual_future": future[row, player].tolist(),
                        "predicted_future": predicted_future[row, player].tolist(),
                    })
                if len(trajectory_examples) >= 4:
                    break
        forecast = {
            "learned": predicted_future,
            "last_seen": np.broadcast_to(cpu["last_xy"].numpy()[:, :, None], future.shape),
            "constant_velocity": cpu["last_xy"].numpy()[:, :, None] + cpu["velocity"].numpy()[:, :, None]
            * (cpu["age"].numpy()[..., None] + np.asarray([.4, 1, 2, 3]))[..., None],
        }
        for name, points in forecast.items():
            distance = np.linalg.norm(points-future, axis=-1)
            forecast_sum[name] += (distance*fmask).sum((0, 1))
        forecast_count += fmask.sum((0, 1))
        logs = output["future_log_std"].cpu().numpy()
        standardized = ((future-predicted_future)*np.exp(-logs))**2
        standardized_square_sum += (standardized.sum(-1)*fmask).sum((0, 1))
        for index, threshold in enumerate((2.278868566, 5.991464547)):
            forecast_coverage[index] += ((standardized.sum(-1) <= threshold)*fmask).sum((0, 1))
        nll = (.5*((future-predicted_future)*np.exp(-logs))**2+logs+.5*math.log(2*math.pi)).sum(-1)
        future_nll_sum += float((nll*fmask).sum())
        future_nll_count += int(fmask.sum())
        for pp, yy in zip(pred, cpu["occupancy"].numpy()):
            true_indices = np.flatnonzero(yy > .5)
            k = len(true_indices)
            if k:
                top = np.argsort(-pp)[:k]
                distance = np.linalg.norm(GRID[true_indices, None] - GRID[top][None], axis=-1).min(-1)
                positive_spatial.extend(distance.tolist())
    y = np.concatenate(targets)
    p = {name: np.concatenate(values) for name, values in predictions.items()}
    all_ages = np.concatenate(ages)
    metric = {
        "occupancy": {name: occupancy_metrics(values, y) for name, values in p.items()},
        "hidden_previously_seen_localization_m": {
            name: float(np.concatenate(values).mean()) if sum(map(len, values)) else None
            for name, values in errors.items()
        },
        "localization_by_memory_age": {},
        "forecast": {
            name: {"mde_m": dict(zip(("0.4", "1", "2", "3"), (value/np.maximum(forecast_count, 1)).tolist())),
                   "ade_m": float((value/np.maximum(forecast_count, 1)).mean()),
                   "valid_targets_per_horizon": forecast_count.astype(int).tolist()}
            for name, value in forecast_sum.items()
        },
        "forecast_gaussian_nll": future_nll_sum / max(1, future_nll_count),
        "forecast_uncertainty": {
            "nominal_68_percent_ellipse_coverage": (forecast_coverage[0]/np.maximum(forecast_count, 1)).tolist(),
            "nominal_95_percent_ellipse_coverage": (forecast_coverage[1]/np.maximum(forecast_count, 1)).tolist(),
            "mean_standardized_squared_radius": (standardized_square_sum/np.maximum(forecast_count, 1)).tolist(),
            "ideal_standardized_squared_radius": 2,
        },
        "hidden_player_count_mae": {name: float(np.mean(value)) for name, value in count_errors.items()},
        "positive_cell_nearest_top_k_localization_m": float(np.mean(positive_spatial)) if positive_spatial else None,
        "evaluation_sampling": "one actor view per nonoverlapping nine-second history+future interval",
        "trajectory_examples": trajectory_examples,
    }
    calibration = config.get("calibration")
    if calibration:
        if calibration["fitting_match"] != "132877":
            raise ValueError("Calibration must be fitted on development only")
        metric["development_calibrated_occupancy"] = {
            name: occupancy_metrics(calibrate_probability(p[name], calibration[name]), y)
            for name in ("learned", "last_seen", "constant_velocity")
        }
        metric["calibration_parameters"] = calibration
    for lo, hi in ((0, .5), (.5, 1), (1, 2), (2, 4), (4, 8)):
        selected = (all_ages >= lo) & (all_ages < hi)
        metric["localization_by_memory_age"][f"{lo}-{hi}s"] = {
            "count": int(selected.sum()),
            **{name: float(np.concatenate(value)[selected].mean()) if selected.any() else None
               for name, value in errors.items()},
        }
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "occupancy_predictions.npz", target=y.astype(np.float16),
                        **{key: value.astype(np.float16) for key, value in p.items()})
    write_json(path, metric)
    return metric
