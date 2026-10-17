"""One bounded, resumable local GPU queue for raw-player trajectory forecasting."""

import argparse
import copy
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.models.trajectory_forecasting import (
    HORIZON_SECONDS, TrajectoryForecaster, constant_velocity, masked_loss, persistence,
)
from src.trajectory_dataset import (
    DEVELOPMENT_MATCH, HORIZONS, TRAIN_MATCHES, VALIDATION_MATCHES,
    BudgetExpired, TrajectoryDataset, build_half, check_deadline, feasibility,
    file_digest, source_signature, validate_splits, write_json,
)

LOG = logging.getLogger("trajectory")


def code_hashes():
    paths = (Path(__file__), Path("src/trajectory_dataset.py"),
             Path("src/models/trajectory_forecasting.py"))
    return {str(p.resolve()): file_digest(p) for p in paths}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_torch(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp.pt")
    torch.save(value, tmp)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def loader(dataset, batch_size, seed=42, order=None, start_batch=0):
    if order is not None:
        dataset = Subset(dataset, order[start_batch * batch_size:])
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def metric_sums(prediction, batch, selected=None):
    mask = batch["target_mask"]
    if selected is not None:
        mask = mask & selected[:, :, None]
    error = torch.linalg.vector_norm(prediction.float() - batch["target"].float(), dim=-1)
    return (
        (error * mask).sum((0, 1)).double().cpu().numpy(),
        mask.sum((0, 1)).double().cpu().numpy(),
    )


def finish_metric(sums, counts):
    means = [float(s / n) if n else None for s, n in zip(sums, counts)]
    valid = [v for v in means if v is not None]
    return {
        "mde_m": dict(zip(("0.4", "1.0", "2.0", "3.0"), means)),
        "ade_m": float(np.mean(valid)) if valid else None,
        "fde_3s_m": means[-1], "valid_targets": counts.astype(int).tolist(),
        "definition": "ADE = unweighted mean of four horizon MDEs; FDE = 3 s MDE",
    }


def pass_frames(data_dir, dataset):
    result = {}
    for index, (match, half) in enumerate(dataset.match_names):
        events = load_json(Path(data_dir) / f"{match}_12_class_events.json")
        if events.get("clock") != "ms from the first frame of the half's released video":
            raise ValueError(f"Unknown BAS timestamp reference for PASS subset: {match}")
        if float(events.get("fps", 0)) != 25:
            raise ValueError(f"Unexpected BAS frame rate: {match}")
        frames = []
        for event in events.get("annotations") or events.get("actions") or []:
            period = str(event.get("gameTime", "")).split(" - ")[0]
            expected_half = "1" if half == "1st" else "2"
            if period == expected_half and str(event.get("label", "")).lower() == "pass":
                position = event.get("position")
                if position is not None:
                    # Released-video first-frame time, not the displayed match clock.
                    frames.append(int(round(float(position) / 40)))
        result[index] = np.asarray(frames, dtype=np.int64)
    return result


@torch.no_grad()
def evaluate(model, dataset, config, device, deadline, directory=None, bas=None,
             calibration_temperature=1.0):
    """Resumable deterministic evaluation, retaining per-window errors for paired analysis."""
    result_path = directory / "metrics.json" if directory else None
    if result_path and result_path.exists():
        return load_json(result_path)
    partial = directory / "evaluation_partial.pt" if directory else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    sums, counts, rows, samples, next_batch = {}, {}, [], [], 0
    calibration = np.zeros((4, 5), dtype=np.float64)
    player_count_sum, windows, possible = 0, 0, np.zeros(4, np.int64)
    if partial and partial.exists():
        saved = torch.load(partial, map_location="cpu", weights_only=False)
        sums, counts, rows, samples = (saved[k] for k in ("sums", "counts", "rows", "samples"))
        next_batch, calibration = saved["next_batch"], saved["calibration"]
        player_count_sum, windows, possible = (saved[k] for k in ("player_count_sum", "windows", "possible"))
    if not isinstance(model, str):
        model.eval()
    order = np.arange(len(dataset))
    batches = loader(dataset, config["batch_size"], order=order, start_batch=next_batch)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def save_partial(batch_index):
        if partial:
            save_torch(partial, {
                "sums": sums, "counts": counts, "rows": rows, "samples": samples,
                "next_batch": batch_index, "calibration": calibration,
                "player_count_sum": player_count_sum, "windows": windows, "possible": possible,
            })

    for batch_index, cpu in enumerate(batches, start=next_batch):
        if time.monotonic() >= deadline:
            save_partial(batch_index)
            raise BudgetExpired("Budget reached during evaluation; exact batch position saved")
        batch = move(cpu, device)
        if isinstance(model, str):
            prediction = persistence(batch) if model == "persistence" else constant_velocity(batch)
            log_scale = None
        else:
            with torch.autocast("cuda", dtype=dtype):
                prediction, log_scale = model(batch)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite forecast during evaluation")
        matches = [dataset.match_names[int(i)][0] for i in cpu["source_index"]]
        grouping = {
            "overall": batch["player_mask"],
            "goalkeeper": batch["player_mask"] & (batch["role"] == 1),
            "outfield": batch["player_mask"] & (batch["role"] == 0),
        }
        for match in sorted(set(matches)):
            flag = torch.tensor([m == match for m in matches], device=device)
            grouping[f"match_{match}"] = batch["player_mask"] & flag[:, None]
        if bas is not None:
            near = []
            for source, frame in zip(cpu["source_index"], cpu["anchor_frame"]):
                events = bas[int(source)]
                near.append(bool(len(events) and np.min(np.abs(events - int(frame))) <= 25))
            grouping["around_PASS"] = batch["player_mask"] & torch.tensor(near, device=device)[:, None]
        for group, selection in grouping.items():
            s, n = metric_sums(prediction, batch, selection)
            sums[group] = sums.get(group, np.zeros(4)) + s
            counts[group] = counts.get(group, np.zeros(4)) + n
        error = torch.linalg.vector_norm(prediction.float() - batch["target"], dim=-1)
        mask = batch["target_mask"]
        error_sums = (error * mask).sum(1).cpu().numpy()
        target_counts = mask.sum(1).cpu().numpy()
        for j, (source, frame) in enumerate(zip(cpu["source_index"], cpu["anchor_frame"])):
            rows.append([int(source), int(frame), *error_sums[j].tolist(), *target_counts[j].tolist()])
        if log_scale is not None:
            sigma = log_scale.float().exp() * calibration_temperature
            radius_squared = ((prediction.float() - batch["target"]) ** 2).sum(-1) / sigma.square()
            calibration[:, 0] += (radius_squared * mask).sum((0, 1)).cpu().numpy()
            calibration[:, 1] += mask.sum((0, 1)).cpu().numpy()
            for column, probability in enumerate((.5, .8, .95), start=2):
                threshold = -2 * math.log(1 - probability)
                calibration[:, column] += ((radius_squared <= threshold) & mask).sum((0, 1)).cpu().numpy()
        for j in range(len(cpu["current"])):
            absolute_index = batch_index * config["batch_size"] + j
            if absolute_index in set(np.linspace(0, len(dataset) - 1, 6, dtype=int)):
                samples.append({
                    "match": matches[j], "half": dataset.match_names[int(cpu["source_index"][j])][1],
                    "frame": int(cpu["anchor_frame"][j]), "current": cpu["current"][j].numpy(),
                    "target": cpu["target"][j].numpy(), "target_mask": cpu["target_mask"][j].numpy(),
                    "prediction": prediction[j].float().cpu().numpy(),
                    "player_mask": cpu["player_mask"][j].numpy(),
                })
        nplayers = int(cpu["player_mask"].sum())
        player_count_sum += nplayers
        windows += len(cpu["current"])
        possible += nplayers
        if partial and (batch_index + 1) % 100 == 0:
            save_partial(batch_index + 1)
    metrics = {group: finish_metric(sums[group], counts[group]) for group in sums}
    metrics["coverage"] = {
        "windows": windows, "mean_observed_players": player_count_sum / max(1, windows),
        "possible_targets_by_horizon": possible.tolist(),
        "valid_fraction_by_horizon": (counts["overall"] / np.maximum(possible, 1)).tolist(),
    }
    if calibration[:, 1].sum():
        metrics["uncertainty"] = {
            "temperature": calibration_temperature,
            "mean_squared_standardized_radius": (calibration[:, 0] / np.maximum(calibration[:, 1], 1)).tolist(),
            "radial_coverage_50_80_95": (
                calibration[:, 2:] / np.maximum(calibration[:, 1:2], 1)
            ).tolist(),
        }
    if directory:
        np.savez_compressed(directory / "window_errors.npz", rows=np.asarray(rows, np.float64))
        save_torch(directory / "plot_samples.pt", samples)
        write_json(result_path, metrics)
        if partial and partial.exists():
            partial.unlink()
    return metrics


def model_kwargs(config):
    return {key: config[key] for key in (
        "width", "layers", "social", "history_frames", "probabilistic"
    )}


def fit(dataset, development, config, directory, device, deadline, max_steps=None):
    """Resume optimizer, AMP and RNG at an exact batch; no successful fit is rerun."""
    directory.mkdir(parents=True, exist_ok=True)
    config_path, result_path = directory / "config.json", directory / "fit_result.json"
    if config_path.exists() and load_json(config_path) != config:
        raise ValueError(f"Resume configuration mismatch: {directory}")
    write_json(config_path, config)
    if result_path.exists():
        return load_json(result_path)
    seed_all(config["seed"])
    model = TrajectoryForecaster(**model_kwargs(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    state = {
        "epoch": 0, "next_batch": 0, "best": float("inf"), "best_epoch": 0,
        "stale": 0, "history": [], "loss_sum": 0.0, "loss_batches": 0, "steps": 0,
    }
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        # torch CPU RNG tensors must remain on CPU even when model/optimizer are on CUDA.
        rng = saved["rng"]
        rng["torch"] = rng["torch"].cpu()
        rng["cuda"] = [v.cpu() for v in rng["cuda"]]
        restore_rng(rng)
        LOG.info("RESUME %s epoch=%s batch=%s", directory.name, state["epoch"], state["next_batch"])

    def checkpoint_now():
        save_torch(checkpoint, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "state": state, "rng": rng_state(),
        })
        write_json(directory / "live.json", {
            "epoch": state["epoch"], "next_batch": state["next_batch"], "steps": state["steps"],
            "best_development_ade_m": state["best"] if math.isfinite(state["best"]) else None,
            "best_epoch": state["best_epoch"], "updated_unix": time.time(),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        })

    last_save = time.monotonic()
    while state["epoch"] < config["epochs"]:
        epoch = state["epoch"]
        order = np.random.default_rng(config["seed"] + epoch).permutation(len(dataset))
        batches = loader(dataset, config["batch_size"], config["seed"] + epoch,
                         order, state["next_batch"])
        model.train()
        for batch_index, cpu in enumerate(batches, start=state["next_batch"]):
            if time.monotonic() >= deadline:
                checkpoint_now()
                raise BudgetExpired("Training budget reached; batch/optimizer/RNG saved")
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                prediction, scale = model(batch)
                loss = masked_loss(prediction, batch["target"], batch["target_mask"], scale)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {directory.name}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            state["loss_sum"] += float(loss.detach())
            state["loss_batches"] += 1
            state["steps"] += 1
            state["next_batch"] = batch_index + 1
            if time.monotonic() - last_save >= 60 or state["steps"] % 100 == 0:
                checkpoint_now()
                last_save = time.monotonic()
            if max_steps is not None and state["steps"] >= max_steps:
                checkpoint_now()
                return {"interrupted_for_smoke": True, "steps": state["steps"]}
        row = {"epoch": epoch + 1, "train_loss": state["loss_sum"] / max(1, state["loss_batches"])}
        if development is not None:
            # Save before dev evaluation so an interruption does not repeat training.
            checkpoint_now()
            dev_metrics = evaluate(model, development, config, device, deadline)
            value = dev_metrics["overall"]["ade_m"]
            row["development_ade_m"] = value
            row["development_fde_3s_m"] = dev_metrics["overall"]["fde_3s_m"]
            if value < state["best"] - config["min_delta_m"]:
                state["best"], state["best_epoch"], state["stale"] = value, epoch + 1, 0
                save_torch(directory / "best.pt", model.state_dict())
                write_json(directory / "best_development_metrics.json", dev_metrics)
            else:
                state["stale"] += 1
        state["history"].append(row)
        state["epoch"], state["next_batch"] = epoch + 1, 0
        state["loss_sum"], state["loss_batches"] = 0.0, 0
        checkpoint_now()
        write_json(directory / "history.json", state["history"])
        LOG.info("%s %s", directory.name, json.dumps(row))
        if development is not None and state["stale"] >= config["patience"]:
            break
    save_torch(directory / "final.pt", model.state_dict())
    result = {
        "completed_epochs": state["epoch"], "best_epoch": state["best_epoch"],
        "best_development_ade_m": state["best"] if development is not None else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": state["steps"], "status": "COMPLETE",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(result_path, result)
    return result


def load_model(directory, config, device, best=False):
    model = TrajectoryForecaster(**model_kwargs(config)).to(device)
    model.load_state_dict(torch.load(
        directory / ("best.pt" if best else "final.pt"), map_location=device, weights_only=True
    ))
    return model


def plot_comparisons(root, completed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts = root / "charts"
    charts.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = {}
    for name, item in completed.items():
        if "metrics" not in item:
            continue
        group = name.rsplit("_seed", 1)[0]
        groups.setdefault(group, []).append(list(item["metrics"]["overall"]["mde_m"].values()))
    for group, values in groups.items():
        arr = np.asarray(values)
        ax.errorbar(HORIZON_SECONDS, arr.mean(0), yerr=arr.std(0), marker="o", label=group)
    ax.set(xlabel="Forecast horizon (seconds)", ylabel="Mean displacement error (m)",
           title="Held-out movement forecasting (seed mean +/- population SD)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(charts / "horizon_comparison.png", dpi=150)
    plt.close(fig)
    sample_path = root / "evaluation" / "social_seed42" / "plot_samples.pt"
    if sample_path.exists():
        for index, sample in enumerate(torch.load(sample_path, weights_only=False)):
            fig, ax = plt.subplots(figsize=(10, 6))
            for player in np.flatnonzero(sample["player_mask"])[:8]:
                current = sample["current"][player]
                valid = sample["target_mask"][player]
                actual = np.vstack([current, current + sample["target"][player, valid]])
                predicted = np.vstack([current, current + sample["prediction"][player, valid]])
                line, = ax.plot(actual[:, 0], actual[:, 1], "-o", markersize=3)
                ax.plot(predicted[:, 0], predicted[:, 1], "--x", color=line.get_color(), markersize=4)
            ax.set(xlim=(-60, 60), ylim=(-40, 40), aspect="equal",
                   xlabel="Pitch x (m)", ylabel="Pitch y (m)",
                   title=f"{sample['match']} {sample['half']} frame {sample['frame']}: "
                         "actual solid; forecast dashed (8 storage-matched players)")
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(charts / f"trajectory_{index + 1}.png", dpi=150)
            plt.close(fig)


def paired_bootstrap(root, left, right):
    """Within-match 30 s temporal-block uncertainty, not a match-population claim."""
    a_path = root / "evaluation" / left / "window_errors.npz"
    b_path = root / "evaluation" / right / "window_errors.npz"
    if not a_path.exists() or not b_path.exists():
        return None
    with np.load(a_path) as data:
        a = data["rows"]
    with np.load(b_path) as data:
        b = data["rows"]
    if not np.array_equal(a[:, :2], b[:, :2]) or not np.array_equal(a[:, 6:], b[:, 6:]):
        raise ValueError("Paired comparison requires identical windows/target masks")
    blocks = {}
    for row_a, row_b in zip(a, b):
        key = (int(row_a[0]), int(row_a[1] // 750))
        blocks.setdefault(key, np.zeros(8))
        blocks[key][:4] += row_a[2:6] - row_b[2:6]
        blocks[key][4:] += row_a[6:]
    grouped = {}
    for (source, _), values in blocks.items():
        grouped.setdefault(source, []).append(values)
    rng = np.random.default_rng(42)
    replicates = []
    for _ in range(2000):
        total = np.zeros(8)
        for values in grouped.values():
            values = np.asarray(values)
            total += values[rng.integers(0, len(values), len(values))].sum(0)
        replicates.append(total[:4] / np.maximum(1, total[4:]))
    total = sum(blocks.values())
    return {
        "difference_m_left_minus_right": (total[:4] / np.maximum(total[4:], 1)).tolist(),
        "within_match_30s_block_95pct_ci_m": np.percentile(replicates, [2.5, 97.5], axis=0).T.tolist(),
        "temporal_blocks": len(blocks),
        "caution": "Overlapping windows block-resampled within four halves; only two held-out matches. "
                   "Not evidence of significance across the population of matches.",
    }


def publish(root, progress, status):
    completed = progress.get("completed", {})
    aggregate, all_groups = {}, {}
    for name, item in completed.items():
        if "metrics" not in item:
            continue
        family = name.rsplit("_seed", 1)[0]
        aggregate.setdefault(family, []).append(item["metrics"]["overall"])
        for group, value in item["metrics"].items():
            if "mde_m" in value:
                all_groups.setdefault(family, {}).setdefault(group, []).append(value)
    summary = {}
    for family, metrics in aggregate.items():
        values = np.asarray([list(row["mde_m"].values()) for row in metrics])
        ade = np.asarray([row["ade_m"] for row in metrics])
        summary[family] = {
            "runs": len(metrics), "mde_mean_m": values.mean(0).tolist(),
            "mde_seed_sd_m": values.std(0).tolist(), "ade_mean_m": float(ade.mean()),
            "ade_seed_sd_m": float(ade.std()), "fde_3s_mean_m": float(values[:, -1].mean()),
            "seed_sd_definition": "population SD across available seeds; one run has SD 0, not certainty",
        }
    group_summary = {}
    for family, groups in all_groups.items():
        group_summary[family] = {}
        for group, metrics in groups.items():
            values = np.asarray([
                [np.nan if x is None else x for x in row["mde_m"].values()] for row in metrics
            ])
            valid = np.isfinite(values).all()
            group_summary[family][group] = {
                "runs": len(metrics),
                "mde_mean_m": values.mean(0).tolist() if valid else [None] * 4,
                "mde_seed_sd_m": values.std(0).tolist() if valid else [None] * 4,
                "ade_mean_m": float(values.mean()) if valid else None,
                "ade_seed_sd_m": float(values.mean(1).std()) if valid else None,
                "valid_targets_per_run": [row["valid_targets"] for row in metrics],
            }
    paired = {}
    for seed in (42, 7, 123):
        for comparator in ("constant_velocity", f"independent_seed{seed}"):
            value = paired_bootstrap(root, f"social_seed{seed}", comparator)
            if value:
                paired[f"social_seed{seed}_minus_{comparator}"] = value
    document = {
        "status": status, "progress": progress, "aggregate_heldout": summary,
        "aggregate_by_match_role_and_PASS": group_summary,
        "paired_comparisons": paired, "horizons_seconds": HORIZON_SECONDS,
        "heldout_policy": "Once per final configuration/seed with resumable batch evaluation; no tuning on held-out.",
        "metrics_units": "meters", "report_is_provisional": status != "COMPLETE",
    }
    write_json(root / "summary.json", document)
    results = Path("results")
    results.mkdir(exist_ok=True)
    write_json(results / "trajectory_forecasting_metrics.json", document)
    lines = [
        "# Multi-agent football trajectory forecasting", "",
        f"Status: **{status}**. Updated {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        f"Experiment directory: `{root.resolve()}`.",
        f"Elapsed wall time: {(time.time() - progress['started_unix']) / 3600:.2f} hours.",
        "", "## Protocol", "",
        "Raw local GSR pitch coordinates in meters; 51 samples over 2 seconds at 25 fps. "
        "Exact targets at +10/+25/+50/+75 frames. Stride 13 frames (0.52 s).",
        "Selection uses six training matches with 132877 as development. Final fits use all seven "
        "training matches; held-out 118577 and 128058 are never used for selection.",
        "No ball observations, action labels, identity numbers, track IDs, jerseys or future positions "
        "are model features. BAS Pass is used only for a +/-1 s evaluation subset.",
        "Unknown identities are excluded. Duplicate observations are invalidated; missing frames, "
        "track/team/role changes and >1.4 m adjacent jumps split continuity. No interpolation.",
        "Only anchor scenes with 18--22 observed players, <=11 per team and >=12 valid 3 s targets "
        "are evaluated. This conditions results on continuity and excludes harder missing-track scenes. "
        "All models use identical windows and target masks.",
        "CV uses least-squares historical velocity over 0.4 s (>=6 real samples), capped at 12 m/s. "
        "Neural models predict metric displacement corrections to this same CV; static ablation has "
        "zero velocity and no history. Independent model has no player interaction. Social attention "
        "has learned relative geometry/team bias, padding masks and no slot embeddings.",
        "The independent-player baseline is also the no-social-attention ablation: it matches the "
        "selected social model's width, block count, learning rate and seeds, with development-selected "
        "duration. Attention blocks become per-player residual MLPs; parameter counts are recorded "
        "and are not identical.",
        "Coordinate grid/repetition statistics and source size/mtime/cache SHA256 are in cache metadata. "
        "Local GSR release commit is unknown (info.version is not a release SHA). Coarse staircase "
        "tracking is not independently verified continuous ground truth.",
        "The feasibility pilot has a 1.05 m x-axis / 0.68 m y-axis coordinate lattice and 85.6% "
        "identical adjacent positions; instantaneous apparent speeds are therefore misleading. "
        "Short-horizon error is limited by quantization. These tracks are usable for coarse movement "
        "experiments, not precise ball contact or interception timing.",
        "", "## Held-out results", "",
        "| Model | Seeds/runs | MDE 0.4 s | MDE 1 s | MDE 2 s | MDE 3 s | ADE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary.items():
        cells = [f"{m:.3f} +/- {s:.3f}" for m, s in zip(value["mde_mean_m"], value["mde_seed_sd_m"])]
        lines.append(f"| {name} | {value['runs']} | " + " | ".join(cells)
                     + f" | {value['ade_mean_m']:.3f} +/- {value['ade_seed_sd_m']:.3f} |")
    lines += ["", "### Match / role / PASS-subset ADE (meters)", "",
              "| Model | Subset | ADE mean +/- seed SD |", "|---|---|---:|"]
    for family, groups in group_summary.items():
        for group, item in groups.items():
            if group != "overall" and item["ade_mean_m"] is not None:
                lines.append(f"| {family} | {group} | {item['ade_mean_m']:.3f} +/- "
                             f"{item['ade_seed_sd_m']:.3f} |")
    lines += [
        "", "All errors are meters. ADE is the equal-weight mean of four horizon MDEs; FDE is 3 s MDE. "
        "Detailed JSON contains each match, role, PASS subset and valid-target/player coverage per run. "
        "Charts live in the experiment's `charts/` directory.",
        "", "## Scientific questions", "",
    ]
    social, cv, independent = (summary.get(key) for key in ("social", "constant_velocity", "independent"))
    if social and cv:
        delta = social["ade_mean_m"] - cv["ade_mean_m"]
        lines.append(f"1. **Beyond constant velocity:** Social mean ADE minus CV is {delta:+.3f} m. "
                     "A negative difference favors the neural model; inspect both matches and horizons.")
    else:
        lines.append("1. **Beyond constant velocity:** PENDING; no conclusion before held-out results.")
    if social and independent:
        delta = social["ade_mean_m"] - independent["ade_mean_m"]
        lines.append(f"2. **Interactions:** Social minus independent mean ADE is {delta:+.3f} m. "
                     "Matched-seed 30 s block CIs are in JSON. With only two validation matches, "
                     "population-level statistical significance is not established.")
    else:
        lines.append("2. **Interactions:** PENDING; a social-vs-CV improvement alone does not establish "
                     "an interaction benefit.")
    lines.append("3. **Prohibitive horizon:** No universal acceptable error threshold was specified. "
                 "Report measured errors through 3 s; do not invent a tactical usability cutoff. "
                 "Spatial quantization and unverified tracking limit fine interception interpretation.")
    static, short = summary.get("static"), summary.get("short_history")
    if social and static:
        lines.append(f"4. **History vs snapshot:** Static minus full-history ADE is "
                     f"{static['ade_mean_m'] - social['ade_mean_m']:+.3f} m"
                     + (f"; 0.4 s minus 2 s history is "
                        f"{short['ade_mean_m'] - social['ade_mean_m']:+.3f} m." if short else
                        "; short-history comparison pending."))
    else:
        lines.append("4. **History vs snapshot:** PENDING until the prespecified static/history ablations finish.")
    lines += [
        "5. **Pass feasibility:** At most a movement-encoding foundation, not measured pass feasibility, "
        "completion, intention or tactical value. Verified ball state/flight, synchronized contact, "
        "receiver intention/control, uncertainty under missing observations and task-specific labels "
        "are still needed.",
        "", "## Reusing the encoder", "",
        "The shared temporal encoder and social player embeddings could be joined to the existing "
        "PASS candidate ranker's actor/candidate features using identity matching outside the network. "
        "Fine-tuning or frozen-feature tests must use the existing split and avoid future-target leakage. "
        "No receiver ranker has been retrained or tactical benefit measured here.",
        "", "## Execution and limitations", "",
        f"Completed entries: {', '.join(sorted(completed)) or 'none yet'}.",
        "Best development checkpoints and selected epochs are retained. Final fits use that frozen "
        "duration, not validation early stopping. CPU threads=4, loader workers=0, CUDA allocator "
        "fraction=0.72, mixed precision, checkpoints every <=100 steps or approximately 60 seconds.",
        "Per-player identities are matching-only; storage order is not encoded. Attention permutation "
        "equivariance, target masks, splits and metric arithmetic are tested before launch.",
        "The optional probabilistic head uses isotropic Gaussian NLL; a scalar radial temperature is "
        "estimated on development only and transferred without held-out fitting.",
        "Within-match block bootstrap accounts partially for overlapping windows but cannot establish "
        "generalization beyond the two held-out matches. Seed SD and match differences are distinct.",
    ]
    if progress.get("blocker"):
        lines += ["", f"Blocker: {progress['blocker']}"]
    lines += ["", "## Resume", "", "```powershell", progress["resume_command"], "```", ""]
    (results / "trajectory_forecasting_report.md").write_text("\n".join(lines), encoding="utf-8")
    if summary:
        plot_comparisons(root, completed)


def experiment_config(base, candidate, family, seed, epochs=None):
    return {
        **{k: base[k] for k in ("batch_size", "weight_decay", "gradient_clip", "patience", "min_delta_m")},
        **candidate, "seed": seed, "epochs": epochs or base["max_epochs"],
        "social": family != "independent",
        "history_frames": 0 if family == "static" else 10 if family == "short_history" else 50,
        "probabilistic": family == "probabilistic",
    }


def smoke(root, base, device):
    """Real raw-track GPU smoke plus exact mid-epoch recovery equivalence."""
    pilot = root / "pilot" / "117092_1st.npz"
    meta = build_half(Path("data") / "117092_1st.json", pilot, frame_limit=3000)
    audit = feasibility(pilot)
    write_json(root / "pilot" / "feasibility.json", audit)
    if not audit["suitable"]:
        raise ValueError("Raw trajectory feasibility failed; no overnight launch")
    # A tiny Dataset wrapper reuses raw pilot windows without fabricating tracking data.
    with np.load(pilot) as blob:
        raw = {key: blob[key] for key in blob.files}
    from src.trajectory_dataset import make_window
    class PilotRows(list):
        match_names = [("117092", "1st")]

    items = PilotRows(make_window(raw, frame) for frame in range(50, 50 + 13 * 24, 13))
    for i, row in enumerate(items):
        row["source_index"], row["anchor_frame"] = 0, 50 + i * 13
    tiny = copy.deepcopy(base)
    tiny.update(batch_size=8, max_epochs=2, patience=2)
    conf = experiment_config(tiny, base["selection_candidates"][0], "social", 42)
    smoke_dir = root / "smoke" / "protocol_v1"
    reference = smoke_dir / "uninterrupted"
    resumed = smoke_dir / "resumed"
    dev = PilotRows(items[:8])
    first = fit(items, dev, conf, resumed, device, float("inf"), max_steps=1)
    if not first.get("interrupted_for_smoke") and not (resumed / "fit_result.json").exists():
        raise AssertionError("Smoke failed to save interruption checkpoint")
    fit(items, dev, conf, resumed, device, float("inf"))
    fit(items, dev, conf, reference, device, float("inf"))
    a = torch.load(resumed / "final.pt", weights_only=True)
    b = torch.load(reference / "final.pt", weights_only=True)
    max_delta = max(float((a[k] - b[k]).abs().max()) for k in a)
    if max_delta != 0:
        raise AssertionError(f"Resume does not reproduce uninterrupted training: {max_delta}")
    model = load_model(resumed, conf, device)
    batch = move(next(iter(loader(items[:8], 8))), device)
    model.eval()
    with torch.no_grad():
        prediction, _ = model(batch)
        permutation = torch.randperm(22, device=device)
        permuted = {k: v[:, permutation] if v.ndim >= 2 and v.shape[1] == 22 else v
                    for k, v in batch.items()}
        other, _ = model(permuted)
        delta = float((other - prediction[:, permutation]).abs().max())
        if delta > 1e-4:
            raise AssertionError(f"Permutation equivariance failed: {delta}")
        s, n = metric_sums(batch["target"] + 1, batch)
        if not np.allclose(s / n, math.sqrt(2), atol=1e-5):
            raise AssertionError("Metric correctness failure")
    eval_directory = smoke_dir / "evaluation"
    try:
        evaluate(model, dev, conf, device, time.monotonic() - 1, eval_directory)
    except BudgetExpired:
        if not (eval_directory / "evaluation_partial.pt").exists():
            raise AssertionError("Evaluation interruption failed to save progress")
    metrics = evaluate(model, dev, conf, device, float("inf"), eval_directory)
    before = (eval_directory / "metrics.json").stat().st_mtime_ns
    again = evaluate(model, dev, conf, device, float("inf"), eval_directory)
    if metrics != again or before != (eval_directory / "metrics.json").stat().st_mtime_ns:
        raise AssertionError("Successful evaluation was rerun instead of reused")
    before = (resumed / "last.pt").stat().st_mtime_ns
    fit(items, dev, conf, resumed, device, float("inf"))
    if before != (resumed / "last.pt").stat().st_mtime_ns:
        raise AssertionError("Successful fit was rerun")
    result = {
        "status": "PASS", "cuda": torch.cuda.get_device_name(), "raw_pilot_quality": meta["quality"],
        "feasibility": audit, "resume_max_parameter_delta": max_delta,
        "permutation_max_delta": delta, "checkpoint_optimizer_rng_recovery": "PASS",
        "metric_known_sqrt2_m": "PASS", "created_unix": time.time(),
        "evaluation_interrupt_resume_and_completed_skip": "PASS",
        "completed_fit_skip": "PASS", "code_sha256": code_hashes(),
    }
    write_json(root / "smoke" / "smoke_result.json", result)
    LOG.info("SMOKE PASS %s", json.dumps(result))


def run_queue(root, base, device, args):
    progress_path = root / "progress.json"
    resume = (f'& "{sys.executable}" train\\run_trajectory_forecasting.py '
              f'--output "{root.resolve()}" --config "{args.config.resolve()}"')
    if progress_path.exists():
        progress = load_json(progress_path)
        if progress["config"] != base:
            raise ValueError("Experiment configuration changed on resume")
    else:
        code_paths = [Path(__file__), Path("src/trajectory_dataset.py"),
                      Path("src/models/trajectory_forecasting.py"), args.config]
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        progress = {
            "started_unix": time.time(), "completed": {}, "selected": {}, "config": base,
            "resume_command": resume, "git_sha": git_sha,
            "code_sha256": {str(p): file_digest(p) for p in code_paths},
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "status": "RUNNING", "pid": os.getpid(),
            "input_protection": "Only raw GSR and BAS evaluation subsets are read. No writes to data/ "
                                "or existing results/review material.",
        }
    current_hashes = {str(Path(p)): file_digest(p) for p in progress["code_sha256"]}
    if current_hashes != progress["code_sha256"]:
        raise ValueError("Code/config changed since launch; refuse scientifically ambiguous resume")
    progress.update(pid=os.getpid(), status="RUNNING", last_started_unix=time.time())
    progress.pop("blocker", None)
    deadline = time.monotonic() + max(0, base["max_hours"] * 3600 - (time.time() - progress["started_unix"]))

    def update(stage):
        progress["stage"] = stage
        progress["updated_unix"] = time.time()
        write_json(progress_path, progress)
        LOG.info("STAGE %s", stage)

    try:
        check_deadline(deadline)
        if not (root / "smoke" / "smoke_result.json").exists():
            raise ValueError("Run --smoke and focused tests before the overnight queue")
        if load_json(root / "smoke" / "smoke_result.json")["status"] != "PASS":
            raise ValueError("Smoke did not pass")
        if load_json(root / "smoke" / "smoke_result.json").get("code_sha256") != code_hashes():
            raise ValueError("Source changed after smoke; rerun --smoke before launch")
        publish(root, progress, "RUNNING")
        cache = root / "cache"
        quality = {}
        for match in (*TRAIN_MATCHES, *VALIDATION_MATCHES):
            for half in ("1st", "2nd"):
                update(f"cache_{match}_{half}")
                check_deadline(deadline)
                source = Path("data") / f"{match}_{half}.json"
                quality[f"{match}_{half}"] = build_half(
                    source, cache / f"{match}_{half}.npz", deadline
                )
                progress.setdefault("cached_halves", {})[f"{match}_{half}"] = quality[f"{match}_{half}"]["quality"]
                write_json(progress_path, progress)
        write_json(root / "dataset_provenance.json", quality)
        development = TrajectoryDataset(cache, [DEVELOPMENT_MATCH], base["stride_frames"])
        fitting = TrajectoryDataset(
            cache, [m for m in TRAIN_MATCHES if m != DEVELOPMENT_MATCH], base["stride_frames"]
        )
        validate_splits(TRAIN_MATCHES, VALIDATION_MATCHES)
        progress["dataset_windows"] = {"development": len(development), "selection_fitting": len(fitting)}
        for name in ("persistence", "constant_velocity"):
            key = f"development_{name}"
            if name not in progress.get("development_baselines", {}):
                update(key)
                metrics = evaluate(name, development, base, device, deadline,
                                   root / "development_baselines" / name)
                # Development baselines are not held-out entries.
                progress.setdefault("development_baselines", {})[name] = metrics
                write_json(progress_path, progress)
        validation, final_training, bas = None, None, None
        for family in base["experiment_order"]:
            check_deadline(deadline)
            update(f"selection_{family}")
            if family not in progress["selected"]:
                if family == "social":
                    candidates = base["selection_candidates"]
                else:
                    candidates = [progress["selected"]["social"]["candidate"]]
                choices = []
                for index, candidate in enumerate(candidates):
                    directory = root / "selection" / f"{family}_{index}"
                    config = experiment_config(base, candidate, family, 42)
                    result = fit(fitting, development, config, directory, device, deadline)
                    choices.append({"candidate": candidate, "fit": result, "directory": str(directory)})
                chosen = min(choices, key=lambda c: (
                    c["fit"]["best_development_ade_m"], c["fit"]["parameters"]
                ))
                chosen["epochs"] = chosen["fit"]["best_epoch"]
                chosen["selection_rule"] = "Minimum development ADE, tie smaller parameter count"
                chosen["temperature"] = 1.0
                if family == "probabilistic":
                    config = experiment_config(base, chosen["candidate"], family, 42)
                    model = load_model(Path(chosen["directory"]), config, device, best=True)
                    calibration = evaluate(model, development, config, device, deadline)
                    radius = calibration["uncertainty"]["mean_squared_standardized_radius"]
                    chosen["temperature"] = float(max(.05, math.sqrt(np.mean(radius) / 2)))
                    del model
                progress["selected"][family] = chosen
                write_json(root / "selected_configurations.json", progress["selected"])
                write_json(progress_path, progress)
            chosen = progress["selected"][family]
            if final_training is None:
                final_training = TrajectoryDataset(cache, TRAIN_MATCHES, base["stride_frames"])
                progress["dataset_windows"]["final_fitting"] = len(final_training)
            for seed in base["seeds"]:
                name = f"{family}_seed{seed}"
                if name in progress["completed"]:
                    continue
                check_deadline(deadline)
                update(f"fit_{name}")
                config = experiment_config(base, chosen["candidate"], family, seed, chosen["epochs"])
                directory = root / "fits" / name
                fit_result = fit(final_training, None, config, directory, device, deadline)
                # Held-out data is instantiated only after development selection is frozen.
                if validation is None:
                    validation = TrajectoryDataset(cache, VALIDATION_MATCHES, base["stride_frames"])
                    bas = pass_frames(Path("data"), validation)
                    progress["dataset_windows"]["heldout"] = len(validation)
                for baseline in ("persistence", "constant_velocity"):
                    if baseline not in progress["completed"]:
                        update(f"evaluate_{baseline}")
                        metrics = evaluate(baseline, validation, base, device, deadline,
                                           root / "evaluation" / baseline, bas)
                        progress["completed"][baseline] = {"metrics": metrics}
                        write_json(progress_path, progress)
                update(f"evaluate_{name}")
                model = load_model(directory, config, device)
                metrics = evaluate(model, validation, config, device, deadline,
                                   root / "evaluation" / name, bas, chosen["temperature"])
                progress["completed"][name] = {"fit": fit_result, "metrics": metrics, "seed": seed}
                del model
                torch.cuda.empty_cache()
                update(f"completed_{name}")
                publish(root, progress, "RUNNING")
        # Verify read-only source provenance again after all fits.
        for meta in quality.values():
            if source_signature(meta["source"]["path"]) != meta["source"]:
                raise ValueError("Raw source file changed during experiment")
        progress["status"] = "COMPLETE"
        update("complete")
        publish(root, progress, "COMPLETE")
    except BudgetExpired as error:
        progress.update(status="TIME_LIMIT", blocker=str(error))
        update("paused_budget")
        publish(root, progress, "TIME_LIMIT")
        LOG.warning("SAFE STOP: %s", error)
    except Exception as error:
        progress.update(status="FAILED", blocker=f"{type(error).__name__}: {error}")
        update("failed")
        publish(root, progress, "FAILED")
        LOG.exception("EXPERIMENT FAILED; checkpoints retained")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/trajectory_forecasting.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trajectory_forecasting_20261016"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root, base = args.output, load_json(args.config)
    if base["train_matches"] != list(TRAIN_MATCHES) or base["validation_matches"] != list(VALIDATION_MATCHES):
        raise ValueError("Do not alter frozen matches")
    if base["development_match"] != DEVELOPMENT_MATCH or base["horizons_frames"] != list(HORIZONS):
        raise ValueError("Invalid development split or exact target offsets")
    if not 0 < base["max_hours"] <= 14 or base["cpu_threads"] != 4 or base["loader_workers"] > 2:
        raise ValueError("Resource policy violation")
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(root / "experiment.log", encoding="utf-8"), logging.StreamHandler()],
    )
    torch.set_num_threads(base["cpu_threads"])
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: refusing silent CPU training fallback")
    torch.cuda.set_per_process_memory_fraction(base["gpu_memory_fraction"], 0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    if args.smoke:
        smoke(root, base, device)
    else:
        # A file lock prevents concurrent orchestrators without deleting stale lock files.
        import msvcrt
        with (root / "orchestrator.lock").open("a+b") as lock:
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            run_queue(root, base, device, args)


if __name__ == "__main__":
    main()
