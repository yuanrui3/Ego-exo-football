"""Single resumable match-cross-fitted receiver integration experiment."""

import argparse
import copy
import itertools
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from src.evaluate_pass_receiver_model import LABELS, rank_candidates, ranking_metrics
from src.models.receiver_forecast_ranker import ReceiverForecastRanker, feature_statistics
from src.receiver_forecast_adapter import (
    FORECAST_NAMES, assert_out_of_fold, base_engineered, config_hash,
    forecast_features, prepare_receiver_inputs, select_forecast_group,
)
from src.receiver_tensor_dataset import read_receiver_rows, validate_match_split
from src.trajectory_dataset import (
    DEVELOPMENT_MATCH, TRAIN_MATCHES, VALIDATION_MATCHES, BudgetExpired,
    TrajectoryDataset, check_deadline, file_digest, write_json,
)
from train.run_trajectory_forecasting import (
    fit as fit_forecaster, load_model as load_forecaster, move,
    restore_rng, rng_state, save_torch, seed_all,
)

LOG = logging.getLogger("receiver_integration")
BASE_TRAIN = tuple(m for m in TRAIN_MATCHES if m != DEVELOPMENT_MATCH)
METRICS = ("top1", "top3", "top5", "mrr", "macro_f1")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    paths = (
        Path(__file__), Path("src/receiver_forecast_adapter.py"),
        Path("src/models/receiver_forecast_ranker.py"), Path("src/trajectory_dataset.py"),
        Path("src/models/trajectory_forecasting.py"), Path("train/run_trajectory_forecasting.py"),
        Path("src/audit_trajectory_integration.py"), Path("src/models/candidate_ranker_receiver.py"),
    )
    return {str(p.resolve()): file_digest(p) for p in paths}


def fold_sets(stage, predicted):
    if stage not in ("development", "final"):
        raise ValueError("Unknown fold stage")
    allowed = BASE_TRAIN if stage == "development" else TRAIN_MATCHES
    fits = tuple(sorted(m for m in allowed if m != str(predicted)))
    assert_out_of_fold(predicted, fits)
    # Inner stopping uses a match inside this fit set, never the prediction match.
    inner = fits[-1]
    return fits, inner


def forecast_config(base, family, epochs=None):
    conf = {k: v for k, v in base["forecast"].items() if k != "stride_frames"}
    conf["epochs"] = conf.pop("max_epochs") if epochs is None else epochs
    conf.pop("max_epochs", None)
    conf.update(seed=base["seed_forecast"], social=family == "social",
                history_frames=50, probabilistic=False)
    return conf


def upstream_model(root, base, family, fitting, inner, deadline, stage_callback):
    specification = {
        "family": family, "fitting_matches": list(fitting), "inner_development_match": inner,
        "configuration": base["forecast"], "seed": base["seed_forecast"],
        "architecture_context": base["reused_development_context"],
    }
    digest = config_hash(specification)
    directory = root / "folds" / digest
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "provenance.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["specification"] != specification:
            raise ValueError("Upstream hash collision or changed fold configuration")
        if manifest["status"] == "COMPLETE":
            if manifest["checkpoint_sha256"] != file_digest(directory / "refit/final.pt"):
                raise ValueError("Upstream checkpoint checksum mismatch")
            return manifest, directory / "refit"
    started = time.monotonic()
    manifest = {"status": "RUNNING", "specification": specification, "hash": digest}
    write_json(manifest_path, manifest)
    cache = Path(base["trajectory_root"]) / "cache"
    inner_fit = tuple(m for m in fitting if m != inner)
    if not inner_fit or inner not in fitting:
        raise ValueError("Invalid inner upstream split")
    stage_callback(f"forecast_{family}_{digest}_inner")
    config = forecast_config(base, family)
    result_path = directory / "inner/fit_result.json"
    if result_path.exists():
        selected = read_json(result_path)
    else:
        fitting_data = TrajectoryDataset(cache, inner_fit, base["forecast"]["stride_frames"])
        stopping_data = TrajectoryDataset(cache, [inner], base["forecast"]["stride_frames"])
        selected = fit_forecaster(fitting_data, stopping_data, config, directory / "inner",
                                  torch.device("cuda"), deadline)
        del fitting_data, stopping_data
    check_deadline(deadline)
    stage_callback(f"forecast_{family}_{digest}_refit")
    refit_config = forecast_config(base, family, selected["best_epoch"])
    if (directory / "refit/fit_result.json").exists():
        refit = read_json(directory / "refit/fit_result.json")
    else:
        fitting_data = TrajectoryDataset(cache, fitting, base["forecast"]["stride_frames"])
        refit = fit_forecaster(fitting_data, None, refit_config, directory / "refit",
                              torch.device("cuda"), deadline)
        del fitting_data
    manifest.update(
        status="COMPLETE", inner_fitting_matches=list(inner_fit), selection_matches=[inner],
        selected_epochs=selected["best_epoch"], best_inner_ade_m=selected["best_development_ade_m"],
        refit_config=refit_config, parameter_count=refit["parameters"],
        checkpoint_sha256=file_digest(directory / "refit/final.pt"),
        completed_unix=time.time(), invocation_seconds=time.monotonic() - started,
    )
    write_json(manifest_path, manifest)
    torch.cuda.empty_cache()
    return manifest, directory / "refit"


def adapter_cache(root, base, rows, timing):
    path = root / "adapter" / f"{timing}.pt"
    if path.exists():
        result = torch.load(path, map_location="cpu", weights_only=False)
        if result["ids"] != [r["id"] for r in rows]:
            raise ValueError("Adapter cache has different frozen example IDs")
        for name, signatures in result["source_signatures"].items():
            cache = Path(base["trajectory_root"]) / "cache" / f"{name}.npz"
            association = Path(base["association_root"]) / f"associations_{name}.json"
            if file_digest(cache) != signatures["cache"] or file_digest(association) != signatures["association"]:
                raise ValueError("Verified input caches changed since adapter construction")
        return result
    result = prepare_receiver_inputs(
        rows, Path(base["trajectory_root"]), Path(base["association_root"]), timing
    )
    path.parent.mkdir(exist_ok=True, parents=True)
    save_torch(path, result)
    write_json(path.with_suffix(".json"), {
        key: result[key] for key in ("ids", "matches", "reasons", "provenance", "source_signatures", "timing")
    })
    return result


@torch.no_grad()
def infer_features(adapter, indices, model, device):
    values = torch.zeros(len(indices), 10, len(FORECAST_NAMES))
    valid = torch.zeros_like(values, dtype=torch.bool)
    available = np.asarray(adapter["covered"])[indices]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if model is not None:
        model.eval()
    for start in range(0, len(indices), 128):
        positions = np.flatnonzero(available[start:start + 128]) + start
        if not len(positions):
            continue
        actual = np.asarray(indices)[positions]
        batch = {k: torch.as_tensor(v[actual]).to(device) for k, v in adapter["inputs"].items()}
        if model is None:
            times = batch["current"].new_tensor([.4, 1., 2., 3.])
            prediction = batch["velocity"][:, :, None] * times[None, None, :, None]
        else:
            with torch.autocast("cuda", dtype=dtype):
                prediction, _ = model(batch)
        future = batch["current"][:, :, None] + prediction.float()
        feature, mask = forecast_features(
            batch["current"], future, batch["player_mask"],
            batch["history"][..., 4].float().mean(-1),
            torch.ones(len(actual), dtype=torch.bool, device=device),
        )
        values[positions], valid[positions] = feature.cpu(), mask.cpu()
    return values, valid


def feature_stage(root, base, adapter, family, stage, deadline, callback):
    path = root / "features" / f"{adapter['timing']}_{family}_{stage}.pt"
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["ids"] != adapter["ids"]:
            raise ValueError("Forecast feature cache row mismatch")
        for record in cached["provenance"]:
            assert_out_of_fold(record["predicted_match"], record["fitting_matches"], record["selection_matches"])
        return cached["values"], cached["valid"]
    n = len(adapter["ids"])
    values = torch.zeros(n, 10, len(FORECAST_NAMES))
    valid = torch.zeros_like(values, dtype=torch.bool)
    manifests = []
    predicted_matches = TRAIN_MATCHES if stage == "final" else (*BASE_TRAIN, DEVELOPMENT_MATCH)
    if stage == "final":
        predicted_matches = (*predicted_matches, *VALIDATION_MATCHES)
    for predicted in predicted_matches:
        check_deadline(deadline)
        indices = [i for i, match in enumerate(adapter["matches"]) if match == predicted]
        if family == "cv":
            fitting, inner, manifest = (), None, None
            unique = "deterministic_cv"
        else:
            fitting, inner = fold_sets(stage, predicted)
            assert_out_of_fold(predicted, fitting, [inner])
            manifest, model_directory = upstream_model(
                root, base, family, fitting, inner, deadline, callback
            )
            unique = manifest["hash"]
        per_path = root / "features" / "fold_cache" / f"{adapter['timing']}_{family}_{predicted}_{unique}.pt"
        if per_path.exists():
            saved = torch.load(per_path, map_location="cpu", weights_only=False)
            if saved["ids"] != [adapter["ids"][i] for i in indices]:
                raise ValueError("Fold prediction cache ID mismatch")
            f, v = saved["values"], saved["valid"]
        else:
            callback(f"infer_{adapter['timing']}_{family}_{stage}_{predicted}")
            before = time.monotonic()
            model = None if family == "cv" else load_forecaster(
                model_directory, manifest["refit_config"], torch.device("cuda")
            )
            f, v = infer_features(adapter, indices, model, torch.device("cuda"))
            if model is not None:
                del model
            torch.cuda.synchronize()
            per_path.parent.mkdir(parents=True, exist_ok=True)
            save_torch(per_path, {
                "ids": [adapter["ids"][i] for i in indices], "values": f, "valid": v,
                "inference_seconds": time.monotonic() - before,
            })
        values[indices], valid[indices] = f, v
        if family != "cv":
            manifests.append({
                "predicted_match": predicted, "fitting_matches": list(fitting),
                "selection_matches": [inner], "fold_hash": unique,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "example_ids": [adapter["ids"][i] for i in indices],
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    save_torch(path, {"values": values, "valid": valid, "provenance": manifests, "ids": adapter["ids"]})
    write_json(path.with_suffix(".json"), {"provenance": manifests, "feature_names": FORECAST_NAMES})
    return values, valid


def metric_from_logits(logits, targets):
    if len(targets) == 0:
        return {"n": 0}
    ranked = [rank_candidates(dict(zip(LABELS, row))) for row in logits.tolist()]
    result = ranking_metrics([LABELS[int(t)] for t in targets], ranked)
    result["prediction_distribution"] = {k: result["per_class"][k]["prediction_count"] for k in LABELS}
    return result


@torch.no_grad()
def receiver_logits(model, data, indices, device):
    model.eval()
    output = []
    for start in range(0, len(indices), 256):
        selected = indices[start:start + 256]
        logits = model(
            data["values"][selected].to(device), data["valid"][selected].to(device),
            data["candidates"][selected].to(device),
        ).float().cpu()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite receiver logits")
        output.append(logits)
    return torch.cat(output)


def receiver_data(adapter, forecast=None, variant="A"):
    values, valid = base_engineered(adapter["states"])
    if variant != "A":
        f, v = select_forecast_group(*forecast, variant)
        values, valid = torch.cat((values, f), -1), torch.cat((valid, v), -1)
    return {
        "values": values, "valid": valid,
        "targets": torch.as_tensor(adapter["targets"], dtype=torch.long),
        "candidates": torch.as_tensor(adapter["states"][:, -1, 1:11, 2], dtype=torch.bool),
    }


def fit_receiver(data, fitting, dev, config, directory, deadline, max_steps=None):
    invocation_started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "config.json").exists() and read_json(directory / "config.json") != config:
        raise ValueError("Receiver resume configuration differs")
    write_json(directory / "config.json", config)
    if (directory / "fit_result.json").exists():
        return read_json(directory / "fit_result.json")
    seed_all(config["seed"])
    mean, std = feature_statistics(data["values"][fitting], data["valid"][fitting])
    model = ReceiverForecastRanker(mean, std, config["width"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    state = {"epoch": 0, "next_batch": 0, "steps": 0, "best_mrr": -1., "best_epoch": 0,
             "stale": 0, "history": [], "loss_sum": 0., "batches": 0}
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        restore_rng(saved["rng"])
    previous_seconds = state.get("runtime_seconds", 0.)
    device = torch.device("cuda")
    last_save = time.monotonic()

    def save():
        state["runtime_seconds"] = previous_seconds + time.monotonic() - invocation_started
        save_torch(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "state": state, "rng": rng_state()})
        write_json(directory / "live.json", {
            "epoch": state["epoch"], "next_batch": state["next_batch"], "steps": state["steps"],
            "best_development_mrr": state["best_mrr"], "best_epoch": state["best_epoch"],
        })

    while state["epoch"] < config["epochs"]:
        epoch = state["epoch"]
        order = np.asarray(fitting)[np.random.default_rng(config["seed"] + epoch).permutation(len(fitting))]
        model.train()
        for start in range(state["next_batch"] * config["batch_size"], len(order), config["batch_size"]):
            if time.monotonic() >= deadline:
                save()
                raise BudgetExpired("Receiver budget reached; exact batch/optimizer/RNG saved")
            selected = order[start:start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(data["values"][selected].to(device), data["valid"][selected].to(device),
                           data["candidates"][selected].to(device))
            loss = nn.functional.cross_entropy(logits, data["targets"][selected].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite receiver loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            state["steps"] += 1
            state["next_batch"] = start // config["batch_size"] + 1
            state["loss_sum"] += float(loss.detach())
            state["batches"] += 1
            if time.monotonic() - last_save >= 60 or state["steps"] % 100 == 0:
                save()
                last_save = time.monotonic()
            if max_steps and state["steps"] >= max_steps:
                save()
                return {"smoke_paused": True}
        row = {"epoch": epoch + 1, "loss": state["loss_sum"] / max(1, state["batches"])}
        if dev is not None:
            scores = receiver_logits(model, data, dev, device)
            metrics = metric_from_logits(scores, data["targets"][dev])
            row["development_mrr"], row["development_top1"] = metrics["mrr"], metrics["top1"]
            if metrics["mrr"] > state["best_mrr"] + 1e-5:
                state["best_mrr"], state["best_epoch"], state["stale"] = metrics["mrr"], epoch + 1, 0
                save_torch(directory / "best.pt", model.state_dict())
            else:
                state["stale"] += 1
        state["epoch"], state["next_batch"] = epoch + 1, 0
        state["loss_sum"], state["batches"] = 0., 0
        state["history"].append(row)
        save()
        write_json(directory / "history.json", state["history"])
        LOG.info("RECEIVER %s %s", directory.name, row)
        if dev is not None and state["stale"] >= config["patience"]:
            break
    save_torch(directory / "final.pt", model.state_dict())
    result = {
        "status": "COMPLETE", "best_epoch": state["best_epoch"], "best_mrr": state["best_mrr"],
        "completed_epochs": state["epoch"], "steps": state["steps"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "feature_count": data["values"].shape[-1],
        "normalization_fitting_examples": len(fitting),
        "training_runtime_seconds": previous_seconds + time.monotonic() - invocation_started,
    }
    write_json(directory / "fit_result.json", result)
    del model
    return result


def load_receiver(directory, data, config, fitting):
    mean, std = feature_statistics(data["values"][fitting], data["valid"][fitting])
    model = ReceiverForecastRanker(mean, std, config["width"]).cuda()
    model.load_state_dict(torch.load(directory / "final.pt", map_location="cpu", weights_only=True))
    return model


def receiver_config(base, width, lr, seed=42, epochs=None):
    r = base["receiver"]
    return {"width": width, "learning_rate": lr, "seed": seed,
            "epochs": r["max_epochs"] if epochs is None else epochs,
            "batch_size": r["batch_size"], "patience": r["patience"], "weight_decay": r["weight_decay"]}


def fit_variant(root, base, adapter, forecast_dev, forecast_final, variant, deadline, callback):
    prefix = f"{adapter['timing']}_{variant}"
    result_path = root / "receiver" / f"{prefix}_result.json"
    if result_path.exists():
        return read_json(result_path)
    matches = np.asarray(adapter["matches"])
    good = np.asarray(adapter["base_valid"])
    fitting = np.flatnonzero(np.isin(matches, BASE_TRAIN) & good)
    development = np.flatnonzero((matches == DEVELOPMENT_MATCH) & good)
    final = np.flatnonzero(np.isin(matches, TRAIN_MATCHES) & good)
    heldout = np.flatnonzero(np.isin(matches, VALIDATION_MATCHES) & good)
    dev_data = receiver_data(adapter, forecast_dev, variant)
    candidates = []
    for width, lr in itertools.product(base["receiver"]["widths"], base["receiver"]["learning_rates"]):
        conf = receiver_config(base, width, lr)
        directory = root / "receiver/selection" / f"{prefix}_{config_hash(conf)}"
        callback(f"receiver_selection_{prefix}_{width}_{lr}")
        fit = fit_receiver(dev_data, fitting, development, conf, directory, deadline)
        candidates.append({"config": conf, "fit": fit})
    chosen = max(candidates, key=lambda c: (c["fit"]["best_mrr"], -c["fit"]["parameter_count"]))
    del dev_data
    # Freeze receiver choice before consulting held-out predictions/metrics.
    selected_path = root / "receiver/selected" / f"{prefix}.json"
    write_json(selected_path, chosen)
    data = receiver_data(adapter, forecast_final, variant)
    runs = []
    for seed in base["receiver_seeds"]:
        conf = {**chosen["config"], "seed": seed, "epochs": chosen["fit"]["best_epoch"]}
        directory = root / "receiver/final" / f"{prefix}_seed{seed}"
        callback(f"receiver_refit_{prefix}_seed{seed}")
        fit = fit_receiver(data, final, None, conf, directory, deadline)
        prediction_path = directory / "predictions.pt"
        if prediction_path.exists():
            prediction = torch.load(prediction_path, map_location="cpu", weights_only=False)
        else:
            check_deadline(deadline)
            model = load_receiver(directory, data, conf, final)
            before = time.monotonic()
            logits = receiver_logits(model, data, heldout, torch.device("cuda"))
            torch.cuda.synchronize()
            prediction = {"indices": heldout, "ids": [adapter["ids"][i] for i in heldout],
                          "logits": logits, "targets": data["targets"][heldout],
                          "inference_seconds": time.monotonic() - before}
            if variant != "A":
                uncovered = ~np.asarray(adapter["covered"])[heldout]
                fallback = torch.load(
                    root / "receiver/final" / f"{adapter['timing']}_A_seed{seed}" / "predictions.pt",
                    map_location="cpu", weights_only=False,
                )
                if prediction["ids"] != fallback["ids"]:
                    raise ValueError("Fallback receiver ordering mismatch")
                prediction["logits"][uncovered] = fallback["logits"][uncovered]
            if variant == "D":
                shuffled_path = directory / "shuffled_forecast_predictions.pt"
                f, v = forecast_final
                # Permute only within held-out match, covered examples; base stays unchanged.
                shuffled_f, shuffled_v = f.clone(), v.clone()
                for match in VALIDATION_MATCHES:
                    ix = np.flatnonzero((matches == match) & good & np.asarray(adapter["covered"]))
                    order = np.random.default_rng(42).permutation(ix)
                    shuffled_f[ix], shuffled_v[ix] = f[order], v[order]
                shuffled_data = receiver_data(adapter, (shuffled_f, shuffled_v), variant)
                shuffled = receiver_logits(model, shuffled_data, heldout, torch.device("cuda"))
                uncovered = ~np.asarray(adapter["covered"])[heldout]
                shuffled[uncovered] = prediction["logits"][uncovered]
                save_torch(shuffled_path, {"logits": shuffled, "indices": heldout,
                                           "targets": data["targets"][heldout]})
                del shuffled_data
            save_torch(prediction_path, prediction)
            with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
                for example_id, scores, target in zip(
                    prediction["ids"], prediction["logits"].tolist(), prediction["targets"].tolist()
                ):
                    ranked = rank_candidates(dict(zip(LABELS, scores)))
                    stream.write(json.dumps({
                        "id": example_id, "true_target": LABELS[target], "predicted_target": ranked[0],
                        "ranked_targets": ranked, "logits": dict(zip(LABELS, scores)),
                    }) + "\n")
            del model
        coverage = np.asarray(adapter["covered"])[heldout]
        common = metric_from_logits(prediction["logits"][coverage], prediction["targets"][coverage])
        full = metric_from_logits(prediction["logits"], prediction["targets"])
        per_match = {}
        for match in VALIDATION_MATCHES:
            flag = (matches[heldout] == match) & coverage
            per_match[match] = metric_from_logits(prediction["logits"][flag], prediction["targets"][flag])
        subsets = {}
        states = torch.as_tensor(adapter["states"][heldout])
        xy = states[:, -1, :, :2] * torch.tensor([52.5, 34])
        distances = torch.linalg.vector_norm(xy[:, 1:11] - xy[:, :1], dim=-1)
        nearest = distances.masked_fill(states[:, -1, 1:11, 2] == 0, float("inf")).argsort(stable=True)[:, 0]
        pressure = (torch.cdist(xy[:, :1], xy[:, 11:]) < 5) & (states[:, -1, 11:, 2, None].transpose(1, 2) > 0)
        movement = torch.linalg.vector_norm(
            (states[:, -1, 1:11, :2] - states[:, 0, 1:11, :2]) * torch.tensor([52.5, 34]), dim=-1
        )
        gaps = distances.sort().values[:, 1] - distances.sort().values[:, 0]
        flags = {
            "receiver_not_nearest_label_conditioned": prediction["targets"] != nearest,
            "actor_opponent_within5m": pressure.any((1, 2)),
            "any_candidate_moving_over0_5m_in0_4s": (movement > .5).any(-1),
            "nearest_two_distance_gap_under2m": gaps < 2,
        }
        for name, flag in flags.items():
            flag = flag & torch.as_tensor(coverage)
            subsets[name] = metric_from_logits(prediction["logits"][flag], prediction["targets"][flag])
        run = {
            "seed": seed, "fit": fit, "common": common, "full_fallback": full,
            "per_match_common": per_match, "subsets_common": subsets,
            "inference_seconds": prediction["inference_seconds"],
        }
        if variant == "D":
            shuffled = torch.load(directory / "shuffled_forecast_predictions.pt", weights_only=False)
            run["shuffled_common"] = metric_from_logits(shuffled["logits"][coverage], prediction["targets"][coverage])
        runs.append(run)
    result = {"variant": variant, "timing": adapter["timing"], "selected": chosen,
              "runs": runs, "coverage": int(np.asarray(adapter["covered"])[heldout].sum()),
              "heldout_base_valid": len(heldout)}
    write_json(result_path, result)
    return result


def matched_historical(root, adapter):
    if adapter["timing"] != "legacy":
        return {}
    heldout = [i for i, match in enumerate(adapter["matches"]) if match in VALIDATION_MATCHES]
    ids = [adapter["ids"][i] for i in heldout]
    covered = np.asarray(adapter["covered"])[heldout]
    results = {}
    for name in ("v2", "engineered_only"):
        results[name] = {}
        for seed in (42, 7, 123):
            directory = (Path("outputs") / f"candidate_ranker_receiver_seed{seed}" if name == "v2"
                         else Path("outputs/candidate_ranker_ablation") / f"engineered_only_seed{seed}")
            path = directory / "val_predictions.jsonl"
            if not path.exists():
                results[name][str(seed)] = {"status": "UNAVAILABLE", "path": str(path)}
                continue
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if [r["id"] for r in rows] != ids:
                raise ValueError("Historical baseline does not match frozen validation order")
            true = [r["true_target"] for r, flag in zip(rows, covered) if flag]
            ranked = [r["ranked_targets"] for r, flag in zip(rows, covered) if flag]
            results[name][str(seed)] = ranking_metrics(true, ranked)
    write_json(root / "historical_v2_matched_subset.json", results)
    return results


def publish(root, progress):
    summaries = {}
    for name, result in progress.get("receiver_results", {}).items():
        entry = {}
        for group in ("common", "full_fallback"):
            entry[group] = {
                metric: {
                    "mean": float(np.mean([r[group][metric] for r in result["runs"]])),
                    "seed_sd": float(np.std([r[group][metric] for r in result["runs"]])),
                } for metric in METRICS
            }
        entry.update(coverage=result["coverage"], heldout_base_valid=result["heldout_base_valid"],
                     feature_count=result["runs"][0]["fit"]["feature_count"],
                     parameters=result["runs"][0]["fit"]["parameter_count"])
        summaries[name] = entry
    completed_folds = []
    for path in sorted((root / "folds").glob("*/provenance.json")):
        fold = read_json(path)
        if fold["status"] == "COMPLETE":
            completed_folds.append(fold)
    progress["completed_forecast_folds"] = len(completed_folds)
    progress["elapsed_seconds"] = progress.get("elapsed_offset_seconds", 0.) + time.time() - progress["started_unix"]
    progress["updated_unix"] = time.time()
    write_json(root / "progress.json", progress)
    metrics = {
        "status": progress["status"], "progress": progress,
        "summary": summaries, "fold_provenance": completed_folds,
        "interpretation": "Next-annotated-actor pseudo-label imitation only. Legacy snapshots 20--60 ms "
                          "after BAS time. Reused-development architecture context; no fresh-test claim.",
    }
    write_json(root / "metrics.json", metrics)
    write_json(Path("results/receiver_forecast_integration_metrics.json"), metrics)
    lines = [
        "# Social-forecast receiver integration", "",
        f"Status: **{progress['status']}**. Elapsed: {progress['elapsed_seconds']/3600:.2f} hours.",
        f"Experiment: `{root.resolve()}`. Completed upstream forecast fit sets: {len(completed_folds)}.",
        f"Stage: {progress.get('stage', 'initializing')}.", "",
        "## Protocol and caveats", "",
        "Seven frozen training matches, development 132877, held-out 118577/128058. Upstream forecasts "
        "are match-excluded: development fitting rows use leave-one-out within six (five-match models), "
        "development uses a six-match model, final fitting uses six-match OOF models and validation "
        "uses seven-match refits. Each fit set selects duration on an inner match in its own fitting "
        "partition, then refits every allowed fitting match. Predicted matches are forbidden in both "
        "fitting and stopping. No forecast uncertainty calibration is used.",
        "The width128/3-layer/lr0.001 forecasting architecture was selected previously using 132877. "
        "This is disclosed reused-development context, not nested fresh architecture selection. "
        "New epoch stopping never sees the predicted match. Validation benchmarks have previously "
        "published results and are not a fresh confirmatory test set.",
        "Primary timing preserves legacy GSR matched frames, approximately 20--60 ms after BAS time. "
        "It must not be called strictly pre-pass. Separate as-of sensitivity regenerates base features, "
        "candidate order/labels and history at floor(BAS_ms/40), not later than the documented event "
        "timestamp; contact/video synchronization and upstream GSR causal generation remain unverified.",
        "Features contain no IDs, actual future positions, receiver target, gap duration, ball speed or "
        "future reception time. Inputs end at t; displacements are predicted at .4/1/2/3 s. "
        "Movement direction is relative to the current actor-to-candidate line, not an assumed "
        "attacking direction. Space is distance to observed defenders, not certified free space.",
        "All rankers have the same shared candidate MLP form and receive the same width128/256 x "
        "lr.001/.0003 development grid, MRR early stopping, and seeds42/7/123. Different feature counts "
        "change parameter counts; baseline is retuned with the same budget. Selection normalization "
        "uses six fitting matches; final normalization uses seven-match OOF features only.",
        "A=base engineered, B=+CV, C=+independent, D=+social, E=D without explicit forecast pressure, "
        "F=D without explicit forecast movement. Removal does not eliminate correlated information.",
        "Primary comparison is the same 1,222 covered validation rows, including re-scored historical "
        "v2 predictions. Full 1,224-row metrics use same-seed A fallback for uncovered rows; no label "
        "or future availability is used for inference filtering. The as-of sensitivity has its own "
        "reported observed-target/coverage denominator and must not be conflated with legacy scores.",
        "", "## Coverage", "", json.dumps(progress.get("coverage", {})), "",
        "## Matched receiver comparison", "",
        "| Timing / config | Covered n | Top1 mean +/- seed SD | Top3 | Top5 | MRR | Macro F1 | Features | Parameters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in summaries.items():
        group = entry["common"]
        cells = [f"{100*group[m]['mean']:.3f} +/- {100*group[m]['seed_sd']:.3f}"
                 if m == "top1" else f"{group[m]['mean']:.4f}" for m in METRICS]
        lines.append(f"| {name} | {entry['coverage']} | " + " | ".join(cells)
                     + f" | {entry['feature_count']} | {entry['parameters']} |")
    historic = progress.get("historical_v2_common", {})
    for name, entries in historic.items():
        available = [v for v in entries.values() if "top1" in v]
        if available:
            lines.append(f"| historical_{name} (unchanged) | {available[0]['n']} | "
                         f"{100*np.mean([v['top1'] for v in available]):.3f} +/- "
                         f"{100*np.std([v['top1'] for v in available]):.3f} | "
                         + " | ".join(f"{np.mean([v[m] for v in available]):.4f}" for m in METRICS[1:])
                         + " | historical | historical |")
    lines += ["", "Full fallback, per-rank recall, prediction distributions, per-match and subgroup "
              "metrics are in the companion JSON. Shuffle controls permute forecast values/masks within "
              "covered examples of each validation match, keeping base states unchanged. No seed SD is "
              "a population confidence interval; there are only two held-out matches.", "",
              "## Research answers", ""]
    a, b, d, c = (summaries.get("legacy_" + v) for v in ("A", "B", "D", "C"))
    verdict = "INCONCLUSIVE"
    if a and d:
        gain = d["common"]["top1"]["mean"] - a["common"]["top1"]["mean"]
        lines.append(f"1. D minus matched A top1: **{100*gain:+.3f} percentage points**. "
                     "The old 64.30% score has a different model/full denominator; see separately "
                     "recomputed historical-v2 common-subset results.")
        historical_v2 = [v["top1"] for v in historic.get("v2", {}).values() if "top1" in v]
        if historical_v2:
            lines.append(f"Against unchanged v2 predictions rescored on the common subset: "
                         f"{100*(d['common']['top1']['mean']-np.mean(historical_v2)):+.3f} pp "
                         f"(historical matched top1 {100*np.mean(historical_v2):.3f}%). "
                         "This is a contextual reference; A/B/D share the new equal-budget selection protocol.")
        if b:
            lines.append(f"2. D minus CV B top1: "
                         f"{100*(d['common']['top1']['mean']-b['common']['top1']['mean']):+.3f} pp.")
        if c:
            lines.append(f"3. D minus independent C top1: "
                         f"{100*(d['common']['top1']['mean']-c['common']['top1']['mean']):+.3f} pp.")
        else:
            lines.append("3. Independent-motion comparison PENDING.")
        for v, explanation in (("E", "explicit forecast pressure"), ("F", "explicit forecast movement")):
            if "legacy_" + v in summaries:
                delta = d["common"]["top1"]["mean"] - summaries["legacy_" + v]["common"]["top1"]["mean"]
                lines.append(f"4. D minus ablation without {explanation}: {100*delta:+.3f} pp; "
                             "retraining comparison, not causal feature importance.")
        if gain <= 0:
            verdict = "NO MEASURED BENEFIT"
        elif b and c and all(d["common"]["top1"]["mean"] > x["common"]["top1"]["mean"] for x in (b, c)):
            # Conservative keep only if each matched seed and both matches show a benefit vs A/B.
            rr = progress["receiver_results"]
            consistent = all(
                dr["common"]["top1"] > ar["common"]["top1"] and dr["common"]["top1"] > br["common"]["top1"]
                and all(dr["per_match_common"][m]["top1"] > ar["per_match_common"][m]["top1"] for m in VALIDATION_MATCHES)
                for dr, ar, br in zip(rr["legacy_D"]["runs"], rr["legacy_A"]["runs"], rr["legacy_B"]["runs"])
            )
            if consistent and gain >= .01:
                verdict = "KEEP FORECASTING"
    else:
        lines += ["1--4. PENDING: final receiver comparisons have not completed."]
    if summaries:
        best_name = max((name for name in summaries if name.startswith("legacy_")),
                        key=lambda name: summaries[name]["common"]["top1"]["mean"], default=None)
        if best_name:
            rr = progress["receiver_results"][best_name]["runs"]
            lines.append(f"Best measured legacy configuration so far: {best_name}; "
                         f"mean ranker fit {np.mean([r['fit'].get('training_runtime_seconds', 0) for r in rr]):.1f} s, "
                         f"mean validation ranker inference {np.mean([r['inference_seconds'] for r in rr]):.4f} s "
                         "(excludes forecast training/inference, separately recorded).")
    lines += [
        f"5. Compute cost: {progress['elapsed_seconds']/3600:.2f} wall hours so far, "
        f"{len(completed_folds)} completed fit sets. Per-fold checkpoint/epoch/parameter provenance "
        "and per-run inference cost are saved; no padded runtime.",
        f"6. Verdict: **{verdict}**. A positive mean alone does not establish robust tactical value.",
        "7. Next-actor pseudo-label accuracy is not pass completion, intended receiver, pass feasibility "
        "or tactical value. Verified ball flight/contact, receiver control/intention, defender reach, "
        "uncertainty under missing observations and task-specific evaluation are still needed.",
        "", "## Historical matched-subset reference", "",
        "See `historical_v2_matched_subset.json` under the experiment directory. It rescored the "
        "unchanged existing full-v2 predictions on exactly the covered legacy rows, with no retraining.",
        "", "## Resume", "", "```powershell", progress["resume_command"], "```", "",
    ]
    if progress.get("blocker"):
        lines += ["Blocker: " + progress["blocker"], ""]
    write_json(root / "verdict.json", {"verdict": verdict, "status": progress["status"]})
    Path("results/receiver_forecast_integration_report.md").write_text("\n".join(lines), encoding="utf-8")
    if summaries:
        charts(root, summaries)


def charts(root, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = [(k, v) for k, v in summaries.items() if k.startswith("legacy")]
    if not values:
        return
    directory = root / "charts"
    directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([k for k, _ in values], [100*v["common"]["top1"]["mean"] for _, v in values],
           yerr=[100*v["common"]["top1"]["seed_sd"] for _, v in values])
    ax.set(ylabel="Top-1 percent on common covered legacy subset", title="Receiver forecast feature comparison")
    fig.tight_layout()
    fig.savefig(directory / "receiver_top1.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([k for k, _ in values], [v["common"]["mrr"]["mean"] for _, v in values],
           yerr=[v["common"]["mrr"]["seed_sd"] for _, v in values])
    ax.set(ylabel="MRR on common covered legacy subset", title="Receiver forecast ranking quality")
    fig.tight_layout()
    fig.savefig(directory / "receiver_mrr.png", dpi=150)
    plt.close(fig)


def run(root, base, args):
    progress_path = root / "progress.json"
    resume = (f'& "{sys.executable}" train\\run_receiver_forecast_integration.py '
              f'--output "{root.resolve()}" --config "{args.config.resolve()}"')
    protected_paths = [Path("data/train_pass_receiver.jsonl"), Path("data/val_pass_receiver.jsonl"),
                       Path("results/trajectory_forecasting_metrics.json"), Path("results/trajectory_forecasting_report.md"),
                       Path("results/trajectory_leakage_audit.md"), Path("results/receiver_forecast_integration_feasibility.md")]
    hashes = {str(p): file_digest(p) for p in protected_paths}
    cache_hashes = {
        str(p): file_digest(p) for p in sorted((Path(base["trajectory_root"]) / "cache").glob("*.npz"))
    }
    if progress_path.exists():
        progress = read_json(progress_path)
        if (progress["config"] != base or progress["source_hashes"] != source_hashes()
                or progress["protected"] != hashes or progress["input_cache_hashes"] != cache_hashes):
            raise ValueError("Resume source/config/frozen-input mismatch")
    else:
        import subprocess
        progress = {
            "status": "RUNNING", "started_unix": time.time(), "config": base,
            "source_hashes": source_hashes(), "protected": hashes,
            "input_cache_hashes": cache_hashes,
            "receiver_results": {}, "resume_command": resume,
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        }
    progress.update(status="RUNNING", pid=os.getpid())
    progress.pop("blocker", None)
    deadline = (float("inf") if base["max_hours"] is None else
                time.monotonic() + max(0, base["max_hours"]*3600 - (time.time() - progress["started_unix"])))

    def callback(stage):
        check_deadline(deadline)
        progress["stage"] = stage
        publish(root, progress)
        LOG.info("STAGE %s", stage)

    try:
        smoke_result = read_json(root / "smoke_result.json")
        if (smoke_result["status"] != "PASS" or smoke_result["source_hashes"] != source_hashes()
                or smoke_result["config_hash"] != config_hash(base)):
            raise ValueError("Final source smoke must pass before launch")
        train = read_receiver_rows("data/train_pass_receiver.jsonl")
        val = read_receiver_rows("data/val_pass_receiver.jsonl")
        validate_match_split(train, val)
        rows = train + val
        if (len(train), len(val)) != (4799, 1224):
            raise ValueError("Frozen receiver counts changed")
        for timing in base["timings"]:
            callback(f"adapter_{timing}")
            adapter = adapter_cache(root, base, rows, timing)
            match = np.asarray(adapter["matches"])
            progress.setdefault("coverage", {})[timing] = {
                "train": int(np.sum(adapter["covered"][:4799])),
                "validation": int(np.sum(adapter["covered"][4799:])),
                "train_total": 4799, "validation_total": 1224,
                "base_valid_train": int(np.sum(adapter["base_valid"][:4799])),
                "base_valid_validation": int(np.sum(adapter["base_valid"][4799:])),
                "reasons": adapter["reasons"],
            }
            if timing == "legacy":
                if (sum(adapter["covered"][:4799]), sum(adapter["covered"][4799:])) != (4794, 1222):
                    raise AssertionError("Adapter coverage differs from verified audit")
                progress["historical_v2_common"] = matched_historical(root, adapter)
            callback(f"features_cv_{timing}")
            cv_dev = feature_stage(root, base, adapter, "cv", "development", deadline, callback)
            cv_final = feature_stage(root, base, adapter, "cv", "final", deadline, callback)
            for variant in ("A", "B"):
                result = fit_variant(root, base, adapter, cv_dev, cv_final, variant, deadline, callback)
                progress["receiver_results"][f"{timing}_{variant}"] = result
                publish(root, progress)
            social_dev = feature_stage(root, base, adapter, "social", "development", deadline, callback)
            social_final = feature_stage(root, base, adapter, "social", "final", deadline, callback)
            for variant in ("D", "E", "F"):
                result = fit_variant(root, base, adapter, social_dev, social_final, variant, deadline, callback)
                progress["receiver_results"][f"{timing}_{variant}"] = result
                publish(root, progress)
            # C is optional and uses exactly matched exclusion sets/selection policy.
            independent_dev = feature_stage(root, base, adapter, "independent", "development", deadline, callback)
            independent_final = feature_stage(root, base, adapter, "independent", "final", deadline, callback)
            result = fit_variant(root, base, adapter, independent_dev, independent_final, "C", deadline, callback)
            progress["receiver_results"][f"{timing}_C"] = result
            publish(root, progress)
            del adapter, social_dev, social_final, independent_dev, independent_final
        if hashes != {str(p): file_digest(p) for p in protected_paths}:
            raise ValueError("Protected original files changed during experiment")
        if cache_hashes != {str(p): file_digest(p) for p in sorted((Path(base["trajectory_root"]) / "cache").glob("*.npz"))}:
            raise ValueError("Original trajectory caches changed during experiment")
        progress.update(status="COMPLETE", stage="complete", protected_unchanged=True)
        publish(root, progress)
        LOG.info("COMPLETE: %s", root)
    except BudgetExpired as error:
        progress.update(status="TIME_LIMIT", blocker=str(error))
        publish(root, progress)
        LOG.warning("Safe time-limit stop; resume state retained")
    except Exception as error:
        progress.update(status="FAILED", blocker=f"{type(error).__name__}: {error}")
        publish(root, progress)
        LOG.exception("Integration stopped; no leakage shortcuts")
        raise


def smoke(root, base):
    """Real adapter and GPU receiver recovery; no upstream full training."""
    seed_all(42)
    rows = read_receiver_rows("data/train_pass_receiver.jsonl")[:24]
    adapter = prepare_receiver_inputs(rows, Path(base["trajectory_root"]), Path(base["association_root"]))
    values, valid = infer_features(adapter, np.arange(len(rows)), None, torch.device("cuda"))
    again, other_valid = infer_features(adapter, np.arange(len(rows)), None, torch.device("cuda"))
    torch.testing.assert_close(values, again, rtol=0, atol=0)
    assert torch.equal(valid, other_valid)
    data = receiver_data(adapter, (values, valid), "D")
    config = receiver_config(base, 128, .001, epochs=2)
    config["batch_size"] = 8
    fitting, dev = np.arange(16), np.arange(16, 24)
    directory = root / "smoke" / config_hash({"conf": config, "source": source_hashes()})
    fit_receiver(data, fitting, dev, config, directory / "resumed", float("inf"), max_steps=1)
    fit_receiver(data, fitting, dev, config, directory / "resumed", float("inf"))
    fit_receiver(data, fitting, dev, config, directory / "reference", float("inf"))
    a = torch.load(directory / "resumed/final.pt", weights_only=True)
    b = torch.load(directory / "reference/final.pt", weights_only=True)
    delta = max(float((a[key] - b[key]).abs().max()) for key in a)
    if delta != 0:
        raise AssertionError(f"Receiver resume differs: {delta}")
    # Exercise shared forecast fit recovery on synthetic bounded tracks, not full data.
    from src.trajectory_dataset import make_window
    class SmokeDataset(list):
        match_names = [("117092", "1st")]
    p = Path(base["trajectory_root"]) / "cache/117092_1st.npz"
    with np.load(p) as blob:
        raw = {k: blob[k][:300] for k in blob.files}
    tiny = SmokeDataset()
    for frame in range(50, 50 + 13*8, 13):
        row = make_window(raw, frame)
        row.update(source_index=0, anchor_frame=frame)
        tiny.append(row)
    fc = forecast_config(base, "social", 1)
    fc["batch_size"] = 4
    fit_forecaster(tiny, tiny, fc, directory / "forecast", torch.device("cuda"), float("inf"), max_steps=1)
    fit_forecaster(tiny, tiny, fc, directory / "forecast", torch.device("cuda"), float("inf"))
    # Exercise the complete ranking/evaluation/fallback/shuffle path before expensive folds.
    all_train = read_receiver_rows("data/train_pass_receiver.jsonl")
    all_val = read_receiver_rows("data/val_pass_receiver.jsonl")
    selected_rows = (
        [r for r in all_train if r["match_id"] == BASE_TRAIN[0]][:12]
        + [r for r in all_train if r["match_id"] == DEVELOPMENT_MATCH][3:11]
        + [r for r in all_val if r["match_id"] == VALIDATION_MATCHES[0]][3:11]
        + [r for r in all_val if r["match_id"] == VALIDATION_MATCHES[1]][3:11]
    )
    small = prepare_receiver_inputs(selected_rows, Path(base["trajectory_root"]), Path(base["association_root"]))
    small_forecast = infer_features(small, np.arange(len(selected_rows)), None, torch.device("cuda"))
    smoke_base = copy.deepcopy(base)
    smoke_base["receiver"].update(widths=[32], learning_rates=[.001], max_epochs=2, patience=2)
    smoke_base["receiver_seeds"] = [42]
    for variant in ("A", "B", "D"):
        fit_variant(directory / "pipeline", smoke_base, small, small_forecast, small_forecast,
                    variant, float("inf"), lambda stage: LOG.info("SMOKE %s", stage))
    write_json(root / "smoke_result.json", {
        "status": "PASS", "source_hashes": source_hashes(), "gpu": torch.cuda.get_device_name(),
        "config_hash": config_hash(base),
        "receiver_recovery_parameter_delta": delta, "adapter_reproducibility": "PASS",
        "forecast_fit_resume": "PASS", "A_B_D_full_evaluation_and_shuffle_path": "PASS",
        "created_unix": time.time(),
    })
    LOG.info("INTEGRATION SMOKE PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/receiver_forecast_integration.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/receiver_forecast_integration_20261016"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    base, root = read_json(args.config), args.output
    if (base["max_hours"] is not None and not 0 < base["max_hours"] <= 12) \
            or base["cpu_threads"] > 4 or base["loader_workers"] > 2:
        raise ValueError("Resource cap violation")
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(root / "experiment.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    torch.cuda.set_per_process_memory_fraction(base["gpu_memory_fraction"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.smoke:
        smoke(root, base)
    else:
        import msvcrt
        with (root / "orchestrator.lock").open("a+b") as lock:
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            run(root, base, args)


if __name__ == "__main__":
    main()
