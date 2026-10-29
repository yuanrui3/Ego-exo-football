"""Train and evaluate one resumable compact action model from the v0.4 index."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from src.backbone.actions import ActionModelSession
from src.backbone.contracts import WorldInput
from src.models.partial_observation_student import PartialObservationStudent
from src.player_centric_dataset import ACTION_LABELS
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
FIT_MATCHES = {"117092", "117093", "118576", "118578", "128057", "132831"}
DEV_MATCHES = {"132877"}
EVAL_MATCHES = {"118577", "128058"}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _save_torch(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_split(index_path: Path, split: str) -> dict:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != "football.action-corpus.0.4" or index.get("status") != "COMPLETE":
        raise ValueError("Training requires the complete v0.4 indexed corpus")
    expected_matches = {
        "fitting": FIT_MATCHES,
        "development": DEV_MATCHES,
        "retrospective_evaluation": EVAL_MATCHES,
    }
    if split not in expected_matches:
        raise ValueError(f"Unknown data split: {split}")
    xs, masks, visibles, labels, sample_ids, matches = [], [], [], [], [], []
    seen = set()
    for shard in index["shards"]:
        if shard["split"] != split:
            continue
        match = str(shard["match"])
        if match not in expected_matches[split]:
            raise ValueError(f"Split contamination: {match} appears in {split}")
        folder = Path(shard["folder"])
        for filename, digest in shard["artifact_hashes"].items():
            path = folder / filename
            if not path.is_file() or file_digest(path) != digest:
                raise ValueError(f"Indexed corpus artifact is missing or changed: {path}")
        with np.load(folder / "student_inputs.npz", allow_pickle=False) as archive:
            shard_ids = archive["sample_ids"].astype(str).tolist()
            x = archive["x"].astype(np.float32)
            mask = archive["entity_mask"].astype(bool)
            visible = archive["visible"].astype(bool)
        target_rows = _read_jsonl(folder / "optional_targets.jsonl")
        target_by_id = {row["sample_id"]: row for row in target_rows}
        if len(target_by_id) != len(target_rows) or set(target_by_id) != set(shard_ids):
            raise ValueError(f"Student/target keyed join mismatch in {match}_{shard['half']}")
        if x.shape != (len(shard_ids), 11, 22, 10) or mask.shape != (len(shard_ids), 11, 22):
            raise ValueError(f"Unexpected model input dimensions in {match}_{shard['half']}")
        if visible.shape != mask.shape:
            raise ValueError(f"Visibility mask shape mismatch in {match}_{shard['half']}")
        shard_labels = []
        for sample_id in shard_ids:
            if sample_id in seen:
                raise ValueError(f"Duplicate sample ID in indexed training data: {sample_id}")
            seen.add(sample_id)
            row = target_by_id[sample_id]
            if row["source"]["split"] != split or row["source"]["match"] != match:
                raise ValueError(f"Target provenance mismatch for {sample_id}")
            label = row["action_class_index"]
            if not isinstance(label, int) or not 0 <= label < len(ACTION_LABELS):
                raise ValueError(f"Invalid action target for {sample_id}")
            shard_labels.append(label)
        if len(shard_ids):
            xs.append(x)
            masks.append(mask)
            visibles.append(visible)
            labels.extend(shard_labels)
            sample_ids.extend(shard_ids)
            matches.extend([match] * len(shard_ids))
    if not sample_ids:
        raise ValueError(f"No eligible rows in required split {split}")
    x = np.concatenate(xs)
    mask = np.concatenate(masks)
    visible = np.concatenate(visibles)
    y = np.asarray(labels, dtype=np.int64)
    match_array = np.asarray(matches, dtype=str)
    return {
        "x": x, "mask": mask, "visible": visible,
        "y": y, "sample_ids": np.asarray(sample_ids, dtype=str),
        "matches": match_array,
        "class_counts": {label: int(np.sum(y == i)) for i, label in enumerate(ACTION_LABELS)},
    }


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    confusion = np.zeros((len(ACTION_LABELS), len(ACTION_LABELS)), dtype=np.int64)
    for y, pred in zip(target, prediction, strict=True):
        confusion[int(y), int(pred)] += 1
    per_class = {}
    f1 = []
    for index, label in enumerate(ACTION_LABELS):
        tp = int(confusion[index, index])
        precision = tp / max(1, int(confusion[:, index].sum()))
        recall = tp / max(1, int(confusion[index, :].sum()))
        score = 2 * precision * recall / max(precision + recall, 1e-12)
        f1.append(score)
        per_class[label] = {
            "support": int(confusion[index].sum()),
            "precision": precision,
            "recall": recall,
            "f1": score,
        }
    return {
        "examples": int(len(target)),
        "coverage": 1.0 if len(target) else 0.0,
        "accuracy": float(np.trace(confusion) / max(1, int(confusion.sum()))),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


@torch.inference_mode()
def _forward(
    model, data: dict, device: torch.device, batch_size: int, *, mixed_precision: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, targets = [], []
    for start in range(0, len(data["y"]), batch_size):
        stop = min(len(data["y"]), start + batch_size)
        x = torch.from_numpy(data["x"][start:stop]).to(device)
        mask = torch.from_numpy(data["mask"][start:stop]).to(device)
        visible = torch.from_numpy(data["visible"][start:stop]).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and mixed_precision,
        ):
            output = model(x, mask, visible, memory_mode="memory")
        logits = output["action"].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite action logits during evaluation")
        scores.append(logits.cpu().numpy())
        targets.append(data["y"][start:stop])
    return np.concatenate(scores), np.concatenate(targets)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(index_path: Path, config_path: Path, output: Path, *, resume: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "football.action-training.0.4":
        raise ValueError("Unsupported training configuration")
    if set(config["fitting_matches"]) != FIT_MATCHES or set(config["development_matches"]) != DEV_MATCHES:
        raise ValueError("Training split must be six fitting matches and development match 132877")
    if set(config["retrospective_evaluation_matches"]) != EVAL_MATCHES:
        raise ValueError("Retrospective split mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    initialization = ROOT / config["initialization"]
    initialization_hash = file_digest(initialization)
    if initialization_hash != config["initialization_sha256"]:
        raise ValueError("Initialization checkpoint hash differs from the declared six-match development checkpoint")
    initial_config = json.loads((initialization.parent / "config.json").read_text(encoding="utf-8"))
    if set(initial_config["fitting_matches"]) != FIT_MATCHES:
        raise ValueError("Historical initialization includes a non-fitting match")
    if initial_config.get("variant") != "S4" or initial_config.get("fov_index") != 1:
        raise ValueError("Historical initialization does not match the supported S4 120-degree contract")
    model_config = config["model"]
    if (model_config["width"] != 96 or model_config["layers"] != 2
            or model_config["heads"] != 4 or model_config["dropout"] != 0.1):
        raise ValueError("This bounded run is fixed to the existing S4 architecture")

    training = load_split(index_path, "fitting")
    development = load_split(index_path, "development")
    if not set(training["matches"]) <= FIT_MATCHES or not set(development["matches"]) <= DEV_MATCHES:
        raise ValueError("Indexed corpus did not resolve to the required training/development matches")
    eval_matches_in_index = {
        str(shard["match"]) for shard in index["shards"]
        if shard["split"] == "retrospective_evaluation"
    }
    if eval_matches_in_index != EVAL_MATCHES:
        raise ValueError("Indexed corpus retrospective split does not match the declared held-out matches")

    opt = config["optimization"]
    seed = int(opt["seed"])
    _seed_all(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartialObservationStudent(
        width=int(model_config["width"]), layers=int(model_config["layers"]),
        heads=int(model_config["heads"]), dropout=float(model_config["dropout"]),
        belief=True,
    ).to(device)
    initial_state = torch.load(initialization, map_location=device, weights_only=True)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt["learning_rate"]),
        weight_decay=float(opt["weight_decay"]),
    )
    config_sha = file_digest(config_path)
    index_sha = file_digest(index_path)
    fingerprint = json.dumps({
        "config_sha256": config_sha,
        "index_sha256": index_sha,
        "initialization_sha256": initialization_hash,
        "training_samples": len(training["y"]),
        "development_samples": len(development["y"]),
    }, sort_keys=True)
    state_path = output / "last.pt"
    best_path = output / "best.pt"
    best_macro_f1 = -1.0
    best_epoch = 0
    next_epoch = 0
    stale = 0
    history = []
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"Cannot resume without a safe epoch checkpoint: {state_path}")
        saved = torch.load(state_path, map_location=device, weights_only=False)
        if saved.get("fingerprint") != fingerprint:
            raise ValueError("Training resume fingerprint mismatch; refusing to reuse stale optimizer/RNG state")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        next_epoch = int(saved["next_epoch"])
        best_macro_f1 = float(saved["best_macro_f1"])
        best_epoch = int(saved["best_epoch"])
        stale = int(saved["stale"])
        history = list(saved["history"])
        _restore_rng(saved["rng"])
    else:
        if state_path.exists():
            raise FileExistsError("Training output already has a checkpoint; pass --resume or use a new output directory")

    batch_size = int(opt["batch_size"])
    max_epochs = int(opt["max_epochs"])
    patience = int(opt["patience"])
    label_counts = Counter(training["y"].tolist())
    write_json(output / "run_config.json", {
        **config,
        "resolved": {
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "configuration_sha256": config_sha,
            "training_index_path": str(index_path.resolve()),
            "training_index_sha256": index_sha,
            "initialization_path": str(initialization.resolve()),
            "initialization_sha256": initialization_hash,
            "initialization_provenance": {
                "variant": initial_config["variant"],
                "fitting_matches": initial_config["fitting_matches"],
                "selected_by_development": "132877, width 96, epoch 17",
                "checkpoint_path": str(initialization.resolve()),
            },
            "training_examples": len(training["y"]),
            "development_examples": len(development["y"]),
            "training_class_counts": {label: int(label_counts.get(i, 0))
                                      for i, label in enumerate(ACTION_LABELS)},
            "teacher_scores": "ABSENT",
            "receiver_and_occupancy_losses": "DISABLED",
            "evaluation_protocol": "Development-only checkpoint selection; retrospective evaluation not run during fitting.",
        },
    })
    while next_epoch < max_epochs and stale < patience:
        epoch = next_epoch
        order = np.random.default_rng(seed + epoch).permutation(len(training["y"]))
        model.train()
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            x = torch.from_numpy(training["x"][indices]).to(device)
            mask = torch.from_numpy(training["mask"][indices]).to(device)
            visible = torch.from_numpy(training["visible"][indices]).to(device)
            y = torch.from_numpy(training["y"][indices]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and bool(opt["mixed_precision"]),
            ):
                result = model(x, mask, visible, memory_mode="memory")
                loss = F.cross_entropy(result["action"].float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite action training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(opt["gradient_clip_norm"]), error_if_nonfinite=True,
            )
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            seen += len(indices)

        dev_logits, dev_targets = _forward(model, development, device, batch_size)
        dev_metrics = _metrics(dev_targets, dev_logits.argmax(axis=-1))
        train_loss = loss_sum / max(1, seen)
        improved = dev_metrics["macro_f1"] > best_macro_f1 + float(
            opt["minimum_development_macro_f1_delta"]
        )
        if improved:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch + 1
            stale = 0
            _save_torch(best_path, {"model": model.state_dict(), "epoch": best_epoch})
        else:
            stale += 1
        history.append({
            "epoch": epoch + 1,
            "training_loss": train_loss,
            "development": dev_metrics,
            "best_epoch": best_epoch,
            "stale_epochs": stale,
        })
        next_epoch = epoch + 1
        _save_torch(state_path, {
            "schema_version": "football.action-train-state.0.4",
            "fingerprint": fingerprint,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": _rng_state(),
            "next_epoch": next_epoch,
            "best_macro_f1": best_macro_f1,
            "best_epoch": best_epoch,
            "stale": stale,
            "history": history,
        })
        write_json(output / "progress.json", {
            "status": "RUNNING",
            "completed_epochs": next_epoch,
            "maximum_epochs": max_epochs,
            "best_epoch": best_epoch,
            "best_development_macro_f1": best_macro_f1,
            "stale_epochs": stale,
            "device": str(device),
        })
        print(
            f"epoch {next_epoch}/{max_epochs}: loss={train_loss:.5f}, "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}, best={best_macro_f1:.4f}",
            flush=True,
        )
    if best_epoch == 0 or not best_path.is_file():
        raise RuntimeError("No development-selected checkpoint was produced")
    best_record = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_record["model"], strict=True)
    model.eval()
    model_config_record = {
        "variant": "S4",
        "fov_index": 1,
        "fixed_epochs": best_epoch,
        "config": {
            "width": int(model_config["width"]),
            "layers": int(model_config["layers"]),
            "heads": int(model_config["heads"]),
            "dropout": float(model_config["dropout"]),
            "learning_rate": float(opt["learning_rate"]),
            "weight_decay": float(opt["weight_decay"]),
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "distill_weight": 0.0,
            "belief_weight": 0.0,
            "mixed_precision": bool(opt["mixed_precision"]),
            "seed": seed,
        },
        "fitting_matches": sorted(FIT_MATCHES),
        "initialization_checkpoint": str(initialization.resolve()),
        "initialization_sha256": initialization_hash,
        "development_matches": sorted(DEV_MATCHES),
        "retrospective_evaluation_matches": sorted(EVAL_MATCHES),
        "classes": list(ACTION_LABELS),
        "loss_scope": "Action-only CE; receiver and occupancy heads were not optimized; teacher distillation disabled.",
    }
    write_json(model_dir / "config.json", model_config_record)
    # ActionModelSession consumes a state_dict directly, as in the original final checkpoints.
    _save_torch(model_dir / "final.pt", model.state_dict())
    train_metrics = _metrics(training["y"], _forward(model, training, device, batch_size)[0].argmax(-1))
    dev_logits, dev_y = _forward(model, development, device, batch_size)
    final_dev_metrics = _metrics(dev_y, dev_logits.argmax(-1))
    report = {
        "schema_version": "football.action-training-result.0.4",
        "status": "COMPLETE",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "training_examples": len(training["y"]),
        "development_examples": len(development["y"]),
        "training_class_counts": training["class_counts"],
        "development_class_counts": development["class_counts"],
        "completed_epochs": next_epoch,
        "best_epoch": best_epoch,
        "best_development_macro_f1": best_macro_f1,
        "training_diagnostic": train_metrics,
        "development_diagnostic": final_dev_metrics,
        "history": history,
        "checkpoint_path": str((model_dir / "final.pt").resolve()),
        "checkpoint_sha256": file_digest(model_dir / "final.pt"),
        "optimizer_rng_resume_checkpoint": str(state_path.resolve()),
        "training_index_sha256": index_sha,
        "fitting_matches": sorted(FIT_MATCHES),
        "development_matches": sorted(DEV_MATCHES),
        "retrospective_evaluation_matches": sorted(EVAL_MATCHES),
        "evaluation_not_run": True,
        "initialization": model_config_record,
    }
    write_json(output / "training_result.json", report)
    write_json(output / "progress.json", {"status": "COMPLETE", "best_epoch": best_epoch})
    return report


def evaluate(index_path: Path, checkpoint: Path, output: Path) -> dict:
    result = json.loads((checkpoint.parent.parent / "training_result.json").read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETE":
        raise ValueError("Evaluation requires a complete selected training run")
    if result["checkpoint_sha256"] != file_digest(checkpoint):
        raise ValueError("Selected checkpoint hash mismatch")
    data = load_split(index_path, "retrospective_evaluation")
    if set(data["matches"]) != EVAL_MATCHES:
        raise ValueError("Retrospective evaluation must include both configured matches")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_session = ActionModelSession(checkpoint, device=device)
    model = action_session.model
    direct_logits, direct_targets = _forward(
        model, data, torch.device(device), 256, mixed_precision=False,
    )
    direct_scores = direct_logits.astype(np.float64).tolist()
    direct_prediction = direct_logits.argmax(-1)
    wrapper_scores = []
    for index in range(len(data["y"])):
        record = data["x"][index].copy()
        record[..., 6] *= 5.0 / 8.0
        world_input = WorldInput(
            x=record,
            mask=data["mask"][index],
            actor_xy=np.zeros(2, np.float32),
            last_xy=np.zeros((22, 2), np.float32),
            identities={},
        )
        inference = action_session.predict(world_input)
        if inference.status != "supported" or inference.scores is None:
            raise RuntimeError(f"Reloaded action adapter rejected evaluation row {data['sample_ids'][index]}")
        wrapper_scores.append([inference.scores[label] for label in ACTION_LABELS])
    wrapper_scores_array = np.asarray(wrapper_scores)
    direct_probabilities = torch.from_numpy(np.asarray(direct_scores, dtype=np.float32)).softmax(-1).numpy()
    np.testing.assert_allclose(wrapper_scores_array, direct_probabilities, atol=1e-6, rtol=1e-5)
    metrics = _metrics(direct_targets, direct_prediction)
    per_match = {}
    for match in sorted(EVAL_MATCHES):
        indices = np.flatnonzero(data["matches"] == match)
        per_match[match] = _metrics(direct_targets[indices], direct_prediction[indices])
        per_match[match]["coverage"] = float(len(indices) / max(1, np.sum(data["matches"] == match)))
    evaluation_report = {
        "schema_version": "football.action-evaluation.0.4",
        "status": "COMPLETE",
        "evaluation_split": "retrospective_evaluation",
        "matches": sorted(EVAL_MATCHES),
        "matches_used_in_earlier_work": True,
        "not_used_for_tuning": True,
        "examples": len(data["y"]),
        "coverage": 1.0,
        "metrics": metrics,
        "per_match": per_match,
        "reloaded_action_model": action_session.provenance,
        "reloaded_checkpoint_prediction_parity": {
            "status": "PASS",
            "examples": len(data["y"]),
            "maximum_absolute_probability_difference": float(
                np.max(np.abs(wrapper_scores_array - direct_probabilities))
            ),
        },
        "interpretation": (
            "Diagnostic retrospective results only. Both matches appeared in earlier research work; "
            "no claim of a newly unseen test set and no tuning after this evaluation."
        ),
    }
    write_json(output, evaluation_report)
    return evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    index = args.index if args.index.is_absolute() else ROOT / args.index
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.evaluate:
        checkpoint = args.checkpoint or output / "model" / "final.pt"
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        report = evaluate(index.resolve(), checkpoint.resolve(), output / "retrospective_evaluation.json")
    else:
        if args.config is None:
            parser.error("--config is required unless --evaluate is specified")
        config = args.config if args.config.is_absolute() else ROOT / args.config
        report = train(index.resolve(), config.resolve(), output.resolve(), resume=args.resume)
    print(json.dumps({
        "status": report["status"],
        "device": report.get("device"),
        "examples": report.get("training_examples", report.get("examples")),
        "checkpoint": report.get("checkpoint_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
