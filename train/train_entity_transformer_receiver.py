"""Select epochs on training match 132877, then fit three fresh full-train seeds."""

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.evaluate_pass_receiver_model import LABELS, rank_candidates, ranking_metrics
from src.models.entity_transformer_receiver import ARCHITECTURE, EntityTransformerReceiver
from src.receiver_tensor_dataset import (
    DEVELOPMENT_MATCH, ENTITIES, FEATURE_NAMES, ReceiverTensorDataset,
    development_indices, read_receiver_rows, shuffle_coordinates, validate_match_split,
)
from train.train_temporal_7class_mild_balance import git_metadata, sha256_file

SEEDS = (42, 7, 123)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, count = 0.0, 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite training logits")
        loss = nn.functional.cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training cross-entropy")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item() * len(targets)
        count += len(targets)
    return total / count


def evaluate(model, loader, device) -> tuple[dict, list[dict]]:
    model.eval()
    records, true_labels, ranked_lists = [], [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for features, targets in loader:
            logits = model(features.to(device)).cpu()
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Non-finite evaluation logits; refusing results")
            loss_sum += nn.functional.cross_entropy(
                logits, targets, reduction="sum"
            ).item()
            for values, target in zip(logits.tolist(), targets.tolist()):
                scores = dict(zip(LABELS, values))
                ranked = rank_candidates(scores)
                true = LABELS[target]
                records.append({
                    "true_target": true, "predicted_target": ranked[0],
                    "ranked_targets": ranked, "logits": scores,
                    "true_rank": ranked.index(true) + 1,
                })
                true_labels.append(true)
                ranked_lists.append(ranked)
    metrics = ranking_metrics(true_labels, ranked_lists)
    metrics["loss"] = loss_sum / len(records)
    metrics["invalid_or_nonfinite_cases"] = 0
    metrics["prediction_distribution"] = {
        label: metrics["per_class"][label]["prediction_count"] for label in LABELS
    }
    return metrics, records


def save_predictions(path: Path, ids: list[str], records: list[dict]) -> None:
    if len(ids) != len(records):
        raise ValueError("Prediction/example count mismatch")
    with path.open("w", encoding="utf-8") as stream:
        for example_id, record in zip(ids, records):
            stream.write(json.dumps({"id": example_id, **record}, allow_nan=False) + "\n")


def select_best_epoch(history: list[dict]) -> int:
    if not history:
        raise ValueError("Empty development history")
    return max(history, key=lambda row: (row["development_mrr"], -row["epoch"]))["epoch"]


def train_run(dataset, config: dict, directory: Path, device, dev_dataset=None):
    directory.mkdir()
    write_json(directory / "run_config.json", config)
    set_seed(config["seed"])
    model = EntityTransformerReceiver().to(device)
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.0001
    )
    loader = make_loader(dataset, config["batch_size"], config["seed"], shuffle=True)
    dev_loader = (
        make_loader(dev_dataset, config["batch_size"], config["seed"], shuffle=False)
        if dev_dataset is not None else None
    )
    history, best_mrr = [], -1.0
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        loss = train_epoch(model, loader, optimizer, device)
        row = {"epoch": epoch, "train_loss": loss}
        if dev_loader is not None:
            metrics, _ = evaluate(model, dev_loader, device)
            row.update(development_mrr=metrics["mrr"], development_loss=metrics["loss"])
            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                torch.save(model.state_dict(), directory / "best_model.pt")
                config["selected_epoch"] = epoch
        history.append(row)
        write_json(directory / "training_history.json", {"epochs": history})
        print(f"{directory.name}: {json.dumps(row)}", flush=True)
    config["training_runtime_seconds"] = time.perf_counter() - started
    config["final_train_loss"] = history[-1]["train_loss"]
    torch.save(model.state_dict(), directory / "final_model.pt")
    if dev_dataset is None:
        # There is no final-validation model selection: final == selected-duration fit.
        torch.save(model.state_dict(), directory / "best_model.pt")
    else:
        config["selected_epoch"] = select_best_epoch(history)
    write_json(directory / "run_config.json", config)
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train_pass_receiver.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/val_pass_receiver.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.max_epochs <= 50 or args.batch_size < 1:
        parser.error("max-epochs must be 1..50 and batch-size must be positive")
    development_dir = args.output_root / "entity_transformer_receiver_development"
    final_dirs = [args.output_root / f"entity_transformer_receiver_seed{seed}" for seed in SEEDS]
    summary_path = args.output_root / "entity_transformer_receiver_summary.json"
    for path in (development_dir, *final_dirs, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment: {path}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = read_receiver_rows(args.train)
    if len(train_rows) != 4799:
        raise ValueError(f"Expected 4799 training examples, got {len(train_rows)}")
    train_indices, dev_indices = development_indices(train_rows)
    train_dataset = ReceiverTensorDataset(train_rows)
    train_hash, val_hash = sha256_file(args.train), sha256_file(args.val)
    git = git_metadata()
    common = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "temporal PASS receiver, frozen high-confidence pseudo-labels",
        "git_commit": git["commit"], "git_working_tree_dirty": git["dirty"],
        "train_sha256": train_hash, "val_sha256": val_hash,
        "train_file": str(args.train), "val_file": str(args.val),
        "architecture": ARCHITECTURE, "features": FEATURE_NAMES, "entity_slots": ENTITIES,
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0, "batch_size": args.batch_size,
        "sampling": "natural unbalanced distribution, shuffled without replacement",
        "deterministic_algorithms": True, "development_match": DEVELOPMENT_MATCH,
        "model_selection": "highest development MRR; earliest epoch breaks ties",
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu-only",
        },
    }
    selection_config = {
        **common, "seed": 42, "epochs": args.max_epochs, "phase": "epoch_selection",
        "training_examples": len(train_indices), "development_examples": len(dev_indices),
        "training_matches": sorted({train_rows[i]["match_id"] for i in train_indices}),
        "validation_used": False,
    }
    selection_model, selection_config = train_run(
        Subset(train_dataset, train_indices), selection_config, development_dir, device,
        dev_dataset=Subset(train_dataset, dev_indices),
    )
    selected_epoch = selection_config["selected_epoch"]
    del selection_model
    print(f"Selected epoch: {selected_epoch} (development match {DEVELOPMENT_MATCH})", flush=True)
    configs = []
    for seed, directory in zip(SEEDS, final_dirs):
        config = {
            **common, "seed": seed, "epochs": selected_epoch,
            "selected_epoch": selected_epoch, "phase": "full_training",
            "training_examples": len(train_rows), "validation_examples": 1224,
            "training_matches": sorted({row["match_id"] for row in train_rows}),
            "train_target_counts": dict(Counter(row["response"] for row in train_rows)),
            "selection_history": str(development_dir / "training_history.json"),
        }
        model, config = train_run(train_dataset, config, directory, device)
        configs.append(config)
        del model
    # The held-out prompts/targets are read only after all final fits are fixed.
    val_rows = read_receiver_rows(args.val)
    if len(val_rows) != 1224:
        raise ValueError(f"Expected 1224 validation examples, got {len(val_rows)}")
    validate_match_split(train_rows, val_rows)
    val_dataset = ReceiverTensorDataset(val_rows)
    val_loader = make_loader(val_dataset, args.batch_size, 42, shuffle=False)
    summary = {"selection": selection_config, "runs": [], "standard_deviation": "population"}
    for seed, config, directory in zip(SEEDS, configs, final_dirs):
        model = EntityTransformerReceiver().to(device)
        model.load_state_dict(torch.load(
            directory / "final_model.pt", map_location=device, weights_only=True
        ))
        started = time.perf_counter()
        metrics, records = evaluate(model, val_loader, device)
        config["evaluation_runtime_seconds"] = time.perf_counter() - started
        config["validation_evaluations"] = 1
        write_json(directory / "val_metrics.json", metrics)
        save_predictions(directory / "val_predictions.jsonl", val_dataset.ids, records)
        if seed == 42:
            shuffled = shuffle_coordinates(val_dataset.features, seed=42)
            shuffled_loader = make_loader(
                torch.utils.data.TensorDataset(shuffled, val_dataset.targets),
                args.batch_size, 42, shuffle=False,
            )
            shuffle_metrics, shuffle_records = evaluate(model, shuffled_loader, device)
            ablation = {
                "seed": 42,
                "method": "shuffle complete coordinate pairs among observed entities per state",
                "metrics": shuffle_metrics,
                "change_from_original": {
                    key: shuffle_metrics[key] - metrics[key] for key in ("top1", "top3", "mrr")
                },
            }
            write_json(directory / "shuffle_metrics.json", ablation)
            save_predictions(directory / "shuffle_predictions.jsonl", val_dataset.ids, shuffle_records)
            summary["shuffle_ablation"] = ablation
        config["validation_target_counts"] = dict(Counter(row["response"] for row in val_rows))
        write_json(directory / "run_config.json", config)
        summary["runs"].append({"seed": seed, "config": config, "metrics": metrics})
        print(f"FINAL seed {seed}: {json.dumps(metrics)}", flush=True)
        del model
    if sha256_file(args.train) != train_hash or sha256_file(args.val) != val_hash:
        raise RuntimeError("Frozen dataset changed during the experiment")
    summary["aggregate"] = {
        key: {
            "mean": statistics.mean(run["metrics"][key] for run in summary["runs"]),
            "std": statistics.pstdev(run["metrics"][key] for run in summary["runs"]),
        }
        for key in ("top1", "top3", "top5", "mrr", "macro_f1")
    }
    write_json(summary_path, summary)
    print("COMPLETE: " + str(summary_path), flush=True)


if __name__ == "__main__":
    main()
