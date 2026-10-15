"""Select candidate ranker alpha/epoch on match 132877, then freeze three fits."""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.models.candidate_ranker_receiver import (
    FEATURE_NAMES, CandidateRankerReceiver, fit_statistics, prior_adjustment,
)
from src.receiver_tensor_dataset import (
    ReceiverTensorDataset, development_indices, read_receiver_rows,
    shuffle_coordinates, validate_match_split,
)
from train.train_entity_transformer_receiver import (
    evaluate, make_loader, save_predictions, set_seed, write_json,
)
from train.train_temporal_7class_mild_balance import git_metadata, sha256_file

METRICS = ("top1", "top3", "top5", "mrr", "mean_rank", "median_rank", "macro_f1")


def train_epoch(model, loader, optimizer, device, frequencies, alpha):
    model.train()
    total, count = 0.0, 0
    for states, targets in loader:
        states, targets = states.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = prior_adjustment(model(states), frequencies, alpha)
        loss = nn.functional.cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite corrected cross entropy")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item() * len(targets)
        count += len(targets)
    return total / count


def select_configuration(configs):
    return max(configs, key=lambda c: (c["best_development_mrr"], -c["alpha"], -c["selected_epoch"]))


def fit(states, targets, directory, config, device, dev=None):
    directory.mkdir()
    set_seed(config["seed"])
    stats = fit_statistics(states)
    model = CandidateRankerReceiver(stats).to(device)
    config["parameter_count"] = sum(p.numel() for p in model.parameters())
    counts = torch.bincount(targets, minlength=10)
    frequencies = counts.to(device).float() / len(targets)
    config["target_counts"] = counts.tolist()
    config["normalization"] = {k: v.tolist() for k, v in stats.items()}
    write_json(directory / "run_config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = make_loader(TensorDataset(states, targets), 64, config["seed"], True)
    dev_loader = make_loader(TensorDataset(*dev), 64, 42, False) if dev is not None else None
    history, best, stale = [], -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        row = {"epoch": epoch, "loss": train_epoch(
            model, loader, optimizer, device, frequencies, config["alpha"]
        )}
        if dev_loader is not None:
            metrics, _ = evaluate(model, dev_loader, device)
            row["development_mrr"] = metrics["mrr"]
            if metrics["mrr"] > best:
                best, stale = metrics["mrr"], 0
                config["selected_epoch"] = epoch
                config["best_development_mrr"] = best
                torch.save(model.state_dict(), directory / "best_model.pt")
            else:
                stale += 1
        history.append(row)
        print(f"{directory.name}: {row}", flush=True)
        if dev is not None and stale >= config["patience"]:
            break
    config["training_runtime_seconds"] = time.perf_counter() - started
    config["completed_epochs"] = len(history)
    config["final_train_loss"] = history[-1]["loss"]
    torch.save(model.state_dict(), directory / "final_model.pt")
    write_json(directory / "training_history.json", {"epochs": history})
    write_json(directory / "run_config.json", config)
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.max_epochs <= 100 or args.patience < 1:
        parser.error("max-epochs must be 1..100; patience must be positive")
    root = args.output_root
    paths = [root / f"candidate_ranker_receiver_dev_alpha{a}" for a in (0, 0.25, 0.5)]
    finals = [root / f"candidate_ranker_receiver_seed{s}" for s in (42, 7, 123)]
    summary_path = root / "candidate_ranker_receiver_summary.json"
    for path in (*paths, *finals, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing overwrite: {path}")
    root.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = Path("data") / "train_pass_receiver.jsonl"
    val_path = Path("data") / "val_pass_receiver.jsonl"
    hashes = {"train": sha256_file(train_path), "val": sha256_file(val_path)}
    rows = read_receiver_rows(train_path)
    if len(rows) != 4799:
        raise ValueError("Expected frozen 4799 training examples")
    dataset = ReceiverTensorDataset(rows)
    indices, dev_indices = development_indices(rows)
    common = {
        "git": git_metadata(), "dataset_hashes": hashes, "feature_names": FEATURE_NAMES,
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "batch_size": 64, "gradient_clip": 1.0, "patience": args.patience,
        "prior_rule": "subtract alpha*log(fitting-partition target frequency) during training only; raw logits at inference",
        "normalization_rule": "fit on six selection training matches; refit on all seven for final training",
        "development_match": "132877", "selection_seed": 42,
        "temporal_offsets_frames": [-10, -5, 0],
        "architecture": {
            "candidate_mlp": [len(FEATURE_NAMES) * 2, 256, 128],
            "scene_d_model": 128, "scene_layers": 3, "scene_heads": 4,
            "scene_feedforward": 256, "scoring_mlp": [512, 128, 1],
            "dropout": 0.1, "pooling": "mean over observed tokens across all frames",
        },
        "selection_rule": "maximum development MRR; ties prefer lower alpha then earlier epoch",
        "training_examples": len(indices), "development_examples": len(dev_indices),
    }
    started = time.perf_counter()
    selections = []
    for alpha, directory in zip((0.0, 0.25, 0.5), paths):
        model, config = fit(
            dataset.features[indices], dataset.targets[indices], directory,
            {**common, "seed": 42, "alpha": alpha, "epochs": args.max_epochs}, device,
            dev=(dataset.features[dev_indices], dataset.targets[dev_indices]),
        )
        selections.append(config)
        del model
    selected = select_configuration(selections)
    print(f"SELECTED alpha={selected['alpha']} epoch={selected['selected_epoch']}", flush=True)
    configs = []
    for seed, directory in zip((42, 7, 123), finals):
        model, config = fit(
            dataset.features, dataset.targets, directory,
            {**common, "seed": seed, "alpha": selected["alpha"],
             "epochs": selected["selected_epoch"], "selected_epoch": selected["selected_epoch"],
             "training_examples": len(rows)}, device,
        )
        configs.append(config)
        del model
    # Held-out prompts and targets are opened only after every final fit is fixed.
    val_rows = read_receiver_rows(val_path)
    validate_match_split(rows, val_rows)
    if len(val_rows) != 1224:
        raise ValueError("Expected frozen 1224 validation examples")
    val = ReceiverTensorDataset(val_rows)
    loader = make_loader(val, 64, 42, False)
    summary = {"selection": selections, "selected_alpha": selected["alpha"],
               "selected_epoch": selected["selected_epoch"], "runs": []}
    for config, directory in zip(configs, finals):
        model = CandidateRankerReceiver(fit_statistics(dataset.features)).to(device)
        model.load_state_dict(torch.load(directory / "final_model.pt", map_location=device, weights_only=True))
        metrics, records = evaluate(model, loader, device)
        config["validation_evaluations"] = 1
        config["validation_examples"] = len(val_rows)
        write_json(directory / "run_config.json", config)
        write_json(directory / "val_metrics.json", metrics)
        save_predictions(directory / "val_predictions.jsonl", val.ids, records)
        summary["runs"].append({"seed": config["seed"], "config": config, "metrics": metrics})
        if config["seed"] == 42:
            shuffled = shuffle_coordinates(val.features, 42)
            sm, sr = evaluate(model, make_loader(TensorDataset(shuffled, val.targets), 64, 42, False), device)
            ablation = {"metrics": sm, "change": {k: sm[k] - metrics[k] for k in ("top1", "top3", "mrr")}}
            write_json(directory / "shuffle_metrics.json", ablation)
            save_predictions(directory / "shuffle_predictions.jsonl", val.ids, sr)
            summary["shuffle"] = ablation
        print(f"FINAL seed={config['seed']} {metrics}", flush=True)
    summary["aggregate"] = {
        key: {"mean": statistics.mean(r["metrics"][key] for r in summary["runs"]),
              "std": statistics.pstdev(r["metrics"][key] for r in summary["runs"])}
        for key in METRICS
    }
    if hashes != {"train": sha256_file(train_path), "val": sha256_file(val_path)}:
        raise RuntimeError("Frozen dataset hashes changed")
    summary["total_runtime_seconds"] = time.perf_counter() - started
    write_json(summary_path, summary)
    print(f"COMPLETE {summary_path}", flush=True)


if __name__ == "__main__":
    main()
