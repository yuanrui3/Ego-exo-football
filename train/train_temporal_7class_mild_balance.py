"""Fresh 7-class temporal LoRA run with mild inverse-sqrt class sampling.

Trains from the base model (never from an existing adapter) for one full epoch
over `len(train)` examples sampled with replacement using
weight_i = 1 / sqrt(N_class(i)). Validation stays unbalanced.
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

LABELS = [
    "PASS",
    "HIGH_PASS",
    "DRIVE",
    "CROSS",
    "SHOT",
    "HEADER",
    "DEFENSIVE_INTERVENTION",
]
MODEL_INPUT_COLUMNS = {"input_ids", "attention_mask", "labels"}
EPOCHS = 1
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def read(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_labels(rows, name):
    if not rows:
        raise ValueError(f"{name} data is empty")
    observed = {row["response"] for row in rows}
    unexpected = sorted(observed - set(LABELS))
    missing = sorted(set(LABELS) - observed)
    if unexpected or missing:
        raise ValueError(
            f"{name} labels must be exactly {LABELS}; "
            f"unexpected={unexpected}, missing={missing}"
        )


def sample_training_rows(rows, num_examples, seed):
    """Sample with replacement; weight_i = 1 / sqrt(N_class(i))."""
    if not rows:
        raise ValueError("Training data is empty")
    counts = Counter(row["response"] for row in rows)
    weights = [1 / math.sqrt(counts[row["response"]]) for row in rows]
    sampled = random.Random(seed).choices(rows, weights=weights, k=num_examples)
    return sampled, counts


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata():
    """Best-effort Git info; never raises."""
    info = {"commit": "unknown", "dirty": "unknown"}
    try:
        repo = Path(__file__).resolve().parent
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        info["dirty"] = bool(status.strip())
    except Exception:
        pass
    return info


def check_output_dir(output_path, resume_from_checkpoint):
    """Refuse a non-empty output directory unless resuming."""
    if resume_from_checkpoint:
        checkpoint = Path(resume_from_checkpoint)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_path}. "
            "Use a new directory or pass --resume-from-checkpoint."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fresh 7-class temporal LoRA training with mild class sampling."
    )
    parser.add_argument("--train", default="data/train_temporal_7class.jsonl")
    parser.add_argument("--val", default="data/val_temporal_7class.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Smoke-test only: stop after this many optimizer steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Checkpoint every N optimizer steps (keeps the latest 2).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume from a checkpoint-N directory inside the output directory.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    check_output_dir(output_path, args.resume_from_checkpoint)
    print(f"output directory: {output_path.resolve()}")
    print(
        "resume status: "
        + (
            f"RESUMING from {args.resume_from_checkpoint}"
            if args.resume_from_checkpoint
            else "fresh run (no resume)"
        )
    )

    source_rows = read(args.train)
    val_rows = read(args.val)
    validate_labels(source_rows, "train")
    validate_labels(val_rows, "validation")
    sampled_rows, class_counts = sample_training_rows(
        source_rows, len(source_rows), args.seed
    )
    sampled_counts = Counter(row["response"] for row in sampled_rows)
    total_sampled = len(sampled_rows)
    print(f"original class counts: {dict(sorted(class_counts.items()))}")
    print(f"sampled class counts: {dict(sorted(sampled_counts.items()))}")
    print(
        "sampled class distribution: "
        + ", ".join(
            f"{label}={100 * sampled_counts[label] / total_sampled:.2f}%"
            for label in LABELS
        )
    )
    print(f"train examples (source/sampled): {len(source_rows)}/{total_sampled}")
    print(f"validation examples (unbalanced): {len(val_rows)}")
    steps_per_epoch = math.ceil(total_sampled / (args.batch_size * args.grad_accum))
    print(f"optimizer steps per epoch: {steps_per_epoch}")

    output_path.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model": args.model,
        "start_adapter": None,
        "train_file": args.train,
        "val_file": args.val,
        "seed": args.seed,
        "sampling": "with replacement, weight_i = 1/sqrt(N_class(i))",
        "epochs": EPOCHS,
        "max_steps_override": args.max_steps,
        "learning_rate": args.lr,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_length": args.max_length,
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": LORA_TARGET_MODULES,
        },
        "dataset_sizes": {
            "train_source": len(source_rows),
            "train_sampled_per_epoch": total_sampled,
            "validation": len(val_rows),
        },
        "train_class_counts": dict(sorted(class_counts.items())),
        "sampled_class_counts": dict(sorted(sampled_counts.items())),
        "validation_class_counts": dict(
            sorted(Counter(row["response"] for row in val_rows).items())
        ),
    }
    config_path = output_path / "run_config.json"

    import peft
    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    git = git_metadata()
    gpu = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"
    )
    config.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git["commit"],
            "git_working_tree_dirty": git["dirty"],
            "train_sha256": sha256_file(args.train),
            "val_sha256": sha256_file(args.val),
            "data_seed": args.seed,
            "save_steps": args.save_steps,
            "save_total_limit": 2,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "environment": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "cuda": torch.version.cuda or "unknown",
                "gpu": gpu,
            },
        }
    )
    if args.resume_from_checkpoint and config_path.exists():
        backup = output_path / f"run_config.before_resume_{int(time.time())}.json"
        config_path.replace(backup)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        ),
    )

    def encode(row):
        prompt_ids = tokenizer(
            row["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_length,
        )["input_ids"]
        response_ids = tokenizer(
            row["response"] + tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, args.max_length - len(prompt_ids)),
        )["input_ids"]
        input_ids = (prompt_ids + response_ids)[: args.max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": ([-100] * len(prompt_ids) + response_ids)[: args.max_length],
        }

    def tokenize(rows):
        dataset = Dataset.from_list(rows)
        tokenized = dataset.map(encode, remove_columns=dataset.column_names)
        if set(tokenized.column_names) != MODEL_INPUT_COLUMNS:
            raise RuntimeError(
                f"Unexpected columns after tokenization: {tokenized.column_names}"
            )
        return tokenized

    train_dataset = tokenize(sampled_rows)
    val_dataset = tokenize(val_rows)

    class DataCollator:
        def __call__(self, features):
            max_size = max(len(feature["input_ids"]) for feature in features)

            def pad(values, value):
                return values + [value] * (max_size - len(values))

            return {
                "input_ids": torch.tensor(
                    [pad(f["input_ids"], tokenizer.pad_token_id) for f in features]
                ),
                "attention_mask": torch.tensor(
                    [pad(f["attention_mask"], 0) for f in features]
                ),
                "labels": torch.tensor([pad(f["labels"], -100) for f in features]),
            }

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=EPOCHS,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="epoch",
        report_to="none",
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(),
    )
    started = time.perf_counter()
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    runtime_seconds = time.perf_counter() - started
    history = trainer.state.log_history
    last_train = next((e for e in reversed(history) if "loss" in e), {})
    last_eval = next((e for e in reversed(history) if "eval_loss" in e), {})
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(output_path)

    config["results"] = {
        "runtime_seconds": runtime_seconds,
        "train_metrics": train_result.metrics,
        "last_logged_train_loss": last_train.get("loss"),
        "final_eval_loss": last_eval.get("eval_loss"),
        "global_step": trainer.state.global_step,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"training runtime seconds: {runtime_seconds:.2f}")
    print(f"train metrics: {train_result.metrics}")
    print(f"validation metrics: {last_eval}")


if __name__ == "__main__":
    main()
