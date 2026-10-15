"""Fresh LoRA training of the temporal PASS-receiver model (TEAMMATE_1..10).

Natural (unbalanced) training distribution, one epoch, loss on response tokens
only. Hyperparameters and LoRA configuration match the validated 7-class run.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_temporal_7class_mild_balance import (
    LORA_TARGET_MODULES,
    check_output_dir,
    git_metadata,
    read,
    sha256_file,
)

LABELS = [f"TEAMMATE_{i}" for i in range(1, 11)]
MODEL_INPUT_COLUMNS = {"input_ids", "attention_mask", "labels"}
EPOCHS = 1


def validate_rows(rows, name):
    if not rows:
        raise ValueError(f"{name} data is empty")
    unexpected = sorted({r["response"] for r in rows} - set(LABELS))
    if unexpected:
        raise ValueError(f"{name} has unexpected labels: {unexpected}")


def encode_example(tokenizer, row, max_length):
    """Tokenize prompt and response separately; loss only on response + EOS."""
    prompt_ids = tokenizer(
        row["prompt"],
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    response_ids = tokenizer(
        row["response"] + tokenizer.eos_token,
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, max_length - len(prompt_ids)),
    )["input_ids"]
    input_ids = (prompt_ids + response_ids)[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": ([-100] * len(prompt_ids) + response_ids)[:max_length],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/train_pass_receiver.jsonl")
    parser.add_argument("--val", default="data/val_pass_receiver.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Smoke-test only: stop after this many optimizer steps.")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    check_output_dir(output_path, args.resume_from_checkpoint)
    print(f"output directory: {output_path.resolve()}")
    print("resume status: " + (
        f"RESUMING from {args.resume_from_checkpoint}"
        if args.resume_from_checkpoint else "fresh run (no resume)"))

    train_rows = read(args.train)
    val_rows = read(args.val)
    validate_rows(train_rows, "train")
    validate_rows(val_rows, "validation")
    train_counts = Counter(r["response"] for r in train_rows)
    val_counts = Counter(r["response"] for r in val_rows)
    print(f"train examples: {len(train_rows)}  validation examples: {len(val_rows)}")
    print(f"train target counts: { {l: train_counts[l] for l in LABELS} }")
    print(f"optimizer steps per epoch: "
          f"{math.ceil(len(train_rows) / (args.batch_size * args.grad_accum))}")

    output_path.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "pass receiver (TEAMMATE_1..10), natural distribution, no balancing",
        "base_model": args.model,
        "start_adapter": None,
        "train_file": args.train,
        "val_file": args.val,
        "seed": args.seed,
        "data_seed": args.seed,
        "epochs": EPOCHS,
        "max_steps_override": args.max_steps,
        "learning_rate": args.lr,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_length": args.max_length,
        "loss": "response tokens + EOS only",
        "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
                 "target_modules": LORA_TARGET_MODULES},
        "dataset_sizes": {"train": len(train_rows), "validation": len(val_rows)},
        "train_target_counts": {l: train_counts[l] for l in LABELS},
        "validation_target_counts": {l: val_counts[l] for l in LABELS},
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "resume_from_checkpoint": args.resume_from_checkpoint,
    }
    config_path = output_path / "run_config.json"

    import peft
    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    git = git_metadata()
    config.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git["commit"],
        "git_working_tree_dirty": git["dirty"],
        "train_sha256": sha256_file(args.train),
        "val_sha256": sha256_file(args.val),
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda": torch.version.cuda or "unknown",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only",
        },
    })
    if args.resume_from_checkpoint and config_path.exists():
        config_path.replace(output_path / f"run_config.before_resume_{int(time.time())}.json")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = (torch.bfloat16 if use_bf16 else
             torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES))

    def tokenize(rows):
        dataset = Dataset.from_list(rows)
        tokenized = dataset.map(
            lambda row: encode_example(tokenizer, row, args.max_length),
            remove_columns=dataset.column_names)
        if set(tokenized.column_names) != MODEL_INPUT_COLUMNS:
            raise RuntimeError(f"Unexpected columns: {tokenized.column_names}")
        return tokenized

    train_dataset = tokenize(train_rows)
    val_dataset = tokenize(val_rows)

    class DataCollator:
        def __call__(self, features):
            size = max(len(f["input_ids"]) for f in features)

            def pad(values, value):
                return values + [value] * (size - len(values))

            return {
                "input_ids": torch.tensor([pad(f["input_ids"], tokenizer.pad_token_id) for f in features]),
                "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features]),
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
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                      eval_dataset=val_dataset, data_collator=DataCollator())
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    runtime = time.perf_counter() - started
    history = trainer.state.log_history
    last_train = next((e for e in reversed(history) if "loss" in e), {})
    last_eval = next((e for e in reversed(history) if "eval_loss" in e), {})
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(output_path)
    config["results"] = {
        "runtime_seconds": runtime,
        "train_metrics": result.metrics,
        "last_logged_train_loss": last_train.get("loss"),
        "final_eval_loss": last_eval.get("eval_loss"),
        "global_step": trainer.state.global_step,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"training runtime seconds: {runtime:.2f}")
    print(f"train metrics: {result.metrics}")
    print(f"validation metrics: {last_eval}")


if __name__ == "__main__":
    main()
