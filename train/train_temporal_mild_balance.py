import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


LABELS = [
    "Pass",
    "Drive",
    "Header",
    "High Pass",
    "Out",
    "Cross",
    "Throw In",
    "Shot",
    "Ball Player Block",
    "Player Successful Tackle",
    "Free Kick",
    "Goal",
]


def read(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sample_training_rows(rows, num_examples, seed):
    if not rows:
        raise ValueError("Training data is empty")
    counts = Counter(row["response"] for row in rows)
    unknown_labels = set(counts) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Unknown training labels: {sorted(unknown_labels)}")
    weights = [1 / math.sqrt(counts[row["response"]]) for row in rows]
    return random.Random(seed).choices(rows, weights=weights, k=num_examples), counts


def main():
    parser = argparse.ArgumentParser(
        description="Continue temporal LoRA training with mild class-aware sampling."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--start-adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--examples", type=int, default=7568)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    train_rows, class_counts = sample_training_rows(
        read(args.train), args.examples, args.seed
    )
    val_rows = read(args.val)
    sampled_counts = Counter(row["response"] for row in train_rows)
    print(f"original class counts: {dict(sorted(class_counts.items()))}")
    print(f"sampled class counts: {dict(sorted(sampled_counts.items()))}")
    print(f"effective examples: {len(train_rows)}")
    print(
        "optimizer steps: "
        f"{math.ceil(len(train_rows) / (args.batch_size * args.grad_accum))}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.start_adapter, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(
        base_model, args.start_adapter, is_trainable=True
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
            "labels": ([-100] * len(prompt_ids) + response_ids)[
                : args.max_length
            ],
        }

    train_dataset = Dataset.from_list(train_rows).map(
        encode, remove_columns=["id", "match_id", "prompt", "response"]
    )
    val_dataset = Dataset.from_list(val_rows).map(
        encode, remove_columns=["id", "match_id", "prompt", "response"]
    )

    class DataCollator:
        def __call__(self, features):
            max_size = max(len(feature["input_ids"]) for feature in features)

            def pad(values, value):
                return values + [value] * (max_size - len(values))

            return {
                "input_ids": torch.tensor(
                    [pad(feature["input_ids"], tokenizer.pad_token_id) for feature in features]
                ),
                "attention_mask": torch.tensor(
                    [pad(feature["attention_mask"], 0) for feature in features]
                ),
                "labels": torch.tensor(
                    [pad(feature["labels"], -100) for feature in features]
                ),
            }

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
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
    train_result = trainer.train()
    runtime_seconds = time.perf_counter() - started
    eval_metrics = next(
        (
            entry
            for entry in reversed(trainer.state.log_history)
            if "eval_loss" in entry
        ),
        {},
    )
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(output_path)
    print(f"training runtime seconds: {runtime_seconds:.2f}")
    print(f"train metrics: {train_result.metrics}")
    print(f"validation metrics: {eval_metrics}")


if __name__ == "__main__":
    main()
