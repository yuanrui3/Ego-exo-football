import argparse
import json
import re
from collections import Counter
from pathlib import Path

LEGACY_LABELS = [
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
PRIMARY_LABELS = [
    "PASS",
    "HIGH_PASS",
    "DRIVE",
    "CROSS",
    "SHOT",
    "HEADER",
    "DEFENSIVE_INTERVENTION",
]
KNOWN_LABEL_SETS = (PRIMARY_LABELS, LEGACY_LABELS)
INVALID = "<invalid>"
MIN_NEW_TOKENS = 16


def read(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical_key(text):
    """Case-, space-, underscore- and hyphen-insensitive comparison key."""
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")


def parse_label_list(value):
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one label")
    keys = [canonical_key(label) for label in labels]
    if len(set(keys)) != len(keys):
        raise ValueError(f"--labels contains duplicate labels: {labels}")
    return labels


def resolve_labels(val_responses, explicit_labels=None):
    """Return the valid label list; raise if validation has any other label."""
    observed = set(val_responses)
    if explicit_labels is not None:
        unexpected = sorted(observed - set(explicit_labels))
        if unexpected:
            raise ValueError(
                f"Validation contains labels outside --labels: {unexpected}"
            )
        return list(explicit_labels)
    for label_set in KNOWN_LABEL_SETS:
        if observed <= set(label_set):
            return list(label_set)
    raise ValueError(
        "Validation labels do not match a known label set "
        f"(7-class: {PRIMARY_LABELS}; 12-class: {LEGACY_LABELS}). "
        f"Observed: {sorted(observed)}"
    )


def normalize_prediction(text, labels):
    """Exact canonical match of the whole generation; no substring matching."""
    lookup = {canonical_key(label): label for label in labels}
    return lookup.get(canonical_key(text), INVALID)


def compute_metrics(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    distribution = Counter(y_pred)
    return {
        "examples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "per_class_f1": {label: report[label]["f1-score"] for label in labels},
        "per_class_support": {label: int(report[label]["support"]) for label in labels},
        "invalid_count": distribution.get(INVALID, 0),
        "invalid_rate": distribution.get(INVALID, 0) / len(y_pred),
        "prediction_distribution": dict(distribution),
        "prediction_percent": {
            label: 100 * count / len(y_pred) for label, count in distribution.items()
        },
        "classes_predicted": sum(1 for label in labels if distribution.get(label, 0)),
    }


def max_new_tokens_arg(value):
    number = int(value)
    if number < MIN_NEW_TOKENS:
        raise argparse.ArgumentTypeError(
            f"--max-new-tokens must be at least {MIN_NEW_TOKENS} so long labels "
            "such as DEFENSIVE_INTERVENTION are not truncated"
        )
    return number


def generate(model, tokenizer, prompt, max_new_tokens):
    import torch

    batch = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )
    batch = {key: value.to(model.device) for key, value in batch.items()}
    with torch.inference_mode():
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, batch["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--val", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--predictions", default="predictions.jsonl")
    parser.add_argument(
        "--labels",
        type=parse_label_list,
        help="Comma-separated valid labels; inferred from the validation "
        "responses (7-class or legacy 12-class) when omitted.",
    )
    parser.add_argument("--max-new-tokens", type=max_new_tokens_arg, default=16)
    parser.add_argument(
        "--metrics",
        help="Optional metrics JSON path; defaults next to the predictions file.",
    )
    args = parser.parse_args()

    rows = read(args.val)
    labels = resolve_labels([row["response"] for row in rows], args.labels)
    print(f"valid labels ({len(labels)}): {labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    y_true, y_pred = [], []
    with open(args.predictions, "w", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            raw = generate(model, tokenizer, row["prompt"], args.max_new_tokens)
            prediction = normalize_prediction(raw, labels)
            y_true.append(row["response"])
            y_pred.append(prediction)
            output.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "true": row["response"],
                        "pred": prediction,
                        "raw": raw,
                    }
                )
                + "\n"
            )
            if index % 50 == 0:
                print(index, "/", len(rows))

    metrics = compute_metrics(y_true, y_pred, labels)
    print("Top-1 accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("Per-class F1:")
    for label in labels:
        print(
            f"  {label}: {metrics['per_class_f1'][label]:.4f} "
            f"(support {metrics['per_class_support'][label]})"
        )
    print(
        "Invalid:",
        metrics["invalid_count"],
        "/",
        metrics["examples"],
        f"({metrics['invalid_rate']:.2%})",
    )
    print("Prediction distribution (%):")
    for label, percent in sorted(
        metrics["prediction_percent"].items(), key=lambda item: -item[1]
    ):
        print(f"  {label}: {percent:.2f}")
    print("Classes predicted:", metrics["classes_predicted"], "/", len(labels))

    metrics_path = (
        Path(args.metrics)
        if args.metrics
        else Path(args.predictions).with_name(
            Path(args.predictions).stem + "_metrics.json"
        )
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Metrics written to", metrics_path)


if __name__ == "__main__":
    main()
