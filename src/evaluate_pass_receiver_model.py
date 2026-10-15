"""Ranking evaluation of the PASS-receiver model.

Every validation prompt is scored against all ten canonical candidates
TEAMMATE_1..TEAMMATE_10. The score is the mean log-probability of the candidate
tokens plus EOS given the prompt (prompt tokens are never scored). Future-event
metadata is never read.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

LABELS = [f"TEAMMATE_{i}" for i in range(1, 11)]


def read(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def rank_candidates(scores):
    """Labels ordered best to worst (ties broken by label order)."""
    return sorted(scores, key=lambda label: (-scores[label], LABELS.index(label)))


def true_rank(ranked, target):
    return ranked.index(target) + 1


def ranking_metrics(true_labels, ranked_lists, labels=LABELS):
    n = len(true_labels)
    if n == 0:
        raise ValueError("no examples")
    ranks = [true_rank(r, t) for r, t in zip(ranked_lists, true_labels)]
    predictions = [r[0] for r in ranked_lists]
    per_class = {}
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(true_labels, predictions) if t == p == label)
        fp = sum(1 for t, p in zip(true_labels, predictions) if p == label and t != label)
        fn = sum(1 for t, p in zip(true_labels, predictions) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = tp + fn
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "prediction_count": tp + fp,
        }
        if support:
            f1s.append(f1)
    return {
        "n": n,
        "top1": sum(r <= 1 for r in ranks) / n,
        "top3": sum(r <= 3 for r in ranks) / n,
        "top5": sum(r <= 5 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks) / n,
        "mean_rank": statistics.mean(ranks),
        "median_rank": statistics.median(ranks),
        "macro_f1": statistics.mean(f1s),
        "per_class": per_class,
    }


def candidate_token_ids(tokenizer, labels=LABELS):
    """Candidate tokens plus EOS, tokenized as in training."""
    return {
        label: tokenizer(
            label + tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]
        for label in labels
    }


def mean_logprob(logits, token_ids):
    """Mean log-probability of token_ids given per-position logits."""
    import torch

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    picked = logprobs[torch.arange(len(token_ids)), torch.tensor(token_ids)]
    return picked.mean().item()


def score_candidates_full(model, prompt_ids, candidates):
    """Reference scoring: full forward over prompt+candidate, candidate positions only."""
    import torch

    scores = {}
    device = next(model.parameters()).device
    for label, ids in candidates.items():
        sequence = torch.tensor([list(prompt_ids) + list(ids)], device=device)
        with torch.no_grad():
            logits = model(input_ids=sequence).logits[0]
        start = len(prompt_ids) - 1
        scores[label] = mean_logprob(logits[start : start + len(ids)], ids)
    return scores


def score_candidates_cached(model, prompt_ids, candidates):
    """Same scores as score_candidates_full, with the prompt encoded once."""
    import torch

    device = next(model.parameters()).device
    prompt = torch.tensor([list(prompt_ids)], device=device)
    with torch.no_grad():
        out = model(input_ids=prompt, use_cache=True)
        cache = out.past_key_values
        first_logits = out.logits[0, -1:]
        scores = {}
        for label, ids in candidates.items():
            if len(ids) > 1:
                tail = torch.tensor([ids[:-1]], device=device)
                logits = model(input_ids=tail, past_key_values=cache, use_cache=True).logits[0]
                cache.crop(len(prompt_ids))
                logits = torch.cat([first_logits, logits], dim=0)
            else:
                logits = first_logits
            scores[label] = mean_logprob(logits, ids)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--val", default="data/val_pass_receiver.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--generate-check", type=int, default=20,
                        help="Generate top-1 text for the first N examples as a parser check.")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(args.adapter)
    output = Path(args.output) if args.output else adapter / "val_predictions.jsonl"
    metrics_path = Path(args.metrics) if args.metrics else adapter / "val_metrics.json"

    rows = read(args.val)
    unexpected = sorted({r["response"] for r in rows} - set(LABELS))
    if unexpected:
        raise ValueError(f"validation has unexpected labels: {unexpected}")
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    candidates = candidate_token_ids(tokenizer)
    print("candidate token ids:", candidates)

    records, invalid = [], 0
    generated_ok = generated_total = 0
    for index, row in enumerate(rows):
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=True, truncation=True,
                               max_length=args.max_length)["input_ids"]
        scores = score_candidates_cached(model, prompt_ids, candidates)
        if not all(math.isfinite(v) for v in scores.values()):
            invalid += 1
        ranked = rank_candidates({k: (v if math.isfinite(v) else -math.inf) for k, v in scores.items()})
        record = {
            "id": row["id"],
            "true_target": row["response"],
            "predicted_target": ranked[0],
            "ranked_targets": ranked,
            "scores": scores,
            "true_rank": true_rank(ranked, row["response"]),
        }
        if index < args.generate_check:
            device = next(model.parameters()).device
            ids = torch.tensor([prompt_ids], device=device)
            with torch.no_grad():
                generated = model.generate(input_ids=ids, max_new_tokens=16, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
            record["generated_text"] = text
            generated_total += 1
            generated_ok += int(text == ranked[0])
        records.append(record)
        if (index + 1) % 100 == 0:
            print(f"scored {index + 1}/{len(rows)}", flush=True)

    metrics = ranking_metrics([r["true_target"] for r in records],
                              [r["ranked_targets"] for r in records])
    metrics["invalid_or_nonfinite_cases"] = invalid
    metrics["generation_check"] = {"examples": generated_total,
                                   "generated_equals_ranked_top1": generated_ok}
    with open(output, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))
    for label, m in metrics["per_class"].items():
        print(label, m)


if __name__ == "__main__":
    main()
