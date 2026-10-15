"""Build a PASS-only receiver dataset (temporal state -> TEAMMATE_i).

Targets are HIGH-confidence pseudo-labels from the receiver-audit method
(src/audit_pass_receivers.py). The future event only builds the target; the
prompt is the same anonymous temporal state used by the action dataset
(t-10, t-5, t frames) with a receiver instruction. Reconstructed ball
coordinates are never used. Existing datasets are never modified.
"""

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_pass_receivers import audit_row, load_decisions, load_events
from make_temporal_sft_dataset import (
    HISTORY_OFFSETS,
    InvalidTemporalDecision,
    find_snapshot,
    prepare_entities,
    serialize_temporal_state,
    stream_states_for_frames,
)

TRAIN_MATCHES = ("117092", "117093", "118576", "118578", "128057", "132831", "132877")
VAL_MATCHES = ("118577", "128058")
MAX_TEAMMATES = 10
LABELS = [f"TEAMMATE_{i}" for i in range(1, MAX_TEAMMATES + 1)]
SYSTEM = (
    "You are a football tactical decision model. Given the temporal football "
    "state, predict which anonymous teammate receives the actor's pass. Return "
    "exactly one label from TEAMMATE_1 through TEAMMATE_10 and no explanation."
)
PROMPT_LINE = re.compile(
    r"^(PERIOD \d|STATE_T_MINUS_0\.4S|STATE_T_MINUS_0\.2S|STATE_T|"
    r"(ACTOR|TEAMMATE_\d+|OPPONENT_\d+) team=\w+ role=\w+ "
    r"x=(NA|-?\d+\.\d\d) y=(NA|-?\d+\.\d\d)|)$"
)
COORD = re.compile(
    r"^(ACTOR|TEAMMATE_\d+|OPPONENT_\d+) team=\S+ role=\S+ x=(\S+) y=(\S+)$"
)


def split_for_match(match_id):
    match_id = str(match_id)
    if match_id in TRAIN_MATCHES:
        return "train"
    if match_id in VAL_MATCHES:
        return "val"
    return None


def classify_pass(decision, events, decisions_index):
    """Audit row for one Pass decision (no ball data is used)."""
    return audit_row(decision, events, decisions_index, {})


def is_high_confidence(row):
    return (
        row["source_action"] == "PASS"
        and row["confidence"] == "HIGH"
        and bool(row["receiver_id"])
        and row["gap_frames"] != ""
        and int(row["gap_frames"]) <= 50
    )


def build_receiver_example(record, row, snapshots):
    """Return a receiver example, or raise InvalidTemporalDecision."""
    current = snapshots.get(0)
    if current is None:
        raise InvalidTemporalDecision("event-time state is unavailable")
    actor_team, entities = prepare_entities(record, current[1])
    formatted = {0: current[1]}
    for offset in (-10, -5):
        if snapshots.get(offset) is None:
            raise InvalidTemporalDecision("historical state is unavailable")
        formatted[offset] = snapshots[offset][1]
    target = next(
        (
            e["label"]
            for e in entities
            if e["key"] == str(row["receiver_id"])
            and e["label"].startswith("TEAMMATE_")
        ),
        None,
    )
    if target is None:
        raise InvalidTemporalDecision("receiver is not a teammate in the t-state")
    if int(target.split("_")[1]) > MAX_TEAMMATES:
        raise InvalidTemporalDecision("receiver label beyond TEAMMATE_10")
    state = serialize_temporal_state(
        record.get("period"), actor_team, entities, formatted
    )
    return {
        "id": record["decision_id"],
        "match_id": str(record["match_id"]),
        "prompt": f"<|system|>\n{SYSTEM}\n<|user|>\n{state}\n<|assistant|>\n",
        "response": target,
        "decision_id": record["decision_id"],
        "source_action": "PASS",
        "passer_id": str(record["actor_id"]),
        "receiver_id": str(row["receiver_id"]),
        "receiver_confidence": row["confidence"],
        "gap_frames": int(row["gap_frames"]),
        "gap_s": float(row["gap_s"]),
        "next_action": row["next_action"],
    }


def prompt_user_state(prompt):
    return prompt.split("<|user|>\n", 1)[1].rsplit("\n<|assistant|>", 1)[0]


def validate_prompt(prompt):
    """Return a list of problems; empty means only past/present anonymous state."""
    problems = []
    state = prompt_user_state(prompt)
    for line in state.split("\n"):
        if not PROMPT_LINE.match(line):
            problems.append(f"unexpected line: {line!r}")
    for header in ("STATE_T_MINUS_0.4S", "STATE_T_MINUS_0.2S", "STATE_T"):
        if state.count(header) != 1 and header != "STATE_T":
            problems.append(f"header count {header}")
    if re.search(r"STATE_T_PLUS|FUTURE|RECEIVER|player_id|jersey", prompt):
        problems.append("forbidden token")
    return problems


def parse_blocks(prompt):
    """Return {header: {label: (x, y) or None}} from the prompt state."""
    blocks, header = {}, None
    for line in prompt_user_state(prompt).split("\n"):
        if line.startswith("STATE_T"):
            header = line
            blocks[header] = {}
        elif header and COORD.match(line):
            label, x, y = COORD.match(line).groups()
            blocks[header][label] = (
                None if "NA" in (x, y) else (float(x), float(y))
            )
    return blocks


def rank_by_distance(prompt):
    """Teammate labels ordered by distance to the actor at STATE_T."""
    now = parse_blocks(prompt)["STATE_T"]
    actor = now["ACTOR"]
    items = []
    for index, (label, xy) in enumerate(now.items()):
        if label.startswith("TEAMMATE_") and xy is not None:
            items.append((math.hypot(xy[0] - actor[0], xy[1] - actor[1]), index, label))
    return [label for _, _, label in sorted(items)]


def max_identity_jump(prompt):
    """Largest displacement of one entity between t-10 and t (metres)."""
    blocks = parse_blocks(prompt)
    first, last = blocks["STATE_T_MINUS_0.4S"], blocks["STATE_T"]
    jumps = [
        math.hypot(last[k][0] - first[k][0], last[k][1] - first[k][1])
        for k in last
        if last[k] and first.get(k)
    ]
    return max(jumps) if jumps else 0.0


def geometric_metrics(examples, seed=42):
    ranks, n_teammates = [], []
    for ex in examples:
        order = rank_by_distance(ex["prompt"])
        n_teammates.append(len(order))
        ranks.append(order.index(ex["response"]) + 1)
    n = len(ranks)
    rng = random.Random(seed)
    random_hits = sum(
        rng.randrange(1, k + 1) == r for r, k in zip(ranks, n_teammates)
    )
    return {
        "n": n,
        "top1": sum(r <= 1 for r in ranks) / n,
        "top3": sum(r <= 3 for r in ranks) / n,
        "top5": sum(r <= 5 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks) / n,
        "mean_rank": statistics.mean(ranks),
        "median_rank": statistics.median(ranks),
        "random_uniform_accuracy_seed42": random_hits / n,
        "random_uniform_expected": sum(1 / k for k in n_teammates) / n,
        "rank_matches_label_rate": sum(
            r == int(ex["response"].split("_")[1]) for r, ex in zip(ranks, examples)
        )
        / n,
    }


def target_distribution(examples):
    counts = Counter(ex["response"] for ex in examples)
    n = len(examples)
    return {
        "counts": {label: counts.get(label, 0) for label in LABELS},
        "distinct_labels": sum(1 for v in counts.values() if v),
        "teammate_1_pct": 100 * counts.get("TEAMMATE_1", 0) / n,
        "top3_pct": 100 * sum(counts.get(l, 0) for l in LABELS[:3]) / n,
    }


def build(data_dir, log=print):
    decisions = load_decisions(data_dir)
    events = load_events(data_dir)
    by_half = defaultdict(list)
    for key, record in decisions.items():
        if split_for_match(key[0]) and record["action"]["label"] == "Pass":
            by_half[(key[0], key[1])].append(record)

    output = {"train": [], "val": []}
    excluded = {"train": Counter(), "val": Counter()}
    totals = {"train": 0, "val": 0}
    for (match_id, period), records in sorted(by_half.items()):
        split = split_for_match(match_id)
        totals[split] += len(records)
        eligible = []
        for record in records:
            row = classify_pass(record, events, decisions)
            if is_high_confidence(row):
                eligible.append((record, row))
            else:
                excluded[split][row["reason"] or "not_high_confidence"] += 1
        gsr_path = Path(data_dir) / f"{match_id}_{'1st' if period == 1 else '2nd'}.json"
        targets = {int(r["frame"]) + o for r, _ in eligible for o in HISTORY_OFFSETS}
        states = stream_states_for_frames(gsr_path, targets)
        for record, row in eligible:
            frame = int(record["frame"])
            snapshots = {}
            for offset in HISTORY_OFFSETS:
                found, players = find_snapshot(frame + offset, states)
                if players is not None:
                    snapshots[offset] = (found, players)
            if len(snapshots) != len(HISTORY_OFFSETS):
                excluded[split]["history_unavailable"] += 1
                continue
            try:
                example = build_receiver_example(record, row, snapshots)
            except InvalidTemporalDecision as error:
                excluded[split][f"invalid: {error}"] += 1
                continue
            output[split].append(example)
        log(f"{match_id} half {period}: kept so far "
            f"train={len(output['train'])} val={len(output['val'])}")
    return output, excluded, totals


def write_jsonl(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def report_markdown(summary):
    lines = ["# PASS-receiver dataset report", ""]
    lines += [
        "Generated by `src/build_pass_receiver_dataset.py`. Labels are HIGH-confidence "
        "pseudo-labels (first subsequent same-team direct action by a different player "
        "within 2 s, no intervening opponent/restart/outcome/block/tackle). They are "
        "circumstantial, not ground truth. Reconstructed ball data is not used.",
        "",
        "## Sizes",
        "| split | PASS decisions | examples | excluded |",
        "|---|---|---|---|",
    ]
    for split in ("train", "val"):
        s = summary[split]
        lines.append(f"| {split} | {s['pass_total']} | {s['examples']} | {s['excluded_total']} |")
    for split in ("train", "val"):
        s = summary[split]
        lines += ["", f"## {split}: exclusion reasons"]
        lines += [f"- {k}: {v}" for k, v in s["excluded"].items()]
        d = s["targets"]
        lines += ["", f"### {split}: target counts",
                  "| label | count |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in d["counts"].items()]
        lines += [
            f"- distinct labels: {d['distinct_labels']}",
            f"- TEAMMATE_1: {d['teammate_1_pct']:.2f}%",
            f"- top-3 ranks: {d['top3_pct']:.2f}%",
            f"- max identity jump t-10 -> t (m): median {s['jump_median']:.2f}, "
            f"max {s['jump_max']:.2f}",
        ]
    lines += ["", "## Geometric baselines (rank by distance to actor at t)",
              "| split | top-1 | top-3 | top-5 | MRR | mean rank | median rank | random (seed 42) |",
              "|---|---|---|---|---|---|---|---|"]
    for split in ("train", "val"):
        m = summary[split]["metrics"]
        lines.append(
            f"| {split} | {m['top1']:.4f} | {m['top3']:.4f} | {m['top5']:.4f} | "
            f"{m['mrr']:.4f} | {m['mean_rank']:.2f} | {m['median_rank']} | "
            f"{m['random_uniform_accuracy_seed42']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train", default="data/train_pass_receiver.jsonl")
    parser.add_argument("--val", default="data/val_pass_receiver.jsonl")
    parser.add_argument("--baselines", default="analysis/pass_receiver_baselines.json")
    parser.add_argument("--report", default="analysis/pass_receiver_dataset_report.md")
    args = parser.parse_args()
    for path in (args.train, args.val, args.baselines, args.report):
        if Path(path).exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    output, excluded, totals = build(args.data_dir)
    summary = {}
    for split in ("train", "val"):
        examples = output[split]
        bad = [p for ex in examples for p in validate_prompt(ex["prompt"])]
        if bad:
            raise ValueError(f"prompt validation failed: {bad[:3]}")
        jumps = [max_identity_jump(ex["prompt"]) for ex in examples]
        summary[split] = {
            "pass_total": totals[split],
            "examples": len(examples),
            "excluded_total": totals[split] - len(examples),
            "excluded": dict(excluded[split].most_common()),
            "targets": target_distribution(examples),
            "metrics": geometric_metrics(examples),
            "jump_median": statistics.median(jumps),
            "jump_max": max(jumps),
            "matches": sorted({ex["match_id"] for ex in examples}),
        }
    write_jsonl(args.train, output["train"])
    write_jsonl(args.val, output["val"])
    with open(args.baselines, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    with open(args.report, "w", encoding="utf-8") as stream:
        stream.write(report_markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
