"""Audit-only: can BAS event sequences identify the receiver of a pass?

Reads existing per-half decision files, BAS event files and ball files.
Writes only the CSV given by --csv; never modifies datasets.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict

import numpy as np

FPS = 25.0
MAX_GAP = int(3 * FPS)
HIGH_GAP = int(2 * FPS)
MULTI_WINDOW = int(0.5 * FPS)
SOURCE = {"Pass": "PASS", "High Pass": "HIGH_PASS", "Cross": "CROSS"}
RECEIVER_EVENTS = {"PASS", "HIGH PASS", "DRIVE", "CROSS", "SHOT", "HEADER"}
STOP_EVENTS = {"OUT", "GOAL", "THROW IN", "FREE KICK"}
DEFENSIVE = {"BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE"}


def load_decisions(data_dir):
    decisions = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*_decisions.jsonl"))):
        if os.path.basename(path).startswith("all_"):
            continue
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    decisions[(row["match_id"], row["period"], row["frame"])] = row
    return decisions


def load_events(data_dir):
    events = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*_12_class_events.json"))):
        with open(path, encoding="utf-8") as stream:
            doc = json.load(stream)
        match_id = str(doc["match_id"])
        for action in doc["actions"]:
            period = int(action["gameTime"].split(" - ")[0])
            events.setdefault((match_id, period), []).append(
                {
                    "frame": int(action["frame"]),
                    "label": action["label"].upper(),
                    "team": action["team"],
                    "player": str(action.get("player_id") or ""),
                }
            )
    for key in events:
        events[key].sort(key=lambda event: event["frame"])
    return events


def load_ball(data_dir):
    ball = {}
    for path in glob.glob(os.path.join(data_dir, "*_ball.npz")):
        name = os.path.basename(path)
        match_id, half, _ = name.split("_")
        period = 1 if half == "1st" else 2
        with np.load(path) as archive:
            ball[(match_id, period)] = {
                "frame": archive["frame"],
                "x": archive["x"],
                "y": archive["y"],
            }
    return ball


def ball_at(ball, match_id, period, frame):
    data = ball.get((match_id, period))
    if data is None:
        return None
    index = np.searchsorted(data["frame"], frame)
    if index >= len(data["frame"]) or data["frame"][index] != frame:
        return None
    return float(data["x"][index]), float(data["y"][index])


def players_by_key(decision):
    return {
        str(p["player_id"] if p.get("player_id") is not None else p.get("track_id")): p
        for p in decision["players"]
    }


def teammate_label(decision, receiver_id):
    """Replicates make_temporal_sft_dataset anonymous labels (distance to actor at t)."""
    actor_id = str(decision["actor_id"])
    players = decision["players"]
    keyed = [
        (
            str(p["player_id"] if p.get("player_id") is not None else p.get("track_id")),
            index,
            p,
        )
        for index, p in enumerate(players)
    ]
    actor = next((p for key, _, p in keyed if key == actor_id), None)
    if actor is None:
        return None
    team = decision["actor_team"]
    group = []
    for key, index, p in keyed:
        if key == actor_id or p.get("team") != team:
            continue
        distance = math.hypot(p["x"] - actor["x"], p["y"] - actor["y"])
        group.append((distance, index, key))
    group.sort()
    for number, (_, _, key) in enumerate(group, 1):
        if key == receiver_id:
            return f"TEAMMATE_{number}"
    return None


def find_receiver(decision, events, decisions):
    """Return an audit record dict for one pass-like decision."""
    match_id, period, frame = (
        decision["match_id"],
        decision["period"],
        decision["frame"],
    )
    passer = str(decision["actor_id"])
    team = decision["actor_team"]
    rec = {
        "receiver_id": "",
        "next_action": "",
        "gap_frames": "",
        "gap_s": "",
        "flags": [],
        "failure": "",
        "intervening": [],
    }
    seq = events.get((match_id, period), [])
    own = [
        i
        for i, e in enumerate(seq)
        if e["frame"] == frame and e["player"] == passer
    ]
    if not own:
        rec["failure"] = "source_event_not_found"
        return rec, None
    start = own[0]
    repeat = False
    for e in seq[start + 1 :]:
        gap = e["frame"] - frame
        if gap > MAX_GAP:
            rec["failure"] = "no_event_within_3s"
            return rec, None
        rec["intervening"].append(f"{e['label']}:{e['team']}:{e['player']}@{gap}")
        if e["label"] in STOP_EVENTS:
            rec["failure"] = "restart_or_outcome_first"
            return rec, None
        if e["team"] != team:
            rec["failure"] = "opponent_action_first"
            return rec, None
        if e["label"] in DEFENSIVE:
            rec["failure"] = "same_team_defensive_event"
            return rec, None
        if not e["player"]:
            rec["failure"] = "next_event_without_player_id"
            return rec, None
        if e["player"] == passer:
            repeat = True
            continue
        if e["label"] in RECEIVER_EVENTS:
            rec.update(
                receiver_id=e["player"],
                next_action=e["label"],
                gap_frames=gap,
                gap_s=round(gap / FPS, 2),
            )
            if repeat:
                rec["flags"].append("passer_acted_again_first")
            third = [
                x
                for x in seq
                if frame + gap < x["frame"] <= frame + gap + MULTI_WINDOW
                and x["team"] == team
                and x["player"] not in (passer, e["player"])
                and x["label"] in RECEIVER_EVENTS
            ]
            if third or any(
                x["frame"] == e["frame"]
                and x["player"] != e["player"]
                and x["team"] == team
                for x in seq
            ):
                rec["flags"].append("multiple_plausible_receivers")
            return rec, e
    rec["failure"] = "no_event_within_3s"
    return rec, None


def audit_row(decision, events, decisions, ball):
    rec, nxt = find_receiver(decision, events, decisions)
    match_id, period, frame = (
        decision["match_id"],
        decision["period"],
        decision["frame"],
    )
    passer = str(decision["actor_id"])
    now = players_by_key(decision)
    passer_state = now.get(passer)
    row = {
        "decision_id": decision["decision_id"],
        "match_id": match_id,
        "period": period,
        "frame": frame,
        "source_action": SOURCE[decision["action"]["label"]],
        "passer_id": passer,
        "passer_team": decision["actor_team"],
        "passer_x": passer_state["x"] if passer_state else "",
        "passer_y": passer_state["y"] if passer_state else "",
        "receiver_id": rec["receiver_id"],
        "next_action": rec["next_action"],
        "gap_frames": rec["gap_frames"],
        "gap_s": rec["gap_s"],
        "receiver_x_t": "",
        "receiver_y_t": "",
        "receiver_x_event": "",
        "receiver_y_event": "",
        "receiver_dist_to_passer_t": "",
        "receiver_teammate_label": "",
        "receiver_in_t_state": "",
        "ball_to_receiver_at_event": "",
        "ball_to_passer_at_t": "",
        "sequence": "|".join(rec["intervening"][:6]),
    }
    flags = list(rec["flags"])
    failure = rec["failure"]
    if passer_state is None:
        flags.append("passer_missing_at_t")
    if nxt is not None:
        rid = rec["receiver_id"]
        r_now = now.get(rid)
        row["receiver_in_t_state"] = r_now is not None
        if r_now is None or r_now.get("team") != decision["actor_team"]:
            flags.append("receiver_absent_from_t_state")
        else:
            row["receiver_x_t"], row["receiver_y_t"] = r_now["x"], r_now["y"]
            if passer_state:
                row["receiver_dist_to_passer_t"] = round(
                    math.hypot(
                        r_now["x"] - passer_state["x"], r_now["y"] - passer_state["y"]
                    ),
                    2,
                )
            row["receiver_teammate_label"] = teammate_label(decision, rid) or ""
        later = decisions.get((match_id, period, nxt["frame"]))
        later_state = players_by_key(later).get(rid) if later else None
        if later_state is None:
            flags.append("receiver_missing_at_event")
        else:
            row["receiver_x_event"], row["receiver_y_event"] = (
                later_state["x"],
                later_state["y"],
            )
        b_event = ball_at(ball, match_id, period, nxt["frame"])
        if b_event and later_state:
            row["ball_to_receiver_at_event"] = round(
                math.hypot(b_event[0] - later_state["x"], b_event[1] - later_state["y"]),
                2,
            )
    b_t = ball_at(ball, match_id, period, frame)
    if b_t and passer_state:
        row["ball_to_passer_at_t"] = round(
            math.hypot(b_t[0] - passer_state["x"], b_t[1] - passer_state["y"]), 2
        )

    if nxt is None:
        confidence = "LOW"
        reason = failure or "no_receiver"
    else:
        blocking = [
            f
            for f in flags
            if f
            in (
                "passer_missing_at_t",
                "receiver_absent_from_t_state",
                "receiver_missing_at_event",
                "multiple_plausible_receivers",
                "passer_acted_again_first",
            )
        ]
        if blocking:
            confidence, reason = "LOW", "+".join(blocking)
        elif rec["gap_frames"] <= HIGH_GAP:
            confidence, reason = "HIGH", ""
        else:
            confidence, reason = "MEDIUM", ""
    row["confidence"] = confidence
    row["reason"] = reason
    row["flags"] = "+".join(flags)
    return row


def stratified_sample(rows, quotas, seed):
    rng = random.Random(seed)
    chosen = []
    for label, quota in quotas.items():
        pool = defaultdict(list)
        for d in rows:
            if SOURCE[d["action"]["label"]] == label:
                pool[(d["match_id"], d["period"])].append(d)
        for key in pool:
            rng.shuffle(pool[key])
        keys = sorted(pool)
        picked = []
        while len(picked) < quota and any(pool[k] for k in keys):
            for key in keys:
                if pool[key] and len(picked) < quota:
                    picked.append(pool[key].pop())
        chosen.extend(picked)
    return chosen


def summarize(rows, title):
    out = [title]
    for label in ("PASS", "HIGH_PASS", "CROSS", "ALL"):
        subset = [r for r in rows if label == "ALL" or r["source_action"] == label]
        counts = Counter(r["confidence"] for r in subset)
        gaps = [r["gap_s"] for r in subset if r["gap_s"] != ""]
        usable = counts["HIGH"] + counts["MEDIUM"]
        out.append(
            f"{label}: n={len(subset)} HIGH={counts['HIGH']} MEDIUM={counts['MEDIUM']} "
            f"LOW={counts['LOW']} usable={100 * usable / max(1, len(subset)):.1f}% "
            f"high-only={100 * counts['HIGH'] / max(1, len(subset)):.1f}% "
            f"median_gap_s={statistics.median(gaps) if gaps else 'NA'}"
        )
        reasons = Counter(r["reason"] for r in subset if r["confidence"] == "LOW")
        out.append("   LOW reasons: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common(6)))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--csv", default="analysis/pass_receiver_audit.csv")
    parser.add_argument("--population-csv", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if os.path.exists(args.csv):
        raise FileExistsError(f"Refusing to overwrite {args.csv}")

    decisions = load_decisions(args.data_dir)
    events = load_events(args.data_dir)
    ball = load_ball(args.data_dir)
    pass_like = [d for d in decisions.values() if d["action"]["label"] in SOURCE]
    print(f"decisions={len(decisions)} pass-like={len(pass_like)}")

    population = [audit_row(d, events, decisions, ball) for d in pass_like]
    print(summarize(population, "FULL POPULATION (all retained pass-like decisions)"))

    sample = stratified_sample(
        pass_like, {"PASS": 150, "HIGH_PASS": 100, "CROSS": 50}, args.seed
    )
    sampled_ids = {d["decision_id"] for d in sample}
    rows = [r for r in population if r["decision_id"] in sampled_ids]
    rows.sort(key=lambda r: (r["source_action"], r["match_id"], r["period"], r["frame"]))
    with open(args.csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(summarize(rows, "SAMPLE"))
    print(
        "sample strata:",
        dict(Counter((r["source_action"], r["match_id"], r["period"]) for r in rows)),
    )
    if args.population_csv:
        with open(args.population_csv, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(population[0].keys()))
            writer.writeheader()
            writer.writerows(population)


if __name__ == "__main__":
    main()
