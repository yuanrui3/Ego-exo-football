import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import ijson

sys.path.insert(0, str(Path(__file__).parent))
from extract_soccertrack_events import decode_frame
from make_sft_dataset import SYSTEM, read, write


HISTORY_OFFSETS = (-10, -5, 0)
STATE_HEADERS = {
    -10: "STATE_T_MINUS_0.4S",
    -5: "STATE_T_MINUS_0.2S",
    0: "STATE_T",
}


class InvalidTemporalDecision(ValueError):
    pass


def player_key(player):
    value = player.get("player_id")
    if value is None:
        value = player.get("track_id")
    return str(value) if value is not None else None


def stream_states_for_frames(gsr_path, targets):
    wanted_frames = set(targets)
    states = {}
    with open(gsr_path, "rb") as stream:
        for annotation in ijson.items(stream, "annotations.item"):
            if annotation.get("supercategory") != "object":
                continue
            attributes = annotation.get("attributes") or {}
            role = attributes.get("role")
            if role not in ("player", "goalkeeper"):
                continue
            frame = decode_frame(annotation.get("image_id"))
            if frame not in wanted_frames:
                continue
            pitch_box = annotation.get("bbox_pitch") or {}
            x = pitch_box.get("x_bottom_middle")
            y = pitch_box.get("y_bottom_middle")
            if x is None or y is None:
                continue
            states.setdefault(frame, []).append(
                {
                    "player_id": attributes.get(
                        "player_id", annotation.get("track_id")
                    ),
                    "track_id": annotation.get("track_id"),
                    "team": attributes.get("team"),
                    "role": role,
                    "jersey": attributes.get("jersey"),
                    "x": float(x),
                    "y": float(y),
                    "vx": None,
                    "vy": None,
                }
            )
    return states


def find_snapshot(target, states):
    players = states.get(target, [])
    return (target, players) if players else (None, None)


def prepare_entities(record, current_players):
    actor_id = record.get("actor_id")
    if actor_id is None:
        raise InvalidTemporalDecision("actor id is missing")
    actor = next(
        (
            player
            for player in current_players
            if player_key(player) == str(actor_id)
        ),
        None,
    )
    if actor is None or actor.get("x") is None or actor.get("y") is None:
        raise InvalidTemporalDecision("actor is missing at event time")
    actor_team = record.get("actor_team") or actor.get("team")
    if not actor_team:
        raise InvalidTemporalDecision("actor team is missing at event time")

    entities = [{"label": "ACTOR", "player": actor, "key": player_key(actor)}]
    if entities[0]["key"] is None:
        raise InvalidTemporalDecision("actor has no stable tracking key")

    teammates = []
    opponents = []
    seen_keys = {entities[0]["key"]}
    for index, player in enumerate(current_players):
        if player is actor:
            continue
        key = player_key(player)
        if key is None or key in seen_keys:
            raise InvalidTemporalDecision(
                "current player states do not have unique tracking keys"
            )
        if not player.get("team"):
            raise InvalidTemporalDecision("current player team is missing")
        seen_keys.add(key)
        try:
            distance = math.hypot(
                float(player["x"]) - float(actor["x"]),
                float(player["y"]) - float(actor["y"]),
            )
        except (KeyError, TypeError, ValueError):
            distance = math.inf
        group = teammates if player.get("team") == actor_team else opponents
        group.append((distance, index, player, key))

    for group, prefix in ((teammates, "TEAMMATE"), (opponents, "OPPONENT")):
        group.sort(key=lambda item: (item[0], item[1]))
        for number, (_, _, player, key) in enumerate(group, 1):
            entities.append(
                {"label": f"{prefix}_{number}", "player": player, "key": key}
            )
    return actor_team, entities


def serialize_temporal_state(period, actor_team, entities, snapshots):
    blocks = []
    for offset, header in STATE_HEADERS.items():
        state_by_key = {
            player_key(player): player for player in snapshots[offset]
        }
        lines = [header]
        if offset == -10:
            lines.insert(0, f"PERIOD {period}")
        for entity in entities:
            player = entity["player"]
            observed = state_by_key.get(entity["key"])
            x = observed.get("x") if observed is not None else None
            y = observed.get("y") if observed is not None else None
            team = actor_team if entity["label"] == "ACTOR" else player.get("team")
            lines.append(
                f"{entity['label']} team={team or 'NA'} "
                f"role={player.get('role') or 'NA'} "
                f"x={'NA' if x is None else f'{x:.2f}'} "
                f"y={'NA' if y is None else f'{y:.2f}'}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_temporal_example(record, snapshots):
    current_offset = snapshots.get(0)
    if current_offset is None:
        raise InvalidTemporalDecision("event-time state is unavailable")
    actor_team, entities = prepare_entities(record, current_offset[1])
    formatted_snapshots = {0: current_offset[1]}
    for offset in (-10, -5):
        snapshot = snapshots.get(offset)
        if snapshot is None:
            raise InvalidTemporalDecision("historical state is unavailable")
        formatted_snapshots[offset] = snapshot[1]

    state = serialize_temporal_state(
        record.get("period"), actor_team, entities, formatted_snapshots
    )
    return {
        "id": record["decision_id"],
        "match_id": str(record["match_id"]),
        "prompt": (
            f"<|system|>\n{SYSTEM}\n<|user|>\n{state}\n<|assistant|>\n"
        ),
        "response": str(record["action"]["label"]).strip(),
    }


def build_for_half(records, gsr_path):
    targets = {
        int(record["frame"]) + offset
        for record in records
        for offset in HISTORY_OFFSETS
    }
    states = stream_states_for_frames(gsr_path, targets)
    examples = []
    unavailable_history = 0
    other_skipped = 0

    for record in records:
        event_frame = int(record["frame"])
        snapshots = {}
        for offset in HISTORY_OFFSETS:
            target = event_frame + offset
            frame, players = find_snapshot(target, states)
            if players is not None:
                snapshots[offset] = (frame, players)
        if len(snapshots) != len(HISTORY_OFFSETS):
            unavailable_history += 1
            continue
        try:
            examples.append(build_temporal_example(record, snapshots))
        except InvalidTemporalDecision:
            other_skipped += 1
    return examples, unavailable_history, other_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Build temporal anonymous SFT examples from SoccerTrack GSR."
    )
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--train-reference", required=True)
    parser.add_argument("--val-reference", required=True)
    parser.add_argument("--gsr-dir", default="data")
    parser.add_argument("--train", default="data/train_temporal.jsonl")
    parser.add_argument("--val", default="data/val_temporal.jsonl")
    args = parser.parse_args()

    decisions = read(args.decisions)
    train_matches = {str(row["match_id"]) for row in read(args.train_reference)}
    val_matches = {str(row["match_id"]) for row in read(args.val_reference)}
    if train_matches & val_matches:
        raise ValueError("Training and validation match sets overlap")

    groups = defaultdict(list)
    for record in decisions:
        groups[(str(record["match_id"]), int(record["period"]))].append(record)

    output = {"train": [], "val": []}
    unavailable_history = 0
    other_skipped = 0
    for (match_id, period), records in sorted(groups.items()):
        gsr_name = "1st" if period == 1 else "2nd"
        gsr_path = Path(args.gsr_dir) / f"{match_id}_{gsr_name}.json"
        if not gsr_path.is_file():
            other_skipped += len(records)
            print(
                f"Missing GSR file; skipped {len(records)} decisions: {gsr_path}",
                file=sys.stderr,
            )
            continue
        examples, missing_history, invalid = build_for_half(records, gsr_path)
        unavailable_history += missing_history
        other_skipped += invalid
        split = "train" if match_id in train_matches else "val"
        if match_id not in train_matches and match_id not in val_matches:
            other_skipped += len(examples)
            continue
        output[split].extend(examples)

    write(args.train, output["train"])
    write(args.val, output["val"])
    print(f"source decisions: {len(decisions)}")
    print(f"train temporal examples: {len(output['train'])}")
    print(f"validation temporal examples: {len(output['val'])}")
    print(f"skipped due to unavailable required history frames: {unavailable_history}")
    print(f"other skipped decisions: {other_skipped}")


if __name__ == "__main__":
    main()
