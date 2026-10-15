import re
import unittest

from src.build_pass_receiver_dataset import (
    TRAIN_MATCHES,
    VAL_MATCHES,
    build_receiver_example,
    classify_pass,
    geometric_metrics,
    is_high_confidence,
    split_for_match,
    validate_prompt,
)

MATCH, PERIOD, FRAME = "900001", 1, 1000
PERMANENT_IDS = ("111", "222", "333", "444", "555", "666")


def pl(pid, team, x, y):
    return {
        "player_id": pid,
        "track_id": pid,
        "team": team,
        "role": "player",
        "x": x,
        "y": y,
    }


def players(shift=0.0):
    # passer 111; teammates ordered by distance: 222 (3 m), 333 (6 m), 444 (9 m)
    return [
        pl("111", "left", 0.0 + shift, 0.0),
        pl("444", "left", 9.0 + shift, 0.0),
        pl("222", "left", 3.0 + shift, 0.0),
        pl("333", "left", 6.0 + shift, 0.0),
        pl("555", "right", 20.0, 0.0),
        pl("666", "right", 25.0, 0.0),
    ]


def decision(frame, actor, label):
    return {
        "decision_id": f"{MATCH}_p{PERIOD}_f{frame}",
        "match_id": MATCH,
        "period": PERIOD,
        "frame": frame,
        "actor_id": actor,
        "actor_team": "left",
        "players": players(),
        "action": {"label": label},
    }


def event(frame, label, team, player):
    return {"frame": frame, "label": label, "team": team, "player": player}


def scenario(next_events):
    seq = [event(FRAME, "PASS", "left", "111")] + next_events
    events = {(MATCH, PERIOD): seq}
    index = {(MATCH, PERIOD, FRAME): decision(FRAME, "111", "Pass")}
    for e in next_events:
        index[(MATCH, PERIOD, e["frame"])] = decision(
            e["frame"], e["player"], "Drive"
        )
    return index[(MATCH, PERIOD, FRAME)], events, index


def snapshots_for(record):
    state = record["players"]
    return {-10: (990, state), -5: (995, state), 0: (FRAME, state)}


class PassReceiverDatasetTest(unittest.TestCase):
    def test_valid_high_confidence_receiver(self):
        rec, events, index = scenario([event(FRAME + 30, "DRIVE", "left", "333")])
        row = classify_pass(rec, events, index)
        self.assertTrue(is_high_confidence(row))
        self.assertEqual(row["receiver_id"], "333")
        self.assertEqual(row["gap_frames"], 30)

    def test_opponent_event_rejects(self):
        rec, events, index = scenario(
            [
                event(FRAME + 10, "DRIVE", "right", "555"),
                event(FRAME + 30, "DRIVE", "left", "333"),
            ]
        )
        self.assertFalse(is_high_confidence(classify_pass(rec, events, index)))

    def test_receiver_after_two_seconds_rejects(self):
        rec, events, index = scenario([event(FRAME + 60, "DRIVE", "left", "333")])
        row = classify_pass(rec, events, index)
        self.assertFalse(is_high_confidence(row))
        self.assertEqual(row["confidence"], "MEDIUM")

    def test_restart_event_rejects(self):
        rec, events, index = scenario(
            [
                event(FRAME + 10, "OUT", "left", "222"),
                event(FRAME + 30, "DRIVE", "left", "333"),
            ]
        )
        self.assertFalse(is_high_confidence(classify_pass(rec, events, index)))

    def test_receiver_maps_to_correct_teammate(self):
        rec, events, index = scenario([event(FRAME + 30, "DRIVE", "left", "333")])
        row = classify_pass(rec, events, index)
        example = build_receiver_example(rec, row, snapshots_for(rec))
        self.assertEqual(example["response"], "TEAMMATE_2")
        self.assertEqual(example["receiver_id"], "333")
        # TEAMMATE_2 is the player 6 m from the actor in the prompt
        self.assertIn("TEAMMATE_2 team=left role=player x=6.00 y=0.00", example["prompt"])

    def test_identity_consistent_across_frames(self):
        rec, events, index = scenario([event(FRAME + 30, "DRIVE", "left", "444")])
        row = classify_pass(rec, events, index)
        snaps = snapshots_for(rec)
        snaps[-10] = (990, players(shift=-1.0))
        # reorder the historical list: identity must follow the player, not the list slot
        snaps[-10] = (990, list(reversed(snaps[-10][1])))
        example = build_receiver_example(rec, row, snaps)
        self.assertEqual(example["response"], "TEAMMATE_3")
        first_block = example["prompt"].split("STATE_T_MINUS_0.2S")[0]
        self.assertIn("TEAMMATE_3 team=left role=player x=8.00 y=0.00", first_block)

    def test_no_permanent_ids_or_future_info_in_prompt(self):
        rec, events, index = scenario([event(FRAME + 30, "DRIVE", "left", "333")])
        row = classify_pass(rec, events, index)
        example = build_receiver_example(rec, row, snapshots_for(rec))
        for pid in PERMANENT_IDS:
            self.assertIsNone(re.search(rf"\b{pid}\b", example["prompt"]))
        self.assertEqual(validate_prompt(example["prompt"]), [])
        self.assertNotIn("DRIVE", example["prompt"])
        self.assertNotIn("PLUS", example["prompt"])

    def test_split_preserved(self):
        self.assertEqual(
            set(TRAIN_MATCHES),
            {"117092", "117093", "118576", "118578", "128057", "132831", "132877"},
        )
        self.assertEqual(set(VAL_MATCHES), {"118577", "128058"})
        self.assertFalse(set(TRAIN_MATCHES) & set(VAL_MATCHES))
        self.assertEqual(split_for_match("118577"), "val")
        self.assertEqual(split_for_match("117092"), "train")
        self.assertIsNone(split_for_match("118575"))

    def test_geometric_metrics(self):
        rec, events, index = scenario([event(FRAME + 30, "DRIVE", "left", "333")])
        row = classify_pass(rec, events, index)
        example = build_receiver_example(rec, row, snapshots_for(rec))
        metrics = geometric_metrics([example])
        self.assertEqual(metrics["top1"], 0.0)
        self.assertEqual(metrics["top3"], 1.0)
        self.assertEqual(metrics["median_rank"], 2)


if __name__ == "__main__":
    unittest.main()
