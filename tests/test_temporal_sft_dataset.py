import json
import tempfile
import unittest
from pathlib import Path

from src.make_temporal_sft_dataset import (
    InvalidTemporalDecision,
    build_temporal_example,
    find_snapshot,
    prepare_entities,
    stream_states_for_frames,
)


def player(player_id, team, x, y):
    return {
        "player_id": player_id,
        "track_id": player_id,
        "team": team,
        "role": "player",
        "x": x,
        "y": y,
    }


class TemporalSftDatasetTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "decision_id": "match_p1_f100_0",
            "match_id": "match",
            "period": 1,
            "frame": 100,
            "actor_id": "a",
            "actor_team": "left",
            "action": {"label": "Pass"},
        }
        self.current = [
            player("a", "left", 0.0, 0.0),
            player("b", "left", 1.0, 0.0),
            player("c", "left", -3.0, 0.0),
            player("d", "right", 0.0, 2.0),
            player("e", "right", 0.0, -5.0),
        ]

    def test_labels_are_assigned_at_event_time_and_held_over_history(self):
        earlier = [
            player("a", "left", -2.0, 0.0),
            player("b", "left", 2.0, 0.0),
            player("c", "left", -2.0, -3.0),
            player("d", "right", 0.0, 4.0),
            player("e", "right", 0.0, -7.0),
        ]
        snapshots = {
            -10: (90, earlier),
            -5: (95, earlier),
            0: (100, self.current),
        }
        example = build_temporal_example(self.record, snapshots)
        prompt = example["prompt"]
        self.assertIn("STATE_T_MINUS_0.4S\nACTOR team=left", prompt)
        self.assertIn("STATE_T_MINUS_0.2S\nACTOR team=left", prompt)
        self.assertIn("STATE_T\nACTOR team=left", prompt)
        first_history = prompt.split("STATE_T_MINUS_0.4S", 1)[1].split(
            "STATE_T_MINUS_0.2S", 1
        )[0]
        self.assertIn("TEAMMATE_1 team=left role=player x=2.00 y=0.00", first_history)
        self.assertIn(
            "TEAMMATE_2 team=left role=player x=-2.00 y=-3.00", first_history
        )
        self.assertIn("OPPONENT_1 team=right role=player x=0.00 y=4.00", first_history)
        self.assertIn("OPPONENT_2 team=right role=player x=0.00 y=-7.00", first_history)
        self.assertNotIn("match_id=", prompt)
        self.assertNotIn("player_id=", prompt)
        self.assertNotIn("track_id=", prompt)
        self.assertNotIn("vx=", prompt)
        self.assertNotIn("vy=", prompt)
        self.assertNotIn("Pass", prompt)
        self.assertEqual(example["response"], "Pass")

    def test_missing_historical_player_is_serialized_as_na(self):
        snapshots = {
            -10: (90, [self.current[0], self.current[2], self.current[3], self.current[4]]),
            -5: (95, self.current),
            0: (100, self.current),
        }
        prompt = build_temporal_example(self.record, snapshots)["prompt"]
        block = prompt.split("STATE_T_MINUS_0.4S", 1)[1].split(
            "STATE_T_MINUS_0.2S", 1
        )[0]
        self.assertIn("TEAMMATE_1 team=left role=player x=NA y=NA", block)

    def test_event_state_lookup_never_uses_a_future_frame(self):
        states = {101: [player("a", "left", 0.0, 0.0)]}
        frame, players = find_snapshot(100, states)
        self.assertIsNone(frame)
        self.assertIsNone(players)

    def test_gsr_states_are_streamed_by_decoded_image_frame(self):
        annotation = {
            "image_id": "99",
            "supercategory": "object",
            "track_id": 17,
            "attributes": {
                "player_id": "a",
                "team": "left",
                "role": "player",
            },
            "bbox_pitch": {"x_bottom_middle": 1.25, "y_bottom_middle": -2.5},
        }
        with tempfile.TemporaryDirectory() as directory:
            gsr_path = Path(directory) / "synthetic.json"
            gsr_path.write_text(
                json.dumps({"images": [], "annotations": [annotation]}),
                encoding="utf-8",
            )
            states = stream_states_for_frames(gsr_path, {99})
        self.assertEqual(states[99][0]["player_id"], "a")
        self.assertEqual((states[99][0]["x"], states[99][0]["y"]), (1.25, -2.5))

    def test_unknown_actor_team_is_rejected(self):
        record = {**self.record, "actor_team": None}
        players = [player("a", None, 0.0, 0.0)]
        with self.assertRaises(InvalidTemporalDecision):
            prepare_entities(record, players)


if __name__ == "__main__":
    unittest.main()
