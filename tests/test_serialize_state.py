import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.make_sft_dataset import main as make_sft_dataset
from src.serialize_state import serialize_state


class AnonymousSerializerTests(unittest.TestCase):
    def test_player_identity_changes_do_not_change_anonymous_state(self):
        state = {
            "match_id": "match-a",
            "period": 1,
            "timestamp_ms": 1234,
            "actor_id": "actor-id",
            "actor_team": "home",
            "possession_team": "home",
            "players": [
                {
                    "player_id": "opponent-id",
                    "track_id": "opponent-track",
                    "jersey": 9,
                    "team": "away",
                    "role": "player",
                    "x": 12.0,
                    "y": 10.0,
                    "vx": 1.0,
                    "vy": 0.0,
                },
                {
                    "player_id": "teammate-far",
                    "track_id": "teammate-track-far",
                    "jersey": 8,
                    "team": "home",
                    "role": "player",
                    "x": 30.0,
                    "y": 10.0,
                    "vx": 0.0,
                    "vy": 1.0,
                },
                {
                    "player_id": "actor-id",
                    "track_id": "actor-track",
                    "jersey": 10,
                    "team": "home",
                    "role": "player",
                    "x": 10.0,
                    "y": 10.0,
                    "vx": 0.5,
                    "vy": 0.0,
                },
                {
                    "player_id": "teammate-near",
                    "track_id": "teammate-track-near",
                    "jersey": 7,
                    "team": "home",
                    "role": "goalkeeper",
                    "x": 11.0,
                    "y": 10.0,
                    "vx": 0.0,
                    "vy": 0.0,
                },
            ],
            "action": {"label": "Pass"},
        }

        changed_ids = copy.deepcopy(state)
        changed_ids["match_id"] = "different-match"
        for index, player in enumerate(changed_ids["players"]):
            player["player_id"] = f"new-player-{index}"
            player["track_id"] = f"new-track-{index}"
            player["jersey"] = index + 20
        changed_ids["actor_id"] = "new-player-2"

        serialized = serialize_state(state, mode="anonymous")
        self.assertEqual(serialized, serialize_state(changed_ids, mode="anonymous"))
        self.assertNotIn("MATCH", serialized)
        self.assertNotIn("player_id", serialized)
        self.assertNotIn("track_id", serialized)
        self.assertNotIn("jersey", serialized)
        self.assertLess(serialized.index("TEAMMATE_1"), serialized.index("TEAMMATE_2"))
        self.assertLess(serialized.index("TEAMMATE_2"), serialized.index("OPPONENT_1"))
        self.assertIn("ACTOR team=home role=player x=10.00 y=10.00", serialized)

    def test_identity_mode_preserves_identity_fields(self):
        state = {
            "actor_id": "actor-id",
            "actor_team": "home",
            "players": [
                {
                    "player_id": "actor-id",
                    "track_id": "actor-track",
                    "jersey": 10,
                    "team": "home",
                    "role": "player",
                    "x": 1.0,
                    "y": 2.0,
                }
            ],
            "action": {"label": "Pass"},
        }
        serialized = serialize_state(state, mode="identity")
        self.assertIn("ACTOR actor-id", serialized)
        self.assertIn("PLAYER id=actor-id track=actor-track", serialized)
        self.assertIn("jersey=10", serialized)

    def test_anonymous_mode_rejects_missing_actor_team(self):
        state = {
            "actor_id": "actor-id",
            "players": [
                {
                    "player_id": "actor-id",
                    "team": None,
                    "role": "player",
                    "x": 1.0,
                    "y": 2.0,
                },
                {
                    "player_id": "other-id",
                    "team": "away",
                    "role": "player",
                    "x": 3.0,
                    "y": 4.0,
                },
            ],
        }

        with self.assertRaisesRegex(ValueError, "requires actor_team"):
            serialize_state(state, mode="anonymous")

    def test_anonymous_mode_rejects_actor_without_coordinates(self):
        state = {
            "actor_id": "actor-id",
            "actor_team": "home",
            "players": [{"player_id": "actor-id", "team": "home", "x": None, "y": 2.0}],
        }

        with self.assertRaisesRegex(ValueError, "actor to be present.*x and y"):
            serialize_state(state, mode="anonymous")

    def test_sft_generation_skips_actor_without_available_team(self):
        row = {
            "decision_id": "decision-1",
            "match_id": "match-1",
            "actor_id": "actor-id",
            "players": [
                {
                    "player_id": "actor-id",
                    "team": None,
                    "role": "player",
                    "x": 1.0,
                    "y": 2.0,
                }
            ],
            "action": {"label": "Pass"},
        }

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            input_path = directory / "input.jsonl"
            train_path = directory / "train.jsonl"
            val_path = directory / "val.jsonl"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            argv = [
                "make_sft_dataset.py",
                "--input",
                str(input_path),
                "--train",
                str(train_path),
                "--val",
                str(val_path),
            ]
            with patch("sys.argv", argv):
                make_sft_dataset()

            self.assertEqual(train_path.read_text(encoding="utf-8"), "")
            self.assertEqual(val_path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
