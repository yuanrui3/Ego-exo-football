import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.extract_soccertrack_events import main


def event(frame, player_id):
    return {
        "gameTime": "1 - 0:01",
        "label": "PASS",
        "position": str(frame * 40),
        "frame": frame,
        "team": "right",
        "player_id": player_id,
    }


def gsr_image(frame, player_id, x=42.5, y=17.25):
    return (
        {"image_id": str(frame), "file_name": f"frame_{frame}.jpg"},
        {
            "image_id": str(frame),
            "supercategory": "object",
            "track_id": player_id,
            "attributes": {"role": "player", "player_id": player_id, "team": "right"},
            "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y},
        },
    )


class ExtractSoccerTrackEventsTests(unittest.TestCase):
    def run_extractor(self, directory, events, frames, include_actor_missing=False):
        bas_path = directory / "bas.json"
        gsr_path = directory / "gsr.json"
        output_path = directory / "decisions.jsonl"
        bas_path.write_text(json.dumps({"actions": events}), encoding="utf-8")

        images = []
        annotations = []
        for frame, player_id in frames:
            image, annotation = gsr_image(frame, player_id)
            images.append(image)
            annotations.append(annotation)
        gsr_path.write_text(
            json.dumps({"images": images, "annotations": annotations}),
            encoding="utf-8",
        )

        argv = [
            "extract_soccertrack_events.py",
            "--bas",
            str(bas_path),
            "--gsr",
            str(gsr_path),
            "--match-id",
            "test",
            "--period",
            "1",
            "--output",
            str(output_path),
        ]
        if include_actor_missing:
            argv.append("--include-actor-missing")

        stdout = io.StringIO()
        with patch("sys.argv", argv), contextlib.redirect_stdout(stdout):
            main()

        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        return rows, stdout.getvalue()

    def test_exact_match_and_plus_one_fallback_record_offsets(self):
        with tempfile.TemporaryDirectory() as temp:
            rows, output = self.run_extractor(
                Path(temp),
                [event(10, "actor-10"), event(20, "actor-20")],
                [(10, "actor-10"), (21, "other-player")],
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["matched_gsr_frame"], 10)
        self.assertEqual(rows[0]["gsr_frame_offset"], 0)
        self.assertEqual(rows[0]["possession_team"], None)
        self.assertEqual(rows[0]["actor_team"], "right")
        self.assertEqual(rows[0]["players"][0]["x"], 42.5)
        self.assertEqual(rows[0]["players"][0]["y"], 17.25)
        self.assertIn("over state-matched events (including actor-missing)", output)
        self.assertIn("-1=0, 0=1, +1=1", output)
        self.assertIn("records written:                         1", output)
        self.assertIn("actor-missing count:                     1", output)
        self.assertIn("actor-presence rate:                     50.00%", output)

    def test_actor_missing_is_skipped_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            rows, output = self.run_extractor(
                Path(temp),
                [event(20, "event-actor")],
                [(20, "different-player")],
            )

        self.assertEqual(rows, [])
        self.assertIn("BAS event count:", output)
        self.assertIn("records written:                         0", output)
        self.assertIn("actor-missing count:                     1", output)
        self.assertIn("skipped because actor missing:           1", output)

    def test_actor_missing_can_be_included_explicitly_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            rows, output = self.run_extractor(
                Path(temp),
                [event(20, "event-actor")],
                [(20, "different-player")],
                include_actor_missing=True,
            )

        self.assertEqual(len(rows), 1)
        self.assertIn("skipped because actor missing:           0", output)


if __name__ == "__main__":
    unittest.main()
