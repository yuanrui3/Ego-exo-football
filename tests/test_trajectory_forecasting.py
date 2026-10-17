import copy
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.models.trajectory_forecasting import (
    TrajectoryForecaster, constant_velocity, masked_loss, persistence,
)
from src.trajectory_dataset import (
    TRAIN_MATCHES, VALIDATION_MATCHES, build_half, causal_velocity, identity, make_window, write_json,
    segment_tracks, validate_splits,
)
from train.run_trajectory_forecasting import finish_metric, metric_sums, pass_frames


def raw_tracks():
    frames = np.arange(160, dtype=np.float32)
    xy = np.zeros((160, 2, 2), dtype=np.float32)
    xy[:, 0, 0] = frames * .04 * 2
    xy[:, 1, 1] = frames * .04
    observed = np.ones((160, 2), bool)
    return {
        "xy": xy, "observed": observed, "segments": np.ones((160, 2), np.int32),
        "teams": np.tile(np.array([1, 2], np.int8), (160, 1)),
        "roles": np.tile(np.array([0, 1], np.int8), (160, 1)),
    }


def tensor_batch(row):
    return {key: torch.from_numpy(value)[None] for key, value in row.items()}


class TrajectoryTests(unittest.TestCase):
    def test_atomic_json_replace_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            real_replace = os.replace
            attempts = 0

            def transient_replace(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("simulated transient file lock")
                real_replace(source, destination)

            with patch("src.trajectory_dataset.os.replace", side_effect=transient_replace), \
                    patch("src.trajectory_dataset.time.sleep"):
                write_json(target, {"status": "RUNNING"})
            self.assertEqual(attempts, 3)
            self.assertEqual(target.read_text(encoding="utf-8"), '{\n  "status": "RUNNING"\n}\n')

    def test_identity_requires_player_and_track(self):
        self.assertEqual(identity({"attributes": {"player_id": 9}, "track_id": 3}), ("9", "3"))
        self.assertIsNone(identity({"attributes": {"player_id": None}, "track_id": 3}))

    def test_exact_targets_velocity_and_masks(self):
        row = make_window(raw_tracks(), 60)
        np.testing.assert_allclose(row["target"][0, :, 0], [.8, 2, 4, 6], atol=1e-5)
        np.testing.assert_allclose(row["velocity"][0], [2, 0], atol=1e-5)
        self.assertEqual(row["history"].shape, (22, 51, 5))
        self.assertEqual(row["target_mask"].sum(), 8)
        self.assertFalse(row["player_mask"][2:].any())
        batch = tensor_batch(row)
        np.testing.assert_allclose(constant_velocity(batch).numpy()[0, :2],
                                   row["target"][:2], atol=1e-5)
        self.assertEqual(persistence(batch).abs().sum(), 0)

    def test_discontinuities_are_never_bridged(self):
        raw = raw_tracks()
        raw["observed"][50, 0] = False
        tracks = np.zeros((160, 2), np.int32)
        tracks[80:, 0] = 2
        raw["xy"][90:, 1, 0] += 10
        _, raw["segments"], stats = segment_tracks(
            raw["xy"], raw["observed"], tracks, raw["teams"], raw["roles"]
        )
        row = make_window(raw, 60)
        self.assertFalse(row["history"][0, :41, 4].any())
        self.assertTrue(row["history"][0, 41:, 4].all())
        self.assertTrue(row["target_mask"][0, 0])
        self.assertFalse(row["target_mask"][0, 1:].any())
        self.assertFalse(row["target_mask"][1, 2:].any())
        self.assertEqual(stats["implausible_adjacent_jumps"], 1)

    def test_future_does_not_change_inputs(self):
        a = raw_tracks()
        b = copy.deepcopy(a)
        b["xy"][61:] += 20
        b["observed"][90:] = False
        x, y = make_window(a, 60), make_window(b, 60)
        for key in ("history", "current", "velocity", "team", "role", "player_mask"):
            np.testing.assert_array_equal(x[key], y[key])

    def test_velocity_safeguards_and_static(self):
        raw = raw_tracks()
        raw["xy"] *= 100
        row = make_window(raw, 60)
        self.assertLessEqual(np.linalg.norm(row["velocity"][0]), 12.00001)
        history = np.zeros((1, 11, 2), np.float32)
        mask = np.zeros((1, 11), bool)
        mask[0, -3:] = True
        self.assertFalse(causal_velocity(history, mask).any())
        row = make_window(raw_tracks(), 60, history_frames=0)
        self.assertFalse(row["velocity"].any())
        self.assertFalse(row["history"][..., :4].any())

    def test_split_isolation(self):
        validate_splits(TRAIN_MATCHES, VALIDATION_MATCHES)
        with self.assertRaises(ValueError):
            validate_splits([*TRAIN_MATCHES, "118577"], VALIDATION_MATCHES)
        with self.assertRaises(ValueError):
            validate_splits(TRAIN_MATCHES, ["118577"])

    def test_metrics_are_meters_and_ignore_invalid(self):
        batch = tensor_batch(make_window(raw_tracks(), 60))
        prediction = batch["target"] + torch.tensor([3., 4.])
        prediction[:, 2:] = 1e6
        sums, counts = metric_sums(prediction, batch)
        result = finish_metric(sums, counts)
        self.assertAlmostEqual(result["ade_m"], 5)
        self.assertAlmostEqual(result["fde_3s_m"], 5)
        loss = masked_loss(prediction, batch["target"], batch["target_mask"])
        self.assertAlmostEqual(float(loss), 3.0)
        bad = batch["target_mask"].clone()
        bad[:] = False
        with self.assertRaises(ValueError):
            masked_loss(prediction, batch["target"], bad)

    def test_social_is_permutation_equivariant_and_padding_safe(self):
        torch.manual_seed(42)
        batch = tensor_batch(make_window(raw_tracks(), 60))
        model = TrajectoryForecaster(width=32, layers=1).eval()
        torch.nn.init.normal_(model.head[-1].weight, std=.01)
        permutation = torch.randperm(22)
        permuted = {k: v[:, permutation] for k, v in batch.items()}
        with torch.no_grad():
            prediction, _ = model(batch)
            other, _ = model(permuted)
            torch.testing.assert_close(other, prediction[:, permutation], atol=1e-5, rtol=1e-5)
            padded_changed = {k: v.clone() for k, v in batch.items()}
            padded_changed["current"][:, 2:] = 1e4
            changed, _ = model(padded_changed)
            torch.testing.assert_close(changed[:, :2], prediction[:, :2], atol=1e-5, rtol=1e-5)

    def test_independent_has_no_other_player_input(self):
        batch = tensor_batch(make_window(raw_tracks(), 60))
        model = TrajectoryForecaster(width=32, layers=1, social=False).eval()
        torch.nn.init.normal_(model.head[-1].weight, std=.01)
        other = {k: v.clone() for k, v in batch.items()}
        other["current"][:, 1] += 30
        other["history"][:, 1] *= 4
        with torch.no_grad():
            a, _ = model(batch)
            b, _ = model(other)
            torch.testing.assert_close(a[:, 0], b[:, 0])

    def test_static_and_short_history_exclude_earlier_velocity(self):
        batch = tensor_batch(make_window(raw_tracks(), 60))
        for history_frames in (0, 10):
            model = TrajectoryForecaster(width=32, layers=1, history_frames=history_frames).eval()
            torch.nn.init.normal_(model.head[-1].weight, std=.01)
            other = {k: v.clone() for k, v in batch.items()}
            first = -(history_frames + 1)
            other["history"][:, :, :first] += 100
            other["history"][:, :, first, 2:4] += 100
            if not history_frames:
                other["velocity"] += 100
            with torch.no_grad():
                a, _ = model(batch)
                b, _ = model(other)
                torch.testing.assert_close(a, b)

    def test_bas_pass_subset_uses_action_schema_and_first_frame_time(self):
        import json
        class Windows:
            match_names = [("117092", "1st"), ("117092", "2nd")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "117092_12_class_events.json"
            path.write_text(json.dumps({
                "fps": 25, "clock": "ms from the first frame of the half's released video",
                "actions": [{"gameTime": "1 - 0:01", "label": "PASS", "position": "1233"},
                            {"gameTime": "2 - 0:01", "label": "PASS", "position": "2000"}],
            }), encoding="utf-8")
            frames = pass_frames(directory, Windows())
            np.testing.assert_array_equal(frames[0], [31])
            np.testing.assert_array_equal(frames[1], [50])

    def test_streaming_schema_duplicate_and_missing_ids(self):
        import json
        data = {
            "info": {"frame_rate": 25, "seq_length": 3},
            "images": [{"image_id": f"390100000{i}", "file_name": f"{i:06d}.jpg"} for i in (1, 2, 3)],
            "annotations": [],
        }
        for frame, player in ((1, 8), (2, 8), (2, 8), (3, None)):
            data["annotations"].append({
                "image_id": f"390100000{frame}", "supercategory": "object", "track_id": 4,
                "attributes": {"player_id": player, "team": "left", "role": "player"},
                "bbox_pitch": {"x_bottom_middle": 0, "y_bottom_middle": 1},
            })
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "gsr.json", Path(directory) / "cache.npz"
            source.write_text(json.dumps(data), encoding="utf-8")
            stats = build_half(source, target)["quality"]
            self.assertEqual(stats["duplicate_identity_records"], 1)
            self.assertEqual(stats["missing_identity"], 1)
            with np.load(target) as raw:
                self.assertEqual(raw["observed"].sum(), 1)


if __name__ == "__main__":
    unittest.main()
