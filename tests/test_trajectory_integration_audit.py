import copy
import unittest

import numpy as np

from src.audit_trajectory_integration import (
    INPUT_KEYS, reconstruct_history, reconstruct_sample, selection_arrays, selection_summary, verify_sparse_raw,
)
from src.trajectory_dataset import make_window


def example_raw(frames=180, players=22):
    xy = np.zeros((frames, players, 2), np.float32)
    xy[..., 0] = np.arange(frames, dtype=np.float32)[:, None] * .04
    xy[..., 1] = np.arange(players, dtype=np.float32)
    return {
        "xy": xy, "observed": np.ones((frames, players), bool),
        "segments": np.ones((frames, players), np.int32),
        "teams": np.tile(np.array([1] * 11 + [2] * 11, np.int8), (frames, 1)),
        "roles": np.zeros((frames, players), np.int8),
    }


class IntegrationAuditTests(unittest.TestCase):
    def test_inference_history_needs_no_future_arrays_or_targets(self):
        raw = example_raw()
        truncated = {key: value[:81] for key, value in raw.items()}
        full, _, _ = reconstruct_history(raw, 80)
        causal, _, _ = reconstruct_history(truncated, 80)
        self.assertEqual(set(causal), set(INPUT_KEYS))
        for key in INPUT_KEYS:
            np.testing.assert_array_equal(full[key], causal[key])

    def test_independent_reconstruction_and_exact_offsets(self):
        raw = example_raw()
        independent, _, _ = reconstruct_sample(raw, 80)
        production = make_window(raw, 80)
        for key in production:
            np.testing.assert_allclose(independent[key], production[key], atol=1e-5)
        np.testing.assert_allclose(independent["target"][0, :, 0], [.4, 1, 2, 3], atol=1e-5)
        self.assertEqual(independent["history"].shape[1], 51)

    def test_future_eligibility_is_selection_not_input_feature(self):
        raw = example_raw()
        changed = copy.deepcopy(raw)
        changed["observed"][125] = False
        changed["segments"][125] = 0
        self.assertTrue(selection_arrays(raw)["accepted"][0])
        self.assertFalse(selection_arrays(changed)["accepted"][0])
        a, _, _ = reconstruct_sample(raw, 50)
        b, _, _ = reconstruct_sample(changed, 50)
        for key in INPUT_KEYS:
            np.testing.assert_array_equal(a[key], b[key])

    def test_exclusion_rules_partition_candidates(self):
        raw = example_raw()
        raw["observed"][50, 0:6] = False
        raw["teams"][63, 11:13] = 1
        raw["observed"][151, :12] = False
        raw["segments"][151, :12] = 0
        stats, _ = selection_summary(raw)
        self.assertEqual(stats["excluded_player_count_sequential"], 1)
        self.assertEqual(stats["excluded_team_count_sequential"], 1)
        self.assertEqual(stats["excluded_future_continuity_sequential"], 1)
        self.assertEqual(
            stats["eligible"] + stats["excluded_player_count_sequential"]
            + stats["excluded_team_count_sequential"] + stats["excluded_future_continuity_sequential"],
            stats["candidate_windows"],
        )

    def test_raw_slot_mapping_verification_detects_wrong_identity(self):
        raw = example_raw()
        states = {50: [{"player_id": "a", "x": 2., "y": 0., "team": "left", "role": "player"}]}
        verified = verify_sparse_raw(raw, {"a": 0}, states)
        self.assertEqual(verified["raw_player_frames_checked"], 1)
        with self.assertRaises(AssertionError):
            verify_sparse_raw(raw, {"a": 1}, states)


if __name__ == "__main__":
    unittest.main()
