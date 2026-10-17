import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.receiver_forecast_ranker import ReceiverForecastRanker, feature_statistics
from src.receiver_forecast_adapter import (
    FORECAST_NAMES, PRESSURE_INDICES, MOVEMENT_INDICES, anonymous_inputs, asof_base,
    assert_out_of_fold, build_history_inputs, config_hash, forecast_features, select_forecast_group,
)
from src.trajectory_dataset import TRAIN_MATCHES, VALIDATION_MATCHES
from train.run_receiver_forecast_integration import (
    fold_sets, forecast_config, metric_from_logits, receiver_config,
)


def raw():
    frames, players = 180, 22
    xy = np.zeros((frames, players, 2), np.float32)
    xy[..., 0] = np.arange(frames, dtype=np.float32)[:, None] * .04 + np.arange(players)
    xy[..., 1] = np.arange(players) * .2
    return {
        "xy": xy, "observed": np.ones((frames, players), bool),
        "segments": np.ones((frames, players), np.int32),
        "teams": np.tile([1] * 11 + [2] * 11, (frames, 1)),
        "roles": np.zeros((frames, players), np.int64),
    }


def feature_inputs():
    current = torch.randn(3, 22, 2)
    velocity = torch.randn(3, 22, 2)
    times = torch.tensor([.4, 1, 2, 3])
    future = current[:, :, None] + velocity[:, :, None] * times[None, None, :, None]
    return current, future, torch.ones(3, 22, dtype=torch.bool), torch.ones(3, 22), torch.ones(3, dtype=torch.bool)


class ReceiverForecastTests(unittest.TestCase):
    def test_in_sample_and_foreign_split_rejected(self):
        with self.assertRaises(ValueError):
            assert_out_of_fold("117092", ["117092", "117093"])
        with self.assertRaises(ValueError):
            assert_out_of_fold("118577", ["117092", "128058"])
        with self.assertRaises(ValueError):
            assert_out_of_fold("118577", ["117092"], ["117093"])
        assert_out_of_fold("118577", TRAIN_MATCHES, ["132877"])

    def test_all_fold_fitting_and_stopping_are_match_excluded(self):
        for stage in ("development", "final"):
            predicted = TRAIN_MATCHES if stage == "development" else (*TRAIN_MATCHES, *VALIDATION_MATCHES)
            for match in predicted:
                fits, inner = fold_sets(stage, match)
                self.assertNotIn(match, fits)
                self.assertIn(inner, fits)
                if stage == "development":
                    self.assertNotIn("132877", fits)
                assert_out_of_fold(match, fits, [inner])
        fitsets = {
            tuple(fold_sets(stage, match)[0]) for stage in ("development", "final")
            for match in (TRAIN_MATCHES if stage == "development" else (*TRAIN_MATCHES, *VALIDATION_MATCHES))
        }
        self.assertEqual(len(fitsets), 14)

    def test_adapter_ignores_future_and_has_no_target_keys(self):
        a = raw()
        b = copy.deepcopy(a)
        for key in ("xy", "observed", "segments", "teams", "roles"):
            b[key][81:] = 0
        x, selected, _ = build_history_inputs(a, 80)
        y, _, _ = build_history_inputs(b, 80)
        self.assertNotIn("target", x)
        self.assertNotIn("target_mask", x)
        self.assertEqual(x["history"].shape, (22, 51, 5))
        np.testing.assert_allclose(x["velocity"][:, 0], 1, atol=1e-5)
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])

    def test_storage_permutation_does_not_change_anonymous_features(self):
        original = raw()
        inputs, selected, _ = build_history_inputs(original, 80)
        names = ("ACTOR",) + tuple(f"TEAMMATE_{i}" for i in range(1, 11)) + tuple(f"OPPONENT_{i}" for i in range(1, 12))
        mapping = dict(zip(names, range(22)))
        a = anonymous_inputs(inputs, selected, mapping)
        perm = np.random.default_rng(42).permutation(22)
        reordered = {key: value[:, perm] for key, value in original.items()}
        new_inputs, new_selected, _ = build_history_inputs(reordered, 80)
        reverse = np.argsort(perm)
        new_mapping = {name: int(reverse[slot]) for name, slot in mapping.items()}
        b = anonymous_inputs(new_inputs, new_selected, new_mapping)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        outputs = []
        for ordered in (a, b):
            current = torch.from_numpy(ordered["current"])[None]
            times = torch.tensor([.4, 1, 2, 3])
            future = current[:, :, None] + torch.from_numpy(ordered["velocity"])[None, :, None] * times[None, None, :, None]
            outputs.append(forecast_features(
                current, future, torch.from_numpy(ordered["player_mask"])[None],
                torch.from_numpy(ordered["history"][..., 4].mean(-1))[None], torch.tensor([True])
            ))
        torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
        self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))

    def test_forecast_horizons_and_pressure_geometry(self):
        current = torch.zeros(1, 22, 2)
        current[:, 1:11, 0] = 10
        current[:, 11:, 0] = 15
        future = current[:, :, None].expand(-1, -1, 4, -1).clone()
        future[:, 1:11, :, 0] += torch.tensor([.4, 1, 2, 3])
        args = (current, future, torch.ones(1, 22, dtype=torch.bool), torch.ones(1, 22), torch.tensor([True]))
        features, valid = forecast_features(*args)
        self.assertEqual(features.shape, (1, 10, 63))
        for h, time in enumerate((.4, 1., 2., 3.)):
            self.assertAlmostEqual(float(features[0, 0, h * 15]), time, places=5)
            self.assertAlmostEqual(float(features[0, 0, h * 15 + 8]), 5 - time, places=5)
        self.assertTrue(valid.all())

    def test_missing_defenders_no_infinity_or_invented_space(self):
        args = list(feature_inputs())
        args[2][:, 11:] = False
        f, v = forecast_features(*args)
        self.assertTrue(torch.isfinite(f).all())
        self.assertFalse(v[..., list(PRESSURE_INDICES)].any())
        self.assertEqual(float(f[..., list(PRESSURE_INDICES)].abs().sum()), 0)

    def test_missing_candidates_and_forecast_unavailability(self):
        args = list(feature_inputs())
        args[2][:, 1] = False
        args[4][1] = False
        f, v = forecast_features(*args)
        self.assertFalse(v[:, 0].any())
        self.assertFalse(v[1, :, :-1].any())
        self.assertTrue(torch.isfinite(f).all())

    def test_defender_storage_permutation_is_invariant(self):
        args = list(feature_inputs())
        a, mask_a = forecast_features(*args)
        perm = torch.cat((torch.arange(11), 11 + torch.randperm(11)))
        b, mask_b = forecast_features(
            args[0][:, perm], args[1][:, perm], args[2][:, perm], args[3][:, perm], args[4]
        )
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.equal(mask_a, mask_b))

    def test_group_ablation_removes_explicit_features(self):
        f, v = forecast_features(*feature_inputs())
        e, _ = select_forecast_group(f, v, "E")
        short, _ = select_forecast_group(f, v, "F")
        self.assertEqual(e.shape[-1], 63 - len(PRESSURE_INDICES))
        self.assertEqual(short.shape[-1], 63 - len(MOVEMENT_INDICES))
        self.assertFalse(set(PRESSURE_INDICES) & set(MOVEMENT_INDICES))

    def test_training_only_statistics_and_unobserved_mask(self):
        values = torch.arange(120, dtype=torch.float32).view(3, 10, 4)
        valid = torch.ones_like(values, dtype=torch.bool)
        valid[0, 0] = False
        mean, std = feature_statistics(values[:2], valid[:2])
        values[2] = 1e9
        other_mean, other_std = feature_statistics(values[:2], valid[:2])
        torch.testing.assert_close(mean, other_mean)
        torch.testing.assert_close(std, other_std)
        model = ReceiverForecastRanker(mean, std, 32).eval()
        candidate = torch.ones(3, 10, dtype=torch.bool)
        candidate[:, -1] = False
        with torch.no_grad():
            scores = model(values, valid, candidate)
        self.assertTrue((scores[:, -1] == -10000).all())

    def test_asof_reorders_without_future_and_relabels_receiver(self):
        source = raw()
        row = {"passer_id": "a", "receiver_id": "r"}
        slots = {"a": 0, "r": 5}
        state, mapping, target = asof_base(source, slots, row, 80)
        self.assertEqual(mapping["ACTOR"], 0)
        self.assertEqual(mapping[f"TEAMMATE_{target+1}"], 5)
        changed = copy.deepcopy(source)
        changed["xy"][81:] = 1e9
        other = asof_base(changed, slots, row, 80)
        np.testing.assert_array_equal(state, other[0])
        self.assertEqual(mapping, other[1])

    def test_hash_reproducibility_and_rank_metrics(self):
        self.assertEqual(config_hash({"a": 1, "b": 2}), config_hash({"b": 2, "a": 1}))
        scores = torch.zeros(2, 10)
        scores[0, 4], scores[1, 8] = 10, 10
        metrics = metric_from_logits(scores, torch.tensor([4, 8]))
        self.assertEqual(metrics["top1"], 1)
        self.assertEqual(metrics["mrr"], 1)


if __name__ == "__main__":
    unittest.main()
