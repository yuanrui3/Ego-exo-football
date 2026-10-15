import unittest

import torch

from src.models.candidate_ranker_receiver import (
    CONTINUOUS_COUNT, FEATURE_NAMES, CandidateRankerReceiver, candidate_features,
    fit_statistics, prior_adjustment,
)
from src.receiver_tensor_dataset import ReceiverTensorDataset, parse_receiver_prompt, shuffle_coordinates
from tests.test_entity_transformer_receiver import prompt
from train.train_candidate_ranker_receiver import select_configuration, train_epoch
from train.train_entity_transformer_receiver import evaluate, make_loader


class CandidateRankerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.states = parse_receiver_prompt(prompt()).unsqueeze(0).repeat(3, 1, 1, 1)

    def feature(self, values, name):
        return values[..., FEATURE_NAMES.index(name)]

    def test_engineered_geometry_motion_and_order(self):
        values, valid = candidate_features(self.states)
        self.assertEqual(values.shape, (3, 10, len(FEATURE_NAMES)))
        self.assertTrue(valid.all())
        torch.testing.assert_close(self.feature(values, "relative_dx_current")[0], torch.arange(1, 11).float())
        self.assertAlmostEqual(self.feature(values, "distance_current")[0, 0].item(), 2 ** 0.5, places=5)
        for entity in ("candidate", "actor"):
            self.assertAlmostEqual(self.feature(values, f"{entity}_displacement_x_full")[0, 0].item(), 2, places=5)
            self.assertAlmostEqual(self.feature(values, f"{entity}_velocity_y_full")[0, 0].item(), -5, places=5)
        torch.testing.assert_close(self.feature(values, "distance_change_full"), torch.zeros(3, 10), atol=1e-5, rtol=0)
        torch.testing.assert_close(self.feature(values, "distance_rank")[0], torch.arange(1, 11).float())
        self.assertEqual(self.feature(values, "candidate_goalkeeper")[0, -1].item(), 1)

    def test_opponent_features_exact(self):
        states = self.states[:1].clone()
        states[:, -1, 1, :2] = 0
        states[:, -1, 11:, 2] = 0
        states[:, -1, 11:14, 2] = 1
        states[:, -1, 11:14, :2] = torch.tensor([[3., 0], [5., 0], [10., 0]]) / torch.tensor([52.5, 34])
        values, _ = candidate_features(states)
        self.assertEqual(self.feature(values, "candidate_nearest_opponent")[0, 0].item(), 3)
        for radius, expected in ((3, 1), (5, 2), (10, 3)):
            self.assertEqual(self.feature(values, f"opponents_within_{radius}m")[0, 0].item(), expected)

    def test_missing_features_masks_and_finite_normalization(self):
        states = self.states.clone()
        states[:, 0, 1, :3] = 0
        states[:, -1, 11:, :3] = 0
        values, valid = candidate_features(states)
        for name in ("relative_dx_minus_0.4", "candidate_velocity_x_full", "candidate_nearest_opponent"):
            self.assertFalse(self.feature(valid, name)[0, 0])
            self.assertEqual(self.feature(values, name)[0, 0].item(), 0)
        stats = fit_statistics(states)
        self.assertTrue(all(torch.isfinite(v).all() for v in stats.values()))
        self.assertTrue(torch.isfinite(CandidateRankerReceiver(stats)(states)).all())

    def test_training_only_normalization(self):
        training = self.states.clone()
        training[1, :, 1:11, :2] *= 2
        stats = fit_statistics(training)
        saved = {k: v.clone() for k, v in stats.items()}
        model = CandidateRankerReceiver(stats).eval()
        heldout = training * 1
        heldout[..., :2] += 100
        with torch.inference_mode():
            model(heldout)
        for name in stats:
            torch.testing.assert_close(getattr(model, name), saved[name])
        values, valid = candidate_features(training)
        normalized = ((values - stats["feature_mean"]) / stats["feature_std"]).masked_fill(~valid, 0)
        torch.testing.assert_close(normalized[..., :CONTINUOUS_COUNT].mean((0, 1)),
                                   torch.zeros(CONTINUOUS_COUNT), atol=1e-4, rtol=0)

    def test_no_future_or_identity_information(self):
        row = {"id": "a", "prompt": prompt(), "response": "TEAMMATE_1", "receiver_id": "secret"}
        other = {**row, "id": "b", "receiver_id": "changed", "next_event": "SHOT", "gap_frames": 50}
        data = ReceiverTensorDataset([row, other])
        values, _ = candidate_features(data.features)
        torch.testing.assert_close(values[0], values[1])
        with self.assertRaises(ValueError):
            parse_receiver_prompt(prompt().replace("STATE_T", "STATE_T_PLUS_1", 1))

    def test_prior_math_and_selection(self):
        logits = torch.zeros(2, 10)
        frequencies = torch.arange(1, 11).float() / 55
        torch.testing.assert_close(prior_adjustment(logits, frequencies, 0.25),
                                   (-0.25 * frequencies.log()).expand(2, -1))
        torch.testing.assert_close(prior_adjustment(logits, frequencies, 0), logits)
        with self.assertRaises(ValueError):
            prior_adjustment(logits, torch.zeros(10), 0.5)
        configs = [{"alpha": a, "selected_epoch": 4, "best_development_mrr": score}
                   for a, score in ((0, .5), (.25, .6), (.5, .6))]
        self.assertEqual(select_configuration(configs)["alpha"], .25)

    def test_shape_mask_and_training_smoke(self):
        states = self.states.clone()
        states[:, :, 10, :3] = 0
        model = CandidateRankerReceiver(fit_statistics(states))
        self.assertEqual(model(states).shape, (3, 10))
        self.assertTrue((model(states)[:, -1] == -10000).all())
        loader = make_loader(torch.utils.data.TensorDataset(states, torch.tensor([0, 1, 2])), 3, 42, False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = train_epoch(model, loader, optimizer, "cpu", torch.ones(10) / 10, .25)
        self.assertGreater(loss, 0)
        metrics, records = evaluate(model, loader, "cpu")
        self.assertEqual(len(records), 3)
        self.assertEqual(metrics["invalid_or_nonfinite_cases"], 0)

    def test_shuffle_recomputed_features_deterministic(self):
        a, _ = candidate_features(shuffle_coordinates(self.states, 42))
        b, _ = candidate_features(shuffle_coordinates(self.states, 42))
        torch.testing.assert_close(a, b)
        original, _ = candidate_features(self.states)
        self.assertFalse(torch.equal(a, original))


if __name__ == "__main__":
    unittest.main()
