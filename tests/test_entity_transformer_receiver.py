import unittest

import torch

from src.evaluate_pass_receiver_model import LABELS, ranking_metrics
from src.models.entity_transformer_receiver import EntityTransformerReceiver
from src.receiver_tensor_dataset import (
    ENTITIES, TRAIN_MATCHES, ReceiverTensorDataset, development_indices,
    parse_receiver_prompt, shuffle_coordinates, target_index, validate_match_split,
)
from train.train_entity_transformer_receiver import (
    evaluate, make_loader, select_best_epoch, train_epoch,
)


def prompt(missing=False, reverse=False):
    blocks = []
    for time, header in enumerate(("STATE_T_MINUS_0.4S", "STATE_T_MINUS_0.2S", "STATE_T")):
        lines = []
        for index, label in enumerate(ENTITIES):
            team = "right" if index < 11 else "left"
            role = "goalkeeper" if index in (10, 21) else "player"
            xy = "x=NA y=NA" if missing and time == 0 and index == 2 else (
                f"x={index + time:.2f} y={index - time:.2f}"
            )
            lines.append(f"{label} team={team} role={role} {xy}")
        blocks.append(header + "\n" + "\n".join(reversed(lines) if reverse else lines))
    return "<|system|>\nreceiver instruction\n<|user|>\nPERIOD 1\n" + (
        "\n\n".join(blocks) + "\n<|assistant|>\n"
    )


class TensorParserTests(unittest.TestCase):
    def test_shape_features_and_identity_slots(self):
        features = parse_receiver_prompt(prompt(reverse=True))
        self.assertEqual(tuple(features.shape), (3, 22, 8))
        torch.testing.assert_close(features, parse_receiver_prompt(prompt()))
        torch.testing.assert_close(features[:, 2, 0], torch.tensor([2, 3, 4]) / 52.5)
        self.assertEqual(features[0, 0, 3].item(), 1)
        self.assertEqual(features[0, 2, 4].item(), 1)
        self.assertEqual(features[0, 21, 5].item(), 1)
        self.assertEqual(features[0, 21, 6].item(), 1)
        self.assertEqual(features[0, 2, 7].item(), 1)

    def test_na_masking(self):
        features = parse_receiver_prompt(prompt(missing=True))
        torch.testing.assert_close(features[0, 2, :3], torch.zeros(3))
        self.assertEqual(features[1, 2, 2].item(), 1)

    def test_absent_slot_padded_not_reindexed(self):
        text = "\n".join(line for line in prompt().splitlines() if not line.startswith("OPPONENT_11 "))
        features = parse_receiver_prompt(text)
        self.assertEqual(features[:, 21, 2].sum().item(), 0)
        self.assertEqual(features[:, 20, 2].sum().item(), 3)

    def test_bad_temporal_structure_rejected(self):
        for text in (
            prompt().replace("STATE_T_MINUS_0.4S", "STATE_T_PLUS_0.4S"),
            prompt().replace("x=0.00 y=0.00", "x=NA y=0.00", 1),
            prompt().replace("TEAMMATE_1 team=right", "TEAMMATE_2 team=right", 1),
            prompt().replace("TEAMMATE_1 team=right", "TEAMMATE_1 team=left", 1),
            prompt() + "TEAMMATE_1",
        ):
            with self.subTest(text=text[:40]), self.assertRaises(ValueError):
                parse_receiver_prompt(text)

    def test_no_metadata_or_permanent_ids_in_features(self):
        first = {"id": "111111", "match_id": "132877", "passer_id": "99999",
                 "receiver_id": "88888", "gap_frames": 1, "next_action": "DRIVE",
                 "prompt": prompt(), "response": "TEAMMATE_2"}
        second = {**first, "id": "222222", "match_id": "118577", "passer_id": "12345",
                  "receiver_id": "54321", "gap_frames": 50, "next_action": "SHOT"}
        data = ReceiverTensorDataset([first, second])
        torch.testing.assert_close(data[0][0], data[1][0])
        self.assertEqual(len(data[0]), 2)
        with self.assertRaises(ValueError):
            parse_receiver_prompt(prompt().replace("role=player", "role=player player_id=12345", 1))

    def test_all_targets_map_to_zero_based_indices(self):
        self.assertEqual([target_index(label) for label in LABELS], list(range(10)))
        for label in ("TEAMMATE_0", "TEAMMATE_11", "PASS"):
            with self.assertRaises(ValueError):
                target_index(label)

    def test_development_split_by_match(self):
        rows = [{"id": f"{match}_{i}", "match_id": match} for i in range(2) for match in TRAIN_MATCHES]
        train, dev = development_indices(rows)
        self.assertEqual({rows[i]["match_id"] for i in dev}, {"132877"})
        self.assertEqual({rows[i]["match_id"] for i in train}, set(TRAIN_MATCHES) - {"132877"})
        self.assertFalse(set(train) & set(dev))
        self.assertEqual(len(train) + len(dev), len(rows))
        validate_match_split(rows, [{"id": "v1", "match_id": "118577"}, {"id": "v2", "match_id": "128058"}])
        with self.assertRaises(ValueError):
            development_indices(rows + [{"id": "v", "match_id": "118577"}])

    def test_coordinate_shuffle_reproducibility_and_pairs(self):
        features = parse_receiver_prompt(prompt(missing=True)).unsqueeze(0).repeat(2, 1, 1, 1)
        original = features.clone()
        shuffled = shuffle_coordinates(features, seed=42)
        torch.testing.assert_close(shuffled, shuffle_coordinates(features, seed=42))
        torch.testing.assert_close(features, original)
        torch.testing.assert_close(shuffled[..., 2:], features[..., 2:])
        self.assertFalse(torch.equal(shuffled, shuffle_coordinates(features, seed=7)))
        for ex in range(2):
            for time in range(3):
                observed = features[ex, time, :, 2].bool()
                before = sorted(map(tuple, features[ex, time, observed, :2].tolist()))
                after = sorted(map(tuple, shuffled[ex, time, observed, :2].tolist()))
                self.assertEqual(before, after)
        torch.testing.assert_close(shuffled[0, 0, 2, :2], torch.zeros(2))


class EntityModelTests(unittest.TestCase):
    def test_model_shape_and_finite_logits(self):
        model = EntityTransformerReceiver().eval()
        features = parse_receiver_prompt(prompt()).unsqueeze(0).repeat(2, 1, 1, 1)
        with torch.no_grad():
            logits = model(features)
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        features[:, :, 10, 2] = 0
        with torch.no_grad():
            self.assertTrue(bool((model(features)[:, 9] == -10000).all()))

    def test_training_and_evaluation_smoke(self):
        dataset = ReceiverTensorDataset([
            {"id": "a", "prompt": prompt(), "response": "TEAMMATE_2"},
            {"id": "b", "prompt": prompt(missing=True), "response": "TEAMMATE_10"},
        ])
        model = EntityTransformerReceiver()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loader = make_loader(dataset, 2, 42, shuffle=False)
        before = model.scoring_head[0].weight.detach().clone()
        loss = train_epoch(model, loader, optimizer, torch.device("cpu"))
        self.assertGreater(loss, 0)
        self.assertFalse(torch.equal(before, model.scoring_head[0].weight))
        metrics, records = evaluate(model, loader, torch.device("cpu"))
        expected = ranking_metrics(
            [record["true_target"] for record in records],
            [record["ranked_targets"] for record in records],
        )
        for key, value in expected.items():
            self.assertEqual(metrics[key], value)
        self.assertEqual(metrics["invalid_or_nonfinite_cases"], 0)
        self.assertEqual(sum(metrics["prediction_distribution"].values()), 2)

    def test_rank_metrics_and_selection_tie_break(self):
        ranked = [LABELS, list(reversed(LABELS))]
        metrics = ranking_metrics(["TEAMMATE_1", "TEAMMATE_1"], ranked)
        self.assertEqual(metrics["top1"], 0.5)
        self.assertEqual(metrics["top3"], 0.5)
        self.assertEqual(metrics["top5"], 0.5)
        self.assertAlmostEqual(metrics["mrr"], 0.55)
        self.assertEqual(metrics["mean_rank"], 5.5)
        history = [{"epoch": epoch, "development_mrr": score}
                   for epoch, score in [(1, 0.5), (2, 0.7), (3, 0.6), (4, 0.7)]]
        self.assertEqual(select_best_epoch(history), 2)


if __name__ == "__main__":
    unittest.main()
