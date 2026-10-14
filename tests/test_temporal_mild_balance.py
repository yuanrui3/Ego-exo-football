import unittest

from train.train_temporal_mild_balance import sample_training_rows


def rows_for_label(label, count):
    return [
        {
            "id": f"{label}-{index}",
            "match_id": "match",
            "prompt": "prompt",
            "response": label,
        }
        for index in range(count)
    ]


class TemporalMildBalanceTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_uses_inverse_sqrt_class_weights(self):
        rows = rows_for_label("Pass", 100) + rows_for_label("Shot", 25)
        sample_a, counts = sample_training_rows(rows, 6000, seed=42)
        sample_b, _ = sample_training_rows(rows, 6000, seed=42)

        self.assertEqual(counts, {"Pass": 100, "Shot": 25})
        self.assertEqual([row["id"] for row in sample_a], [row["id"] for row in sample_b])
        sampled_pass = sum(row["response"] == "Pass" for row in sample_a)
        self.assertAlmostEqual(sampled_pass / len(sample_a), 2 / 3, delta=0.02)


if __name__ == "__main__":
    unittest.main()
