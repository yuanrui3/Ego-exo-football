import unittest

from src.recode_primary_actions import recode_records


class RecodePrimaryActionsTests(unittest.TestCase):
    def test_maps_merges_excludes_and_preserves_record_metadata(self):
        labels = (
            "Pass",
            "High Pass",
            "Drive",
            "Cross",
            "Shot",
            "Header",
            "Ball Player Block",
            "Player Successful Tackle",
            "Goal",
            "Out",
            "Throw In",
            "Free Kick",
        )
        records = [
            {
                "id": f"example-{index}",
                "match_id": "match-a",
                "prompt": f"prompt {index}",
                "response": label,
            }
            for index, label in enumerate(labels)
        ]

        output, counts, excluded_count = recode_records(records)

        self.assertEqual(
            [record["response"] for record in output],
            [
                "PASS",
                "HIGH_PASS",
                "DRIVE",
                "CROSS",
                "SHOT",
                "HEADER",
                "DEFENSIVE_INTERVENTION",
                "DEFENSIVE_INTERVENTION",
            ],
        )
        self.assertEqual(output[0]["source_label"], "Pass")
        self.assertEqual(output[6]["source_label"], "Ball Player Block")
        self.assertEqual(output[7]["source_label"], "Player Successful Tackle")
        self.assertEqual(counts["DEFENSIVE_INTERVENTION"], 2)
        self.assertEqual(excluded_count, 4)
        self.assertEqual(
            [record["prompt"] for record in output],
            [record["prompt"] for record in records[:8]],
        )
        self.assertEqual(
            [record["match_id"] for record in output],
            ["match-a"] * 8,
        )
        self.assertEqual(
            [record["id"] for record in output],
            [record["id"] for record in records[:8]],
        )

    def test_rejects_unexpected_labels(self):
        with self.assertRaisesRegex(ValueError, "Unexpected source action labels"):
            recode_records([{"response": "Unknown"}])


if __name__ == "__main__":
    unittest.main()
