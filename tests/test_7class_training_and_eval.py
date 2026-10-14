import unittest
from collections import Counter

from src.evaluate_action_model import (
    INVALID,
    LEGACY_LABELS,
    PRIMARY_LABELS,
    canonical_key,
    compute_metrics,
    max_new_tokens_arg,
    normalize_prediction,
    resolve_labels,
)
from src.recode_primary_actions import SOURCE_TO_PRIMARY_ACTION
from train.train_temporal_7class_mild_balance import (
    LABELS as TRAIN_LABELS,
    check_output_dir,
    git_metadata,
    sample_training_rows,
    sha256_file,
    validate_labels,
)


def rows_for(label, count):
    return [
        {"id": f"{label}-{i}", "match_id": "m", "prompt": "p", "response": label}
        for i in range(count)
    ]


class SevenClassConsistencyTests(unittest.TestCase):
    def test_label_lists_agree_with_recoder_output(self):
        expected = set(SOURCE_TO_PRIMARY_ACTION.values())
        self.assertEqual(set(PRIMARY_LABELS), expected)
        self.assertEqual(set(TRAIN_LABELS), expected)
        self.assertEqual(len(PRIMARY_LABELS), 7)

    def test_sampler_is_seeded_sized_and_inverse_sqrt(self):
        rows = rows_for("PASS", 400) + rows_for("SHOT", 100)
        a, counts = sample_training_rows(rows, len(rows), 42)
        b, _ = sample_training_rows(rows, len(rows), 42)
        c, _ = sample_training_rows(rows, len(rows), 43)
        self.assertEqual(len(a), len(rows))
        self.assertEqual(counts, {"PASS": 400, "SHOT": 100})
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
        self.assertNotEqual([r["id"] for r in a], [r["id"] for r in c])
        big, _ = sample_training_rows(rows, 20000, 42)
        share = Counter(r["response"] for r in big)["PASS"] / len(big)
        self.assertAlmostEqual(share, 2 / 3, delta=0.02)

    def test_train_label_validation_requires_exactly_seven_classes(self):
        good = [row for label in TRAIN_LABELS for row in rows_for(label, 2)]
        validate_labels(good, "train")
        with self.assertRaisesRegex(ValueError, "unexpected=\\['Goal'\\]"):
            validate_labels(good + rows_for("Goal", 1), "train")
        with self.assertRaisesRegex(ValueError, "missing=\\['HEADER'\\]"):
            validate_labels([r for r in good if r["response"] != "HEADER"], "val")


class RunSafetyTests(unittest.TestCase):
    def test_output_dir_protection_and_resume(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            check_output_dir(out, None)
            out.mkdir()
            check_output_dir(out, None)
            (out / "x").write_text("1")
            with self.assertRaises(FileExistsError):
                check_output_dir(out, None)
            ckpt = out / "checkpoint-2"
            ckpt.mkdir()
            check_output_dir(out, str(ckpt))
            with self.assertRaises(FileNotFoundError):
                check_output_dir(out, str(out / "checkpoint-9"))

    def test_sha256_and_git_metadata(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
        info = git_metadata()
        self.assertEqual(set(info), {"commit", "dirty"})


class EvaluatorLabelTests(unittest.TestCase):
    def test_canonical_forms_all_normalize_to_seven_labels(self):
        cases = {
            "PASS": "PASS",
            "high pass": "HIGH_PASS",
            "High_Pass": "HIGH_PASS",
            "HIGH-PASS": "HIGH_PASS",
            " drive\n": "DRIVE",
            "cross": "CROSS",
            "Shot": "SHOT",
            "header": "HEADER",
            "defensive intervention": "DEFENSIVE_INTERVENTION",
            "DEFENSIVE_INTERVENTION": "DEFENSIVE_INTERVENTION",
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_prediction(raw, PRIMARY_LABELS), expected, raw)

    def test_matching_is_exact_not_substring(self):
        for raw in (
            "DEFENSIVE",
            "DEFENSIVE_INTER",
            "PASS then DRIVE",
            "HIGH_PASS_EXTRA",
            "PASSING",
            "",
        ):
            self.assertEqual(normalize_prediction(raw, PRIMARY_LABELS), INVALID, raw)

    def test_truncated_defensive_label_would_be_invalid(self):
        self.assertEqual(
            normalize_prediction("DEFENSIVE_INTERVEN", PRIMARY_LABELS), INVALID
        )

    def test_resolve_labels_infers_seven_or_legacy_and_fails_otherwise(self):
        self.assertEqual(resolve_labels(["PASS", "DRIVE"]), PRIMARY_LABELS)
        self.assertEqual(resolve_labels(["Pass", "Goal"]), LEGACY_LABELS)
        with self.assertRaisesRegex(ValueError, "known label set"):
            resolve_labels(["PASS", "Goal"])
        with self.assertRaisesRegex(ValueError, "outside --labels"):
            resolve_labels(["PASS", "Out"], PRIMARY_LABELS)

    def test_max_new_tokens_must_be_at_least_sixteen(self):
        self.assertEqual(max_new_tokens_arg("16"), 16)
        with self.assertRaises(Exception):
            max_new_tokens_arg("8")

    def test_canonical_key(self):
        self.assertEqual(canonical_key(" High  pass "), "HIGH_PASS")

    def test_metrics_report_invalid_rate_and_distribution(self):
        y_true = ["PASS", "PASS", "DRIVE", "SHOT"]
        y_pred = ["PASS", INVALID, "DRIVE", "PASS"]
        metrics = compute_metrics(y_true, y_pred, PRIMARY_LABELS)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["invalid_rate"], 0.25)
        self.assertEqual(metrics["prediction_distribution"]["PASS"], 2)
        self.assertEqual(metrics["classes_predicted"], 2)
        self.assertEqual(set(metrics["per_class_f1"]), set(PRIMARY_LABELS))
        self.assertEqual(metrics["per_class_f1"]["SHOT"], 0.0)


if __name__ == "__main__":
    unittest.main()
