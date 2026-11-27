from src import synthetic_goalkeeper_pilot as pilot

import json
from pathlib import Path
import unittest
from unittest import mock
import uuid

import numpy as np


class CausalPretrainingTests(unittest.TestCase):
    def setUp(self):
        ident = uuid.uuid4().hex
        self.paths = [Path(f"../goalkeeper-fixture-{ident}-{i}.json") for i in range(3)]
        self.assertFalse(any(path.exists() for path in self.paths), "fixture paths already exist")
        self.addCleanup(self.clean_files)

    def clean_files(self):
        for path in self.paths:
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".pending").unlink(missing_ok=True)

    def test_future_and_masked_mutation_never_change_prefix(self):
        history = pilot.generate_history(0, 0)
        observed = pilot.contiguous_mask(3, family=0, scene=0)
        before = pilot.prefix_features(history[:4], observed[:4])
        baseline = pilot.cv_prediction(history[:4], observed[:4])
        changed = history.copy()
        changed[~observed] = 1e9
        np.testing.assert_array_equal(before, pilot.prefix_features(changed[:4], observed[:4]))
        self.assertEqual(baseline, pilot.cv_prediction(changed[:4], observed[:4]))
        future_changed = history.copy()
        future_changed[4:] = 1e9
        np.testing.assert_array_equal(pilot.visibility_mask(history[:4]),
                                      pilot.visibility_mask(future_changed[:4]))

    def test_contiguous_mask_and_target_never_in_features(self):
        for end in range(1, pilot.HISTORY - 1):
            for scene in range(30):
                observed = pilot.contiguous_mask(end, family=0, scene=scene)
                hidden = np.flatnonzero(~observed[:end + 1])
                self.assertTrue(len(hidden) and np.all(np.diff(hidden) == 1))
                self.assertFalse(observed[end + 1:].any())
                history = pilot.generate_history(0, scene)
                x = pilot.prefix_features(history[:end + 1], observed[:end + 1])
                self.assertTrue(np.all(x[1 + end + 1:1 + pilot.HISTORY] == 0))

    def test_cv_uses_observation_times_and_no_hidden_values(self):
        history = np.array([1.0, 1000.0, 3.0, 1000.0])
        observed = np.array([True, False, True, False])
        self.assertEqual(pilot.cv_prediction(history, observed), 5.0)
        self.assertEqual(pilot.cv_prediction(history, np.zeros(4, dtype=bool)), 0.0)

    def test_ridge_converges_on_noiseless_linear_fixture(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(50000, pilot.FEATURES))
        x[:, 0] = 1
        expected = rng.normal(size=pilot.FEATURES)
        estimated = pilot.fit_moments(x.T @ x, x.T @ (x @ expected))
        np.testing.assert_allclose(estimated, expected, atol=0.0002)

    def test_resume_matches_uninterrupted_and_never_touches_reserved(self):
        original = pilot.generate_history
        seen = set()

        def checked(family, scene):
            self.assertNotIn(family, pilot.RESERVED_HELD_FAMILIES)
            seen.add(family)
            return original(family, scene)

        with mock.patch.object(pilot, "generate_history", side_effect=checked):
            partial = pilot.run(self.paths[0], max_new_families=3)
            self.assertEqual(partial["families_completed"], 3)
            self.assertIsNone(partial["pretraining_receipt"])
            done = pilot.run(self.paths[0])
            full = pilot.run(self.paths[1])
        self.assertEqual(done, full)
        self.assertEqual(done["examples"], len(pilot.TRAIN_FAMILIES) * pilot.TRAIN_SCENES * (pilot.HISTORY - 2))
        self.assertEqual(done["development"]["examples"], len(pilot.DEV_FAMILIES) * pilot.DEV_SCENES * (pilot.HISTORY - 2))
        self.assertEqual(seen, set(pilot.TRAIN_FAMILIES) | set(pilot.DEV_FAMILIES))
        self.assertEqual(done["pretraining_receipt"]["sha256"],
                         pilot._digest(done["pretraining_receipt"]["contents"]))

    def test_checkpoint_tamper_and_protocol_drift_rejected(self):
        path = self.paths[0]
        pilot.run(path, max_new_families=1)
        envelope = json.loads(path.read_text())
        envelope["state"]["examples"] = -1
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "checksum"):
            pilot.run(path)
        path.unlink()
        pilot.run(path, max_new_families=1)
        with mock.patch.object(pilot, "RIDGE", 2.0):
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "mismatch"):
                pilot.run(path)
            self.assertEqual(before, path.read_bytes())

    def test_source_change_during_fit_never_publishes_receipt(self):
        path = self.paths[0]
        pilot.run(path, max_new_families=1)
        changed = False
        fit = pilot.fit_moments

        def mutate_source_during_fit(xtx, xty):
            nonlocal changed
            result = fit(xtx, xty)
            changed = True
            return result

        with mock.patch.object(pilot, "fit_moments", side_effect=mutate_source_during_fit), \
             mock.patch.object(pilot, "_source_hash",
                               side_effect=lambda: "changed" if changed else pilot._IMPORT_SOURCE_HASH):
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                pilot.run(path)
            saved = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                pilot.run(path)
            self.assertEqual(saved, path.read_bytes())
        state = json.loads(saved)["state"]
        self.assertEqual(state["families_completed"], len(pilot.TRAIN_FAMILIES))
        self.assertIsNone(state["pretraining_receipt"])
        self.assertIsNotNone(pilot.run(path)["pretraining_receipt"])

    def test_atomic_write_ignores_predictable_symlink_and_cleans_up(self):
        checkpoint, target = self.paths[:2]
        target.write_text("untouched")
        link = checkpoint.with_name(checkpoint.name + ".pending")
        link.symlink_to(target.resolve())
        pilot.run(checkpoint, max_new_families=0)
        self.assertEqual(target.read_text(), "untouched")
        self.assertTrue(link.is_symlink())
        with mock.patch.object(pilot.os, "replace", side_effect=OSError("test failure")):
            with self.assertRaisesRegex(OSError, "test failure"):
                pilot._atomic_write(checkpoint, {"test": 1})
        self.assertEqual(list(checkpoint.parent.glob(f".{checkpoint.name}.*.pending")), [])

    def test_reserved_families_and_in_tree_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "only training and development"):
            pilot.generate_history(pilot.RESERVED_HELD_FAMILIES[0], 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            pilot.run("fixture-checkpoint.json")


if __name__ == "__main__":
    unittest.main()
