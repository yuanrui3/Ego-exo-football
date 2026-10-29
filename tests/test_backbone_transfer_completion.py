import numpy as np
import pytest

from src.backbone.transfer_v06 import (
    FOLDS, FIT_MATCHES, _history_features, validate_fold, validate_probabilities,
)


def test_match_excluded_folds_cover_six_fitting_matches_once():
    predicted = []
    for heldout in FOLDS:
        fitting = sorted(FIT_MATCHES - set(heldout))
        validate_fold(fitting, list(heldout))
        predicted.extend(heldout)
        with pytest.raises(ValueError, match="leaks"):
            validate_fold(fitting + [heldout[0]], list(heldout))
    assert len(predicted) == len(set(predicted)) == 6
    assert set(predicted) == FIT_MATCHES


def test_teacher_probability_validation_rejects_bad_rows():
    probabilities = np.full((3, 7), 1 / 7, np.float32)
    validate_probabilities(probabilities, 3)
    for invalid in (probabilities[:, :6], probabilities * 0.5,
                    np.full((3, 7), np.nan)):
        with pytest.raises(ValueError):
            validate_probabilities(invalid, 3)


def test_teacher_history_is_causal_and_ignores_labels_and_future():
    class Source:
        raw = {
            "xy": np.zeros((70, 3, 2), np.float32),
            "observed": np.ones((70, 3), bool),
            "segments": np.zeros((70, 3), np.int16),
            "teams": np.tile(np.array([1, 1, 2]), (70, 1)),
            "roles": np.zeros((70, 3), np.int8),
        }
        half = "1st"
    source = Source()
    source.raw["xy"][:, 1, 0] = np.arange(70) / 25
    metadata = {"sample_id": "test", "query_frame": 50, "actor_track": "track_0",
                "input_frame_indices": list(range(0, 51, 5))}
    before = _history_features(source, metadata)
    source.raw["xy"][51:] = 9999
    after = _history_features(source, {**metadata, "true_action": "SHOT"})
    assert all(np.array_equal(left, right) for left, right in zip(before, after))
    with pytest.raises(ValueError, match="Unsupported teacher history"):
        _history_features(source, {**metadata, "input_frame_indices": list(range(1, 52, 5))})
