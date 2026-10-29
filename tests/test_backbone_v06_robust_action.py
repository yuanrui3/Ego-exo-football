from pathlib import Path

import numpy as np
import pytest

from src.backbone.batch import validate_job
from src.backbone.contracts import Measurement, Observation, TrackingFrame
from src.backbone.replay_resume_probe_v06 import _interrupt_cursor
from src.backbone.train_action_robust_v06 import (
    CORRUPTION,
    _corrupt_history,
    _corrupted_history,
    _encode_history,
)
from src.backbone.evaluate_robustness_v06 import EVALUATION_CORRUPTIONS, evaluate


def _history() -> list[Observation]:
    observations = []
    for step in range(11):
        measurements = tuple(
            Measurement(
                track=f"track_{player}",
                segment=0,
                xy=(float(player + step), float(player - step)),
                team=1 if player < 2 else 2,
                goalkeeper=player == 3,
            )
            for player in range(4)
        )
        observations.append(Observation(
            source_id="fixture",
            half="1st",
            index=step * 5,
            timestamp_s=step * 0.2,
            actor_track="track_0",
            measurements=measurements,
        ))
    return observations


class _FrameSource:
    def frame(self, index: int) -> TrackingFrame:
        measurements = tuple(
            Measurement(
                track=f"track_{player}",
                segment=0,
                xy=(float(player), float(player - 2)),
                team=1 if player < 2 else 2,
                goalkeeper=player == 3,
            )
            for player in range(4)
        )
        return TrackingFrame("fixture", "1st", index, index / 25, measurements, 0)


def test_corruption_is_deterministic_and_preserves_actor_history() -> None:
    policy = {
        **CORRUPTION,
        "identity_break_probability": 1.0,
        "additional_correlated_gap_probability": 1.0,
    }
    first, first_record = _corrupt_history(
        _history(), "fixture:1", seed=12, corruption=policy,
    )
    second, second_record = _corrupt_history(
        _history(), "fixture:1", seed=12, corruption=policy,
    )

    assert first_record == second_record
    assert first_record["identity_break_track"] is not None
    assert first_record["correlated_gap_track"] is not None
    assert first_record["additional_detections_removed"] > 0
    assert [observation.index for observation in first] == list(range(0, 51, 5))
    for left, right in zip(first, second, strict=True):
        assert left == right
        assert left.actor is not None
        assert left.actor.segment == 0

    encoded = _encode_history(first, "fixture:1")
    assert encoded["x"].shape == (11, 22, 10)
    assert encoded["mask"].shape == (11, 22)
    assert encoded["visible"].shape == (11, 22)
    assert np.all(encoded["x"][:, :, 9].sum(axis=1) == 1)
    assert np.isfinite(encoded["x"]).all()


def test_corrupted_history_uses_the_requested_evaluation_policy() -> None:
    metadata = {
        "actor_track": "track_0",
        "query_frame": 50,
        "input_frame_indices": list(range(0, 51, 5)),
    }
    clean_policy = {
        **CORRUPTION,
        "provider_dropout": False,
        "coordinate_noise_std_m": 0.0,
    }
    stress_policy = {
        **clean_policy,
        "coordinate_noise_std_m": 0.30,
        "seed": 62027,
    }
    default_history, default_alignment = _corrupted_history(
        _FrameSource(), metadata, "fixture:3",
    )
    explicit_history, explicit_alignment = _corrupted_history(
        _FrameSource(), metadata, "fixture:3", corruption=CORRUPTION,
    )
    assert default_alignment == explicit_alignment
    assert np.array_equal(
        _encode_history(default_history, "fixture:3")["x"],
        _encode_history(explicit_history, "fixture:3")["x"],
    )
    clean_history, clean_alignment = _corrupted_history(
        _FrameSource(), metadata, "fixture:3", corruption=clean_policy,
    )
    stress_history, stress_alignment = _corrupted_history(
        _FrameSource(), metadata, "fixture:3", corruption=stress_policy,
    )

    assert clean_alignment == stress_alignment
    assert clean_alignment["query_lag_frames"] == 0
    clean_encoded = _encode_history(clean_history, "fixture:3")
    stress_encoded = _encode_history(stress_history, "fixture:3")
    assert not np.array_equal(clean_encoded["x"], stress_encoded["x"])
    assert all(observation.actor is not None for observation in stress_history)
    assert stress_history[-1].index == metadata["query_frame"]


def test_corruption_does_not_mutate_source_observations() -> None:
    source = _history()
    source_first = source[0]
    policy = {
        **CORRUPTION,
        "identity_break_probability": 1.0,
        "additional_correlated_gap_probability": 1.0,
    }
    corrupted, _ = _corrupt_history(
        source, "fixture:2", seed=7, corruption=policy,
    )

    assert source[0] is source_first
    assert len(source[0].measurements) == 4
    assert len(corrupted[0].measurements) <= 4
    assert source[0].measurements[1].segment == 0
    assert any(
        measurement.segment != 0
        for observation in corrupted
        for measurement in observation.measurements
        if measurement.track != "track_0"
    )


def test_resume_probe_uses_a_safe_midpoint_for_each_segment_length() -> None:
    assert _interrupt_cursor(600) == 300
    assert _interrupt_cursor(1500) == 750


def test_evaluation_corruptions_isolate_each_component() -> None:
    assert set(EVALUATION_CORRUPTIONS) == {
        "training_regimen", "intensified_stress", "provider_dropout_only",
        "correlated_gap_only", "coordinate_noise_only", "identity_break_only",
    }
    provider = EVALUATION_CORRUPTIONS["provider_dropout_only"]
    assert provider["provider_dropout"] is True
    assert provider["additional_correlated_gap_probability"] == 0.0
    assert provider["identity_break_probability"] == 0.0
    assert provider["coordinate_noise_std_m"] == 0.0

    gap = EVALUATION_CORRUPTIONS["correlated_gap_only"]
    assert gap["provider_dropout"] is False
    assert gap["additional_correlated_gap_probability"] == 1.0
    assert gap["identity_break_probability"] == 0.0
    assert gap["coordinate_noise_std_m"] == 0.0

    noise = EVALUATION_CORRUPTIONS["coordinate_noise_only"]
    assert noise["provider_dropout"] is False
    assert noise["additional_correlated_gap_probability"] == 0.0
    assert noise["identity_break_probability"] == 0.0
    assert noise["coordinate_noise_std_m"] == 0.15

    identity = EVALUATION_CORRUPTIONS["identity_break_only"]
    assert identity["provider_dropout"] is False
    assert identity["additional_correlated_gap_probability"] == 0.0
    assert identity["identity_break_probability"] == 1.0
    assert identity["coordinate_noise_std_m"] == 0.0

    component_policies = [
        EVALUATION_CORRUPTIONS[name]
        for name in (
            "provider_dropout_only", "correlated_gap_only",
            "coordinate_noise_only", "identity_break_only",
        )
    ]
    assert all(policy["version"] for policy in EVALUATION_CORRUPTIONS.values())
    assert all(policy["seed"] != CORRUPTION["seed"] for policy in component_policies)
    assert len({policy["seed"] for policy in component_policies}) == len(component_policies)


def test_isolated_gap_and_identity_policies_apply_only_the_requested_change() -> None:
    gap_history, gap_record = _corrupt_history(
        _history(), "fixture:gap-only", seed=62029,
        corruption=EVALUATION_CORRUPTIONS["correlated_gap_only"],
    )
    assert gap_record["correlated_gap_track"] is not None
    assert gap_record["identity_break_track"] is None
    assert gap_record["additional_detections_removed"] > 0
    assert all(observation.actor is not None for observation in gap_history)

    identity_history, identity_record = _corrupt_history(
        _history(), "fixture:identity-only", seed=62031,
        corruption=EVALUATION_CORRUPTIONS["identity_break_only"],
    )
    assert identity_record["identity_break_track"] is not None
    assert identity_record["correlated_gap_track"] is None
    assert identity_record["additional_detections_removed"] == 0
    assert all(observation.actor is not None for observation in identity_history)


def test_isolated_provider_dropout_and_noise_change_observations() -> None:
    class MovingSource:
        def frame(self, index: int) -> TrackingFrame:
            actor_x = index / 25
            measurements = [
                Measurement("track_0", 0, (actor_x, 0.0), 1, False),
                *(
                    Measurement(
                        f"track_{player}", 0,
                        (actor_x + 5 * player, float(player - 1)),
                        1 if player < 2 else 2, player == 7,
                    )
                    for player in range(1, 8)
                ),
            ]
            return TrackingFrame(
                "fixture", "1st", index, index / 25, tuple(measurements), 0,
            )

    metadata = {
        "actor_track": "track_0",
        "query_frame": 50,
        "input_frame_indices": list(range(0, 51, 5)),
    }
    clean_policy = {
        **CORRUPTION,
        "provider_dropout": False,
        "additional_correlated_gap_probability": 0.0,
        "identity_break_probability": 0.0,
        "coordinate_noise_std_m": 0.0,
    }
    clean_history, _ = _corrupted_history(
        MovingSource(), metadata, "fixture:component-clean", corruption=clean_policy,
    )
    dropout_history, _ = _corrupted_history(
        MovingSource(), metadata, "fixture:component-clean",
        corruption=EVALUATION_CORRUPTIONS["provider_dropout_only"],
    )
    noise_history, _ = _corrupted_history(
        MovingSource(), metadata, "fixture:component-clean",
        corruption=EVALUATION_CORRUPTIONS["coordinate_noise_only"],
    )
    clean = _encode_history(clean_history, "fixture:component-clean")
    noise = _encode_history(noise_history, "fixture:component-clean")
    assert any(
        len(clean_frame.measurements) != len(dropout_frame.measurements)
        for clean_frame, dropout_frame in zip(clean_history, dropout_history, strict=True)
    )
    assert not np.array_equal(clean["x"], noise["x"])
    assert np.array_equal(clean["mask"], noise["mask"])


def test_evaluator_preserves_existing_stage_e_artifacts(tmp_path: Path) -> None:
    marker = tmp_path / "prior.json"
    marker.write_text("historical\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing Stage E artifacts"):
        evaluate(tmp_path / "unused-index.json", tmp_path / "unused-checkpoint.pt", tmp_path)

    assert marker.read_text(encoding="utf-8") == "historical\n"


def test_batch_requires_explicit_mode_for_a_continuous_ten_minute_job() -> None:
    job = {
        "id": "long-stress-fixture",
        "intended_use": "retrospective_replay",
        "match": "118577",
        "half": "1st",
        "start_frame": 1,
        "samples": 3000,
        "actors": ["track_0"],
        "selection": "Fully observed retrospective test actor",
        "action_queries": [],
    }
    with pytest.raises(ValueError, match="11..1500"):
        validate_job(job)

    validate_job({**job, "long_sequence_mode": True})

    with pytest.raises(ValueError, match="restricted to retrospective replay"):
        validate_job({
            **job, "match": "117092",
            "intended_use": "training_export", "long_sequence_mode": True,
        })
