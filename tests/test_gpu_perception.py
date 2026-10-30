import numpy as np
import hashlib
import json
import pytest

from src.backbone.gpu_perception import restore_boxes, tile_bounds, tile_labels
from src.backbone.gpu_perception import aggregate_scores, enable_detector_gradients
from src.backbone.gpu_perception import score_predictions
from src.backbone.gpu_perception import finalize_interrupted
from src.backbone.gpu_perception_post import assign_global, combined_diagnostics, track_history_population
import torch


def test_tall_tiles_keep_native_geometry_and_both_offsets():
    bounds = tile_bounds(3840, 1906)
    assert len(bounds) == 8
    assert len(tile_bounds(4096, 1080)) == 4
    box = np.array([[10, 20, 30, 40]])
    assert restore_boxes(box, bounds[-1]).tolist() == [[2826, 877, 2846, 897]]


def test_labels_use_bottom_center_and_preserve_overlap_positives():
    truth = [{"bbox_xywh_px": [900, 950, 24, 8]}]
    bounds = tile_bounds(3840, 1906)
    labels = [tile_labels(truth, b)[0] for b in bounds]
    assert sum(map(len, labels)) == 4
    for rows, b in zip(labels, bounds):
        for _, cx, cy, width, height in rows:
            assert abs((cy + height / 2) * (b[3] - b[1]) + b[1] - 958) < 1e-6
            assert abs(cx * (b[2] - b[0]) + b[0] - 912) < 1e-6


def test_outside_footpoint_and_nonpositive_boxes_do_not_become_labels():
    assert tile_labels([{"bbox_xywh_px": [1, 1, 4, 4]}], (10, 10, 20, 20))[0] == []


def test_stripped_detector_gradients_are_enabled_except_fixed_dfl():
    model = torch.nn.ModuleDict({"conv": torch.nn.Linear(2, 2),
                                "dfl": torch.nn.Linear(2, 2, bias=False)})
    model.requires_grad_(False)
    wrapped = torch.nn.ModuleDict({"model": model})
    enable_detector_gradients(wrapped)
    assert wrapped.model.conv.weight.requires_grad
    assert not wrapped.model.dfl.weight.requires_grad


def test_history_and_adjacent_transition_denominators_are_explicit():
    frames = [{"frame_index": i * 5, "detections": [
        {"footpoint_xy_px": [1, 2], "track": "one"}]} for i in range(12)]
    labels = {i * 5: [{"footpoint_xy_px": [1, 2], "track_id": "truth"}] for i in range(12)}
    score = aggregate_scores(frames, labels, 4096)["thresholds"]["40"]
    assert score["complete_11_sample_history_observations"] == 2
    assert score["all_track_observation_opportunities"] == 12
    assert score["matched_adjacent_transitions"] == 11
    assert score["identity_switches"] == 0
    assert score["identity_consistent_matched_history_observations"] == 2
    assert score["continuous_track_length_samples_histogram"] == {12: 1}


def test_global_association_minimizes_total_cost_without_reference_identity():
    tracks = {
        "a": {"footpoint_xy_px": [0., 0.], "velocity_xy_px": [0., 0.], "last_sample": 0, "observations": 1},
        "b": {"footpoint_xy_px": [4., 0.], "velocity_xy_px": [0., 0.], "last_sample": 0, "observations": 1},
    }
    detections = [{"footpoint_xy_px": [3., 0.]}, {"footpoint_xy_px": [6., 0.]}]
    assert assign_global(detections, tracks, 3, 1, 10) == 3
    assert [row["track"] for row in detections] == ["a", "b"]


def test_gap_continuity_is_separate_from_adjacent_switches():
    frames = [
        {"frame_index": 0, "detections": [{"footpoint_xy_px": [1, 2], "track": "a"}]},
        {"frame_index": 5, "detections": []},
        {"frame_index": 10, "detections": [{"footpoint_xy_px": [1, 2], "track": "a"}]},
    ]
    labels = {i: [{"footpoint_xy_px": [1, 2], "track_id": "truth"}] for i in (0, 5, 10)}
    score = aggregate_scores(frames, labels, 4096)["thresholds"]["40"]
    assert score["matched_adjacent_transitions"] == 0
    assert score["matched_returns_after_gap"] == 1
    assert score["same_runtime_track_returns_after_gap"] == 1


def test_long_runtime_track_does_not_imply_history_for_the_same_real_player():
    frames = [{"frame_index": i * 5, "detections": [
        {"footpoint_xy_px": [1, 2], "track": "one"}]} for i in range(12)]
    labels = {i * 5: [{"footpoint_xy_px": [1, 2], "track_id": str(i // 6)}]
              for i in range(12)}
    score = aggregate_scores(frames, labels, 4096)["thresholds"]["40"]
    assert score["complete_11_sample_history_observations"] == 2
    assert score["identity_consistent_matched_history_observations"] == 0


def test_control_sampling_does_not_claim_five_hz_history_contract():
    frames = [{"frame_index": 0, "detections": [{"footpoint_xy_px": [1, 2], "track": "a"}]}]
    labels = {0: [{"footpoint_xy_px": [1, 2], "track_id": "truth"}]}
    score = aggregate_scores(frames, labels, 4096, sampling_hz=1)["thresholds"]["40"]
    assert score["continuous_track_duration_seconds_median"] == 0
    assert score["covered_sample_time_seconds_median"] == 1
    assert not score["backbone_history_compatible_sampling"]


@pytest.mark.parametrize("indices", [
    [0, 5, 10, 15],
    [0, 5, 5, 15, 20],
    [20, 15, 10, 5, 0],
])
def test_sealed_scores_reject_missing_duplicate_or_reversed_source_frames(tmp_path, indices):
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps({"frame_index": i, "detections": []}) for i in indices))
    record = {
        "path": str(path),
        "prediction_jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sampling_hz": 5,
        "interval_seconds": [0, 1],
    }
    with pytest.raises(ValueError, match="complete locked interval"):
        score_predictions(tmp_path, record, {})


def test_combined_metrics_pool_actual_runs_and_errors_not_interval_medians():
    rows = []
    for length, errors in ((12, [1., 2.]), (2, [10.])):
        frames = [{"frame_index": i * 5, "detections": [
            {"footpoint_xy_px": [1, 2], "track": "one" if length == 12 else str(i)}]}
                  for i in range(length)]
        labels = {i * 5: [{"footpoint_xy_px": [1, 2], "track_id": "truth"}]
                  for i in range(length)}
        score = aggregate_scores(frames, labels, 4096)
        score["projection"] = {
            "conditional_errors_m": errors, "projected": length, "unprojected": 0,
            "supported_truth_instances": length,
            "pixel_matched_in_support_with_pitch_truth": len(errors),
        }
        rows.append(score)
    result = combined_diagnostics(rows)
    assert result["frames"] == 14
    assert result["continuous_track_runs"] == 3
    assert result["median_continuous_track_span_seconds"] == 0
    assert result["conditional_projection_median_m"] == 2
    assert np.isclose(result["conditional_projection_p90_m"], 8.4)
    assert result["identity_consistent_matched_history_observations"] == 2
    assert result["truth_instances"] == 14


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_interrupted_fit_finalization_preserves_failure_and_confirmed_step_bound(tmp_path, monkeypatch, encoding):
    log = tmp_path / "training.log"
    log.write_text("RuntimeError: Nonfinite detector gradients; no automatic recovery", encoding=encoding)
    weights = tmp_path / "selected.pt"
    weights.write_bytes(b"checkpoint")
    (tmp_path / "training_progress.json").write_text(json.dumps({
        "selected": str(weights), "optimizer_steps": 6750, "epochs": [{"epoch": 15}],
    }))
    (tmp_path / "recipe.json").write_text(json.dumps({"created_utc": "2026-10-29T13:13:02+00:00"}))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {
        "epoch": 14, "ema": torch.nn.Linear(1, 1),
    })
    result = finalize_interrupted(tmp_path)
    assert result["optimizer_steps"] == 6750
    assert result["interrupted_epoch"] == 16
    assert "INTERRUPTED" in result["status"]
    assert not result["release_promotion"]
    assert "lower bound" in result["optimizer_step_count_semantics"]
    with pytest.raises(FileExistsError):
        finalize_interrupted(tmp_path)


def test_actor_history_population_requires_consecutive_not_accumulated_samples():
    frames = [{"detections": [{"track": "continuous"}] + (
        [{"track": "gapped"}] if i != 6 else [])} for i in range(13)]
    result = track_history_population(frames)
    assert result["distinct_runtime_tracks"] == 2
    assert result["runtime_tracks_ever_complete_11_samples"] == 1
    assert result["runtime_track_complete_history_fraction"] == .5
