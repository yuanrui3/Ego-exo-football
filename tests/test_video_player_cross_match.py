from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.backbone.video_player_cross_match import (
    DEFAULT_OUTPUT,
    DEFAULT_PROTOCOL,
    _maximum_matches,
    _read_annotations,
    _restore_tracker,
    _write_yolo_label_file,
    interval_by_name,
    load_protocol,
    normalized_tolerance,
    project_footpoint,
)
from src.trajectory_dataset import file_digest


def test_locked_protocol_uses_distinct_matches_and_fixed_intervals() -> None:
    assert DEFAULT_PROTOCOL.parent.name == "configs"
    assert DEFAULT_OUTPUT.parent.name == "outputs"
    assert DEFAULT_OUTPUT.name == "video_player_cross_match_20261028_130610"
    protocol = load_protocol(DEFAULT_PROTOCOL)
    development = protocol["selection"]["development_match"]
    evaluation = protocol["selection"]["evaluation_match"]

    assert development["id"] == "117092"
    assert evaluation["id"] == "118576"
    assert development["id"] != evaluation["id"]
    assert interval_by_name(protocol, "117092", "final1") == (1200, 1320)
    assert interval_by_name(protocol, "118576", "final2") == (2400, 2520)


def test_matching_tolerance_scales_with_native_width() -> None:
    assert normalized_tolerance(3840, 40) == 37.5
    assert normalized_tolerance(4096, 40) == 40.0
    assert normalized_tolerance(3840, 20) == 18.75


def test_footpoint_matching_is_one_to_one() -> None:
    predicted = [
        {"footpoint_xy_px": [10.0, 10.0]},
        {"footpoint_xy_px": [11.0, 10.0]},
    ]
    truth = [{"footpoint_xy_px": [10.0, 10.0]}]

    matches = _maximum_matches(predicted, truth, 5.0)

    assert len(matches) == 1
    assert matches[0][1] == 0


def test_resume_restores_active_track_velocity_and_next_id() -> None:
    records = [
        {
            "detections": [
                {"track": "video-player-0001", "footpoint_xy_px": [20.0, 50.0]},
            ],
        },
        {
            "detections": [
                {"track": "video-player-0001", "footpoint_xy_px": [24.0, 54.0]},
            ],
        },
    ]

    tracks, next_id = _restore_tracker(records)

    assert next_id == 2
    assert tracks["video-player-0001"]["footpoint_xy_px"] == [24.0, 54.0]
    assert tracks["video-player-0001"]["velocity_xy_px"] == [4.0, 4.0]
    assert tracks["video-player-0001"]["observations"] == 2


def test_training_tile_labels_restore_source_coordinates() -> None:
    import tempfile
    from pathlib import Path

    annotation = {
        "bbox_xywh_px": [490.0, 100.0, 20.0, 40.0],
        "footpoint_xy_px": [500.0, 140.0],
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "label.txt"
        count = _write_yolo_label_file(
            path,
            tile_index=0,
            tile_left=0,
            tile_width=1088,
            frame_width=4096,
            frame_height=1080,
            annotations=[annotation],
        )
        values = [float(value) for value in path.read_text().split()]

    assert count == 1
    assert values[0] == 0.0
    assert abs(values[1] - (500.0 / 1088)) < 1e-8
    assert abs(values[2] - (120.0 / 1080)) < 1e-8


def test_local_projector_preserves_roi_and_hull_abstentions() -> None:
    projector = {
        "fixed_roi": {"principal_x_px": 5.0, "half_width_px": 650.0},
        "mapping": {
            "camera_K": [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]],
            "distortion_D": [[0.0], [0.0], [0.0], [0.0]],
            "rectified_camera_Knew": [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]],
            "fit_hull_rectified_px": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            "rectified_to_pitch_homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    }

    projected, status = project_footpoint([5.0, 5.0], projector)
    assert projected is not None
    assert max(abs(value - 5.0) for value in projected) < 1e-3
    assert status == "UNVALIDATED_EXPERIMENTAL_LOCAL_PROJECTION"
    assert project_footpoint([1000.0, 5.0], projector) == (
        None,
        "OUTSIDE_FIXED_SETUP_ROI",
    )
    assert project_footpoint([20.0, 5.0], projector) == (
        None,
        "OUTSIDE_FIT_LANDMARK_HULL",
    )


def test_annotation_hash_is_checked_before_scoring_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.backbone.video_player_cross_match as cross_match

    annotation_path = tmp_path / "labels.json"
    annotation_path.write_text(json.dumps({
        "annotations": [{
            "image_id": 1,
            "category_id": 1,
            "bbox_image": {"x": 10, "y": 20, "w": 4, "h": 8},
            "bbox_pitch": {"x_bottom_middle": 12, "y_bottom_middle": 5},
            "attributes": {"player_id": "p1", "team": "left", "role": "player"},
            "track_id": "t1",
        }],
    }), encoding="utf-8")
    source = {
        "annotation": "labels.json",
        "annotation_sha256": file_digest(annotation_path),
        "image_id_frame_zero": 0,
    }
    monkeypatch.setattr(cross_match, "ROOT", tmp_path)

    labels = _read_annotations(source, {1})
    assert labels[1][0]["footpoint_xy_px"] == [12.0, 28.0]
    source["annotation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="locked hash"):
        _read_annotations(source, {1})
