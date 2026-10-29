from __future__ import annotations

from src.backbone.video_player_replay import (
    _assign_tracklets,
    _build_external_rows,
)


def _detection(x: float, y: float, track: str | None = None) -> dict:
    return {
        "bbox_xyxy_px": [x - 5, y - 10, x + 5, y],
        "footpoint_xy_px": [x, y],
        "pitch_xy_m": [x / 100, y / 100],
        "projection_status": "EXPERIMENTAL_SUPPORTED_HULL",
        "track": track,
        "team": 0,
        "team_prediction": "left",
        "team_confidence": 0.6,
        "goalkeeper": None,
        "role_prediction": "player",
        "role_confidence": 0.7,
        "detector_confidence": 0.8,
    }


def test_tracklet_assignment_preserves_nearby_identity() -> None:
    tracks: dict[str, dict] = {}
    detections = [_detection(100, 200)]
    next_id = _assign_tracklets(detections, tracks, 1, 0)
    original_id = detections[0]["track"]

    next_detections = [_detection(110, 205)]
    next_id = _assign_tracklets(next_detections, tracks, next_id, 1)

    assert next_detections[0]["track"] == original_id
    assert tracks[original_id]["observations"] == 2


def test_attribute_assisted_rows_require_a_matched_annotation() -> None:
    frames = [{
        "record_type": "frame",
        "frame_index": 100,
        "timestamp_s": 4.0,
        "segment_id": "segment",
        "detections": [_detection(10, 20), _detection(500, 500)],
    }]
    labels = {
        100: [{
            "footpoint_xy_px": [10, 20],
            "team": "left",
            "role": "player",
        }],
    }

    autonomous = _build_external_rows(
        frames, None, assisted_attributes=False,
    )[0]
    assisted = _build_external_rows(
        frames, labels, assisted_attributes=True,
    )[0]

    assert len(autonomous["measurements"]) == 2
    assert len(assisted["measurements"]) == 1
    assert assisted["measurements"][0]["team"] == 1
    assert assisted["measurements"][0]["goalkeeper"] is False
    assert assisted["measurements"][0]["quality"]["attribute_source"].startswith(
        "MATCHED_HELDOUT_ANNOTATION_ASSISTED",
    )
