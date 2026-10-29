import json

from src.backbone.external import ExternalObservationSource, VERSION
from src.backbone.video_pitch_experimental import _external_jsonl, _select_action_query_index


def test_experimental_export_preserves_unknown_attributes_and_pitch_origin(tmp_path):
    frames = [{
        "record_type": "frame",
        "frame_index": 37500,
        "timestamp_s": 1500.0,
        "segment_id": "117093_1st_1500",
        "identity_reset": False,
        "camera_cut": False,
        "measurements": [{
            "track": "pixel-tracklet-1",
            "segment": 0,
            "team": "UNKNOWN",
            "goalkeeper": None,
            "position_status": "observed",
            "xy": [52.5, 34.0],
        }],
        "absent_count": 0,
        "detections": [],
    }]
    path = tmp_path / "observations.jsonl"

    _external_jsonl(path, frames, {"video_path": "video.mp4", "source_detection_path": "tracks.jsonl",
                                  "calibration": {"status": "UNVALIDATED"}})
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header["schema_version"] == VERSION

    source = ExternalObservationSource(path)
    replay_frame = next(source.frames(37500, 1))
    measurement = replay_frame.measurements[0]
    assert measurement.xy == (0.0, 0.0)
    assert measurement.team == 0
    assert measurement.goalkeeper is None


def test_action_probe_uses_causal_actor_coverage_not_model_score():
    frames = [
        {"measurements": [{"track": "actor"}] if 5 <= index <= 18 else []}
        for index in range(25)
    ]

    query_index, basis = _select_action_query_index(frames, "actor")

    assert query_index == 15
    assert basis == "first supplied actor sample with an 11-sample continuous history"
