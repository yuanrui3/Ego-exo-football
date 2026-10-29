import json
from pathlib import Path
import time

import numpy as np
import pytest

import src.backbone.video_ingest_v06 as ingest
from src.backbone.video_ingest_v06 import (
    _atomic_json,
    _atomic_jsonl,
    _chunk_rows,
    _clamped_duration_s,
    _fingerprint,
    _sampled_frame_indices,
    _validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "outputs/backbone_v0_6_20261021/stage_b/video_config_v06.json"


def test_frozen_workload_has_distinct_fit_and_development_minutes():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    clips = _validate_config(config)
    assert sum(split == "fitting" for split, _ in clips) == 6
    assert sum(split == "development" for split, _ in clips) == 2
    assert sum(int(clip["duration_s"]) for split, clip in clips if split == "fitting") == 7200
    assert sum(int(clip["duration_s"]) for split, clip in clips if split == "development") == 1200


def test_sampled_native_frames_are_unique_and_exactly_five_hz():
    frames = _sampled_frame_indices(12.0, 60, 25.0, 5)
    assert len(frames) == 300
    assert len(set(frames)) == 300
    assert frames[0] == 300
    assert frames[-1] == 1795
    assert all(right - left == 5 for left, right in zip(frames, frames[1:]))


def test_requested_excerpt_is_clamped_at_real_video_boundary():
    assert _clamped_duration_s(2690.0, 1200, 25.0, 67375) == 5
    assert _clamped_duration_s(2800.0, 1200, 25.0, 67375) == 0


def test_completed_chunk_requires_matching_fingerprint_and_content_hash(tmp_path):
    folder = tmp_path / "excerpt"
    rows = [{"frame_index": 5}, {"frame_index": 10}]
    chunk = folder / "chunks/chunk_0000.jsonl"
    completion = folder / "chunks/chunk_0000.json"
    _atomic_jsonl(chunk, rows)
    fingerprint = _fingerprint({"config": "exact"})
    from src.trajectory_dataset import file_digest

    _atomic_json(completion, {
        "fingerprint": fingerprint,
        "rows_sha256": file_digest(chunk),
    })
    cached, _ = _chunk_rows(folder, 0, fingerprint, [5, 10])
    assert cached == rows
    with pytest.raises(ValueError, match="configuration differs"):
        _chunk_rows(folder, 0, _fingerprint({"config": "changed"}), [5, 10])
    chunk.write_text('{"frame_index":5}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash check"):
        _chunk_rows(folder, 0, fingerprint, [5, 10])


def test_clip_resume_reuses_only_completed_chunks_without_duplicate_rows(tmp_path, monkeypatch):
    class Capture:
        def __init__(self):
            self.frame = 0

        def isOpened(self):
            return True

        def get(self, _prop):
            if _prop == ingest.cv2.CAP_PROP_FPS:
                return 25.0
            if _prop == ingest.cv2.CAP_PROP_FRAME_COUNT:
                return 1000
            return 128 if _prop == ingest.cv2.CAP_PROP_FRAME_WIDTH else 64

        def set(self, prop, value):
            self.frame = int(value)
            return True

        def read(self):
            image = np.full((64, 128, 3), self.frame % 255, dtype=np.uint8)
            return True, image

        def release(self):
            pass

    video = tmp_path / "videos" / "117092" / "117092_panorama_1st_half.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"synthetic-video-source")
    monkeypatch.setattr(ingest.cv2, "VideoCapture", lambda *_args, **_kwargs: Capture())
    monkeypatch.setattr(ingest, "_histogram", lambda _image: np.ones((16, 8), np.float32))
    monkeypatch.setattr(
        ingest,
        "_detect_frame",
        lambda *_args, **_kwargs: [{
            "bbox_xyxy_px": [10.0, 10.0, 20.0, 30.0],
            "detector_confidence": 0.9,
        }],
    )
    monkeypatch.setattr(ingest, "_annotate", lambda _image, _detections, path, _frame: path.touch())
    config = {
        "source_paths": {"video_root": "videos"},
        "workload": {
            "target_fps": 5,
            "chunk_samples": 25,
            "device": "cuda",
            "detector": {"tile_columns": 1, "tile_overlap_px": 0, "confidence_threshold": 0.1},
        },
    }
    clip = {"match": "117092", "half": "1st", "start_s": 1, "duration_s": 10}
    output = tmp_path / "out"
    output.mkdir()
    kwargs = {
        "config": config,
        "config_sha": "config-hash",
        "root": tmp_path,
        "output": output,
        "model": object(),
        "weights_sha": "weights-hash",
        "started": time.monotonic(),
        "max_runtime_s": 120,
    }
    first_totals = {
        "new_inference_frames": 0, "reused_inference_frames": 0,
        "reused_chunks": 0, "detections": 0,
        "fitting_frames": 0, "fitting_unique_source_frames": 0,
        "development_frames": 0, "development_unique_source_frames": 0,
    }
    first = ingest._run_clip(
        "fitting", clip, **kwargs, totals=first_totals, resume=False,
    )
    assert first["inference_frames"] == 50
    assert first_totals["new_inference_frames"] == 50

    resumed_totals = {
        "new_inference_frames": 0, "reused_inference_frames": 0,
        "reused_chunks": 0, "detections": 0,
        "fitting_frames": 0, "fitting_unique_source_frames": 0,
        "development_frames": 0, "development_unique_source_frames": 0,
    }
    resumed = ingest._run_clip(
        "fitting", clip, **kwargs, totals=resumed_totals, resume=True,
    )
    assert resumed["inference_frames"] == 50
    assert resumed["new_inference_frames"] == 0
    assert resumed["reused_inference_frames"] == 50
    assert resumed_totals["reused_chunks"] == 2
    assert sum(1 for _ in (output / resumed["observations_path"]).open(encoding="utf-8")) == 50
