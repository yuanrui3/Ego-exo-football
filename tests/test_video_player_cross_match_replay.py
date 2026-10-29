from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import cv2
import numpy as np

from src.backbone.video_player_cross_match_replay import (
    _read_video_sample,
    _select_video_actors,
)


class _FakeCapture:
    def __init__(self) -> None:
        self.next_frame = 0

    def read(self) -> tuple[bool, np.ndarray]:
        frame = np.array([self.next_frame], dtype=np.uint8)
        self.next_frame += 1
        return True, frame

    def grab(self) -> bool:
        self.next_frame += 1
        return True


def test_appearance_samples_use_the_same_first_frame_as_video_predictions() -> None:
    capture = _FakeCapture()

    first = _read_video_sample(cast(cv2.VideoCapture, capture), 100)
    second = _read_video_sample(cast(cv2.VideoCapture, capture), 105)

    assert int(first[0]) == 0
    assert int(second[0]) == 5
    assert capture.next_frame == 10


def test_actor_selection_uses_contiguous_video_observations_not_box_counts() -> None:
    candidate_frames = []
    selection_frames = []
    for sample_index in range(110):
        detections = [
            {"track": track, "footpoint_xy_px": [sample_index, 10.0]}
            for track in ("unprojected-box-heavy", "continuous-a", "continuous-b", "late")
        ]
        measurements = []
        if sample_index < 12:
            measurements.append(SimpleNamespace(
                track="continuous-a", team=1, goalkeeper=False,
            ))
        if sample_index < 6 or 8 <= sample_index < 12:
            measurements.append(SimpleNamespace(
                track="continuous-b", team=2, goalkeeper=False,
            ))
        if sample_index >= 100:
            measurements.append(SimpleNamespace(
                track="late", team=1, goalkeeper=False,
            ))
        selection_frames.append(SimpleNamespace(
            index=sample_index * 5,
            measurements=tuple(measurements),
        ))
        candidate_frames.append({
            "frame_index": sample_index * 5,
            "detections": detections,
        })

    actors = _select_video_actors(candidate_frames, selection_frames)

    assert [actor["track"] for actor in actors] == ["continuous-a", "continuous-b"]
    assert [actor["longest_usable_run_samples"] for actor in actors] == [12, 6]
    assert "late" not in {actor["track"] for actor in actors}
    assert all(
        actor["selection_basis"]
        == "video_estimated_position_and_confidence_qualified_attributes"
        for actor in actors
    )
