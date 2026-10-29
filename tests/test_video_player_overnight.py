from __future__ import annotations

from pathlib import Path

import numpy as np

from src.backbone.video_player_overnight import (
    _tile_for_center,
    _tiles,
    _write_yolo_labels,
)
from src.backbone.video_player_attributes import (
    _maximum_footpoint_matches,
    appearance_feature,
)


def test_native_panorama_tiles_retain_expected_offsets() -> None:
    image = np.zeros((1080, 4096, 3), dtype=np.uint8)
    tiles, offsets = _tiles(image)

    assert len(tiles) == 4
    assert offsets == [0, 960, 1984, 3008]
    assert [tile.shape[1] for tile in tiles] == [1088, 1152, 1152, 1088]


def test_player_label_is_written_only_to_core_tile(tmp_path: Path) -> None:
    annotation = {
        "bbox_xywh_px": [1000.0, 100.0, 48.0, 43.0],
        "footpoint_xy_px": [1024.0, 143.0],
    }

    first_count = _write_yolo_labels(
        tmp_path / "tile0.txt", 0, 0, 1088, [annotation],
    )
    second_path = tmp_path / "tile1.txt"
    second_count = _write_yolo_labels(
        second_path, 1, 960, 1152, [annotation],
    )

    assert first_count == 0
    assert second_count == 1
    values = [float(value) for value in second_path.read_text().split()]
    assert values[0] == 0
    assert np.allclose(values[1:], [64 / 1152, 121.5 / 1080, 48 / 1152, 43 / 1080])


def test_tile_assignment_is_disjoint_at_core_boundaries() -> None:
    assert [_tile_for_center(x) for x in (0, 1023.9, 1024, 2048, 3072, 4095)] == [
        0, 0, 1, 2, 3, 3,
    ]


def test_footpoint_matching_finds_maximum_cardinality() -> None:
    detections = [
        {"bbox_xyxy_px": [39, 0, 41, 0]},
        {"bbox_xyxy_px": [-31, 0, -29, 0]},
    ]
    annotations = [
        {"footpoint_xy_px": [0, 0]},
        {"footpoint_xy_px": [100, 0]},
    ]

    matches = _maximum_footpoint_matches(detections, annotations, 100)

    assert len(matches) == 2
    assert {(detection, annotation) for detection, annotation, _ in matches} == {
        (0, 1), (1, 0),
    }


def test_appearance_feature_has_stable_shape_for_nonempty_and_empty_crops() -> None:
    image = np.zeros((100, 60, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 255)

    feature = appearance_feature(image, [10, 10, 50, 90])
    empty = appearance_feature(image, [20, 20, 20, 20])

    assert feature.shape == (104,)
    assert feature.sum() > 0
    assert empty.shape == feature.shape
