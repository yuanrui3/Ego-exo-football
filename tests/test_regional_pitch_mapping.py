import cv2
import numpy as np
import pytest

from src.backbone.regional_pitch_mapping_v06 import (
    _transform_image_points,
    _write_pitch_residual_overlay,
)


def test_raw_native_coordinate_transform_preserves_pixels():
    image = np.array([[0.0, 5.0], [100.0, 50.0]])
    camera = np.eye(3)
    distortion = np.zeros((4, 1))
    rectified_camera = np.eye(3)

    transformed = _transform_image_points(
        image, camera, distortion, rectified_camera, "raw_native",
    )

    np.testing.assert_array_equal(transformed, image)
    assert transformed is not image


def test_fisheye_transform_matches_opencv():
    image = np.array([[10.0, 20.0], [80.0, 45.0]])
    camera = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    distortion = np.zeros((4, 1))
    rectified_camera = camera.copy()

    transformed = _transform_image_points(
        image, camera, distortion, rectified_camera, "fisheye_undistorted",
    )

    expected = cv2.fisheye.undistortPoints(
        image[:, None], camera, distortion, P=rectified_camera,
    )[:, 0]
    np.testing.assert_allclose(transformed, expected, atol=1e-9)


def test_unknown_coordinate_transform_is_rejected():
    with pytest.raises(ValueError, match="Unsupported image coordinate transform"):
        _transform_image_points(
            np.zeros((1, 2)), np.eye(3), np.zeros((4, 1)), np.eye(3), "implicit",
        )


def test_pitch_residual_overlay_marks_interpolation_and_extrapolation(tmp_path):
    output = tmp_path / "residuals.png"
    landmarks = [
        {
            "block": 0,
            "landmark_index": 2,
            "pitch_xy_true_m": [10.0, 20.0],
            "pitch_xy_predicted_m": [11.0, 21.0],
            "outside_fit_hull": False,
        },
        {
            "block": 1,
            "landmark_index": 7,
            "pitch_xy_true_m": [40.0, 30.0],
            "pitch_xy_predicted_m": [42.0, 32.0],
            "outside_fit_hull": True,
        },
    ]

    _write_pitch_residual_overlay(output, landmarks)

    image = cv2.imread(str(output))
    assert image is not None
    assert image.shape == (764, 1080, 3)
