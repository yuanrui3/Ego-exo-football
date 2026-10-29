"""Attempt a frozen, annotation-assisted regional image-to-pitch validation."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np

from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]


def _transform_image_points(
    image: np.ndarray,
    camera: np.ndarray,
    distortion: np.ndarray,
    rectified_camera: np.ndarray,
    coordinate_transform: str,
) -> np.ndarray:
    if coordinate_transform == "raw_native":
        return image.copy()
    if coordinate_transform == "fisheye_undistorted":
        return cv2.fisheye.undistortPoints(
            image[:, None], camera, distortion, P=rectified_camera,
        )[:, 0]
    raise ValueError(f"Unsupported image coordinate transform: {coordinate_transform}")


def _write_pitch_residual_overlay(path: Path, landmarks: list[dict]) -> None:
    scale = 8.0
    left, top = 120, 110
    width, height = round(105.0 * scale), round(68.0 * scale)
    canvas = np.full((top * 2 + height, left * 2 + width, 3), 248, dtype=np.uint8)
    right, bottom = left + width, top + height
    cv2.rectangle(canvas, (left, top), (right, bottom), (40, 100, 40), 3)
    cv2.line(
        canvas, (left + width // 2, top), (left + width // 2, bottom),
        (150, 150, 150), 2,
    )
    cv2.putText(
        canvas, "Withheld landmark residuals; arrow scale = 8 px/m",
        (left, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2,
    )
    cv2.putText(
        canvas, "green: inside fit hull; red: outside fit hull",
        (left, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1,
    )
    for item in landmarks:
        true_x, true_y = item["pitch_xy_true_m"]
        pred_x, pred_y = item["pitch_xy_predicted_m"]
        start = (round(left + true_x * scale), round(top + true_y * scale))
        end = (round(left + pred_x * scale), round(top + pred_y * scale))
        color = (0, 0, 220) if item["outside_fit_hull"] else (30, 150, 30)
        cv2.arrowedLine(canvas, start, end, color, 2, tipLength=0.25)
        cv2.circle(canvas, start, 4, (220, 100, 20), -1)
        cv2.putText(
            canvas, f"{item['block']}:{item['landmark_index']}",
            (start[0] + 5, start[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (20, 20, 20), 1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write landmark residual overlay: {path}")


def evaluate(
    metadata_path: Path,
    policy_path: Path,
    video_path: Path,
    *,
    output_path: Path,
) -> dict:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    match = policy["selected_match"]
    if match not in metadata or match == "release_notes":
        raise ValueError(f"Calibration metadata does not include selected match {match}")
    calibration = metadata[match]["calibration"]
    keypoint_entry = next(
        item for name, item in calibration.items() if name.endswith("_keypoints.json")
    )
    intrinsics_entry = next(
        item for name, item in calibration.items() if name.endswith("_camera_intrinsics.npz")
    )
    point_path = Path(keypoint_entry["path"])
    intrinsics_path = Path(intrinsics_entry["path"])
    points = json.loads(point_path.read_text(encoding="utf-8"))
    pitch = np.asarray([ast.literal_eval(key) for key in points], dtype=np.float64)
    image = np.asarray(list(points.values()), dtype=np.float64)
    if image.shape != pitch.shape or image.ndim != 2 or image.shape[1] != 2:
        raise ValueError("Released keypoints must provide paired 2D image and pitch coordinates")
    if not np.isfinite(image).all() or not np.isfinite(pitch).all():
        raise ValueError("Nonfinite released calibration correspondence")
    with np.load(intrinsics_path, allow_pickle=False) as archive:
        camera = archive["K"]
        distortion = archive["D"]
        rectified_camera = archive["Knew"]
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open selected panorama for dimension verification: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if np.any(image[:, 0] < 0) or np.any(image[:, 0] > width):
        raise ValueError("Released image keypoint lies outside the selected panorama width")
    if np.any(image[:, 1] < 0) or np.any(image[:, 1] > height):
        raise ValueError("Released image keypoint lies outside the selected panorama height")
    coordinate_transform = policy.get(
        "coordinate_transform", "fisheye_undistorted",
    )
    projection_image = _transform_image_points(
        image, camera, distortion, rectified_camera, coordinate_transform,
    )
    if not np.isfinite(projection_image).all():
        raise ValueError("Image-to-pitch coordinate transform produced nonfinite points")

    order = np.argsort(pitch[:, 0], kind="stable")
    blocks = np.array_split(order, int(policy["minimum_separated_blocks"]))
    rows = []
    errors_all = []
    landmark_residuals = []
    test_count = 0
    for block_index, testing in enumerate(blocks):
        fitting = np.setdiff1d(np.arange(len(pitch)), testing)
        matrix, inliers = cv2.findHomography(
            projection_image[fitting], pitch[fitting], cv2.RANSAC,
            float(policy["ransac_reprojection_threshold_m"]),
        )
        if matrix is None:
            rows.append({
                "block": block_index,
                "status": "FIT_FAILED",
                "fitting_count": len(fitting),
                "withheld_count": len(testing),
            })
            continue
        projected = cv2.perspectiveTransform(projection_image[testing, None], matrix)[:, 0]
        errors = np.linalg.norm(projected - pitch[testing], axis=1)
        if not np.isfinite(errors).all():
            rows.append({
                "block": block_index,
                "status": "NONFINITE_PROJECTION",
                "fitting_count": len(fitting),
                "withheld_count": len(testing),
            })
            continue
        hull = cv2.convexHull(projection_image[fitting].astype(np.float32))
        outside = [
            cv2.pointPolygonTest(hull, tuple(map(float, point)), False) < 0
            for point in projection_image[testing]
        ]
        residual_vectors = projected - pitch[testing]
        landmark_keys = list(points)
        for offset, landmark_index in enumerate(testing):
            landmark_residuals.append({
                "block": block_index,
                "landmark_index": int(landmark_index),
                "landmark_key": landmark_keys[int(landmark_index)],
                "image_xy_raw_px": image[int(landmark_index)].tolist(),
                "image_xy_rectified_px": projection_image[int(landmark_index)].tolist(),
                "pitch_xy_true_m": pitch[int(landmark_index)].tolist(),
                "pitch_xy_predicted_m": projected[offset].tolist(),
                "residual_xy_m": residual_vectors[offset].tolist(),
                "error_m": float(errors[offset]),
                "outside_fit_hull": bool(outside[offset]),
            })
        errors_all.extend(map(float, errors))
        test_count += len(testing)
        rows.append({
            "block": block_index,
            "status": "MEASURED",
            "fitting_count": len(fitting),
            "withheld_count": len(testing),
            "ransac_inliers": int(inliers.sum()),
            "median_pitch_error_m": float(np.median(errors)),
            "p90_pitch_error_m": float(np.percentile(errors, 90)),
            "mean_residual_vector_m": np.mean(residual_vectors, axis=0).tolist(),
            "median_residual_vector_m": np.median(residual_vectors, axis=0).tolist(),
            "outside_fit_hull_fraction": float(np.mean(outside)),
            "image_bbox_px": [
                float(projection_image[testing, 0].min()),
                float(projection_image[testing, 1].min()),
                float(projection_image[testing, 0].max()),
                float(projection_image[testing, 1].max()),
            ],
            "pitch_bbox_m": [
                float(pitch[testing, 0].min()),
                float(pitch[testing, 1].min()),
                float(pitch[testing, 0].max()),
                float(pitch[testing, 1].max()),
            ],
            "matrix_image_to_pitch_m": matrix.tolist(),
        })

    median = float(np.median(errors_all)) if errors_all else None
    p90 = float(np.percentile(errors_all, 90)) if errors_all else None
    measured_blocks = sum(row["status"] == "MEASURED" for row in rows)
    gate_pass = (
        test_count >= int(policy["minimum_total_withheld_correspondences"])
        and measured_blocks >= int(policy["minimum_separated_blocks"])
        and median is not None
        and p90 is not None
        and median <= float(policy["acceptance"]["median_pitch_error_m_max"])
        and p90 <= float(policy["acceptance"]["p90_pitch_error_m_max"])
    )
    image_left = float(image[:, 0].min())
    image_right = float(image[:, 0].max())
    local_left = image_left + (image_right - image_left) / 3
    local_right = image_left + 2 * (image_right - image_left) / 3
    local = np.flatnonzero((image[:, 0] >= local_left) & (image[:, 0] <= local_right))
    local_result: dict = {
        "region": "middle image-x third of annotated support; diagnostic ROI only, not a camera claim",
        "image_bounds_px": [local_left, local_right],
        "correspondences": int(len(local)),
        "status": "INSUFFICIENT_LOCAL_CORRESPONDENCES",
    }
    if len(local) >= 8:
        local_order = local[np.argsort(pitch[local, 1], kind="stable")]
        local_test_count = max(1, len(local_order) // 3)
        local_testing = local_order[-local_test_count:]
        local_fitting = np.setdiff1d(local_order, local_testing)
        local_result.update({
            "fitting_count": len(local_fitting),
            "withheld_count": len(local_testing),
        })
        local_matrix, local_inliers = cv2.findHomography(
            projection_image[local_fitting], pitch[local_fitting], cv2.RANSAC,
            float(policy["ransac_reprojection_threshold_m"]),
        )
        if local_matrix is not None:
            local_projected = cv2.perspectiveTransform(
                projection_image[local_testing, None], local_matrix,
            )[:, 0]
            local_errors = np.linalg.norm(
                local_projected - pitch[local_testing], axis=1,
            )
            if np.isfinite(local_errors).all():
                local_hull = cv2.convexHull(projection_image[local_fitting].astype(np.float32))
                local_outside = [
                    cv2.pointPolygonTest(local_hull, tuple(map(float, point)), False) < 0
                    for point in projection_image[local_testing]
                ]
                local_result = {
                    **local_result,
                    "status": "MEASURED_DIAGNOSTIC_ONLY",
                    "fitting_count": len(local_fitting),
                    "withheld_count": len(local_testing),
                    "ransac_inliers": int(local_inliers.sum()),
                    "median_pitch_error_m": float(np.median(local_errors)),
                    "p90_pitch_error_m": float(np.percentile(local_errors, 90)),
                    "outside_fit_hull_fraction": float(np.mean(local_outside)),
                    "deployment": "BLOCKED: fewer than 50 withheld correspondences in three separated blocks",
                    "matrix_image_to_pitch_m": local_matrix.tolist(),
                }
            else:
                local_result["status"] = "NONFINITE_LOCAL_PROJECTION"
        else:
            local_result["status"] = "LOCAL_HOMOGRAPHY_FIT_FAILED"
    result = {
        "schema_version": "football.backbone-regional-pitch-mapping.0.6",
        "status": "PASS" if gate_pass else "BLOCKED_VALIDATION_GATE",
        "mapping_type": policy["mapping_type"],
        "coordinate_transform": coordinate_transform,
        "match": match,
        "video_path": str(video_path.resolve()),
        "video_sha256": file_digest(video_path),
        "video_dimensions_px": [width, height],
        "keypoints_path": str(point_path.resolve()),
        "keypoints_sha256": file_digest(point_path),
        "keypoints_declared_sha256": keypoint_entry["sha256"],
        "intrinsics_path": str(intrinsics_path.resolve()),
        "intrinsics_sha256": file_digest(intrinsics_path),
        "intrinsics_declared_sha256": intrinsics_entry["sha256"],
        "release_metadata_path": str(metadata_path.resolve()),
        "release_revision": "325bd4c57ee8e135ab652f3921d274526ada496b",
        "correspondences": len(pitch),
        "withheld_correspondences": test_count,
        "separated_blocks": measured_blocks,
        "median_pitch_error_m": median,
        "p90_pitch_error_m": p90,
        "blocks": rows,
        "withheld_landmark_residuals": landmark_residuals,
        "local_region_attempt": local_result,
        "policy": policy,
        "independence_limits": [
            "Annotation-assisted; not an autonomous video-only calibration.",
            "The upstream release does not establish that lens intrinsics were fitted independently of these keypoints.",
            "Spatial pitch bands are evaluation blocks, not claims about physical cameras.",
            "No seam model, temporally causal calibration, player identity, team, or goalkeeper inference is provided.",
        ],
        "deployment": "NOT EXPORTED_TO_OBSERVATION_ADAPTER",
    }
    overlay_path = output_path.with_name(
        f"{output_path.stem}_pitch_residual_overlay.png",
    )
    _write_pitch_residual_overlay(overlay_path, landmark_residuals)
    result["pitch_residual_overlay"] = str(overlay_path.resolve())
    write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.metadata.resolve(), args.policy.resolve(), args.video.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps({
        "status": result["status"],
        "withheld_correspondences": result["withheld_correspondences"],
        "separated_blocks": result["separated_blocks"],
        "median_pitch_error_m": result["median_pitch_error_m"],
        "p90_pitch_error_m": result["p90_pitch_error_m"],
        "residual_landmark_count": len(result["withheld_landmark_residuals"]),
        "pitch_residual_overlay": result["pitch_residual_overlay"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
