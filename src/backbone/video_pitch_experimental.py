"""Experimental, annotation-assisted local video-to-pitch diagnostics and replay."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from src.backbone.actions import ActionModelSession
from src.backbone.external import ExternalObservationSource, VERSION as EXTERNAL_SCHEMA_VERSION
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "outputs" / "backbone_v0_6_20261021" / "stage_b"
CALIBRATION = PRIOR / "calibration_assisted" / "video" / "revision_metadata"
VIDEO_ROOT = ROOT / "data" / "video_review_v1_2" / "videos"
CLIP_ROOT = PRIOR / "video_corpus" / "fitting"
BASELINE_CONFIG = ROOT / "configs" / "backbone_release_baseline.json"
SAMPLE_FPS = 5
NATIVE_VIDEO_FPS = 25
ROI_HALF_WIDTH_PX = 650.0
LOCAL_MATCH = "117093"
LOCAL_HALF = "1st"
LOCAL_START_S = 1500
LOCAL_DURATION_S = 60


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _landmark_data(match: str) -> tuple[np.ndarray, np.ndarray, list[str], dict, dict]:
    suffix = "_2nd" if match == "118578" else ""
    mapping_path = ROOT / "outputs" / "video_pitch_bridge_20261027" / f"mapping_{match}{suffix}.json"
    mapping = _read_json(mapping_path)
    points_path = Path(mapping["keypoints_path"])
    intrinsics_path = Path(mapping["intrinsics_path"])
    if file_digest(points_path) != mapping["keypoints_sha256"]:
        raise ValueError(f"Calibration landmarks changed: {points_path}")
    if file_digest(intrinsics_path) != mapping["intrinsics_sha256"]:
        raise ValueError(f"Calibration intrinsics changed: {intrinsics_path}")
    raw = _read_json(points_path)
    labels = list(raw)
    image = np.asarray(list(raw.values()), dtype=np.float64)
    pitch = np.asarray([ast.literal_eval(label) for label in labels], dtype=np.float64)
    with np.load(intrinsics_path, allow_pickle=False) as archive:
        camera, distortion, rectified_camera = (
            archive["K"], archive["D"], archive["Knew"],
        )
    rectified = cv2.fisheye.undistortPoints(
        image[:, None], camera, distortion, P=rectified_camera,
    )[:, 0]
    if image.shape != pitch.shape or not np.isfinite(rectified).all():
        raise ValueError(f"Invalid field landmark arrays for match {match}")
    return image, pitch, labels, mapping, {
        "camera": camera,
        "distortion": distortion,
        "rectified_camera": rectified_camera,
        "rectified": rectified,
    }


def _inside_hull(points: np.ndarray, hull: np.ndarray) -> np.ndarray:
    return np.asarray([
        cv2.pointPolygonTest(hull, tuple(map(float, point)), False) >= 0
        for point in points
    ], dtype=bool)


def _error_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    errors = predicted - target
    magnitudes = np.linalg.norm(errors, axis=1)
    return {
        "count": int(len(magnitudes)),
        "median_m": float(np.median(magnitudes)),
        "p90_m": float(np.percentile(magnitudes, 90)),
        "mean_residual_vector_m": errors.mean(axis=0).tolist(),
        "median_residual_vector_m": np.median(errors, axis=0).tolist(),
        "residual_rms_m": float(np.sqrt(np.mean(np.square(errors)))),
    }


def _cross_fitted_predictions(
    points: np.ndarray, pitch: np.ndarray, fit_indices: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    predicted, actual = [], []
    for held_index in fit_indices:
        subfit = fit_indices[fit_indices != held_index]
        matrix, _ = cv2.findHomography(
            points[subfit], pitch[subfit], cv2.RANSAC, threshold_m,
        )
        if matrix is None:
            continue
        point = cv2.perspectiveTransform(points[[held_index], None], matrix)[0, 0]
        if np.isfinite(point).all():
            predicted.append(point)
            actual.append(pitch[held_index])
    if len(predicted) < 6:
        raise ValueError("Too few fitting-only cross-fitted landmarks for residual correction")
    return np.asarray(predicted), np.asarray(actual)


def _fit_residual_corrections(
    crossfit_prediction: np.ndarray, crossfit_actual: np.ndarray,
) -> dict:
    translation = np.median(crossfit_actual - crossfit_prediction, axis=0)
    design = np.column_stack((crossfit_prediction, np.ones(len(crossfit_prediction))))
    coefficients, _, _, _ = np.linalg.lstsq(design, crossfit_actual, rcond=None)
    affine = coefficients[:2].T
    offset = coefficients[2]
    u, singular_values, vh = np.linalg.svd(affine)
    rotation = u @ vh
    angle = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return {
        "translation_correction_m": translation.tolist(),
        "affine_correction": {
            "matrix": affine.tolist(),
            "offset_m": offset.tolist(),
            "rotation_degrees": float(angle),
            "principal_scales": singular_values.tolist(),
            "anisotropy_ratio": float(max(singular_values) / max(min(singular_values), 1e-12)),
        },
        "training_only_crossfit_count": int(len(crossfit_prediction)),
        "training_only_translation_fit_error": _error_metrics(
            crossfit_prediction + translation, crossfit_actual,
        ),
        "training_only_affine_fit_error": _error_metrics(
            design @ coefficients, crossfit_actual,
        ),
    }


def _pitch_residual_overlay(
    path: Path, pitch: np.ndarray, fitting_indices: np.ndarray,
    test_indices: np.ndarray, predicted: np.ndarray, inside: np.ndarray,
    errors: np.ndarray, match: str, fold: int,
) -> None:
    scale = 8.0
    margin = 70
    width, height = round(105 * scale + 2 * margin), round(68 * scale + 2 * margin)
    canvas = np.full((height, width, 3), (246, 246, 246), np.uint8)
    left, top = margin, margin
    right, bottom = left + round(105 * scale), top + round(68 * scale)
    cv2.rectangle(canvas, (left, top), (right, bottom), (30, 120, 30), 2)
    cv2.line(canvas, (left + round(52.5 * scale), top),
             (left + round(52.5 * scale), bottom), (170, 170, 170), 1)
    cv2.circle(canvas, (left + round(52.5 * scale), top + round(34 * scale)),
               round(9.15 * scale), (170, 170, 170), 1)

    def xy(point):
        return (left + round(float(point[0]) * scale),
                bottom - round(float(point[1]) * scale))

    for index in fitting_indices:
        cv2.circle(canvas, xy(pitch[index]), 3, (150, 150, 150), -1)
    for local_index, index in enumerate(test_indices):
        actual = xy(pitch[index])
        estimate = xy(predicted[local_index])
        color = (0, 150, 0) if inside[local_index] else (0, 145, 255)
        cv2.circle(canvas, actual, 6, color, -1)
        cv2.drawMarker(canvas, estimate, (20, 20, 220), cv2.MARKER_CROSS, 14, 2)
        cv2.arrowedLine(canvas, actual, estimate, (20, 20, 220), 2, tipLength=0.18)
        status = "IN" if inside[local_index] else "OUT"
        cv2.putText(canvas, f"{index} {status} {errors[local_index]:.1f}m",
                    (actual[0] + 8, actual[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{match} held-out block {fold}; green=inside fit hull, orange=outside",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write landmark residual overlay: {path}")


def diagnose_global_mappings(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    csv_path = output / "global_landmark_residuals.csv"
    csv_fields = [
        "match", "fold", "point_index", "pitch_key", "image_x_px", "image_y_px",
        "target_x_m", "target_y_m", "predicted_x_m", "predicted_y_m",
        "residual_x_m", "residual_y_m", "residual_norm_m", "inside_fit_hull",
        "translation_corrected_x_m", "translation_corrected_y_m",
        "translation_corrected_residual_norm_m",
        "affine_corrected_x_m", "affine_corrected_y_m",
        "affine_corrected_residual_norm_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for match in ("117093", "118578"):
            image, pitch, labels, mapping, transform = _landmark_data(match)
            points = transform["rectified"]
            order = np.argsort(pitch[:, 0], kind="stable")
            blocks = np.array_split(order, 3)
            fold_summary = []
            for fold, test_indices in enumerate(blocks):
                fit_indices = np.setdiff1d(np.arange(len(pitch)), test_indices)
                block_record = mapping["blocks"][fold]
                matrix = np.asarray(block_record["matrix_image_to_pitch_m"], dtype=np.float64)
                prediction = cv2.perspectiveTransform(
                    points[test_indices, None], matrix,
                )[:, 0]
                hull = cv2.convexHull(points[fit_indices].astype(np.float32))
                inside = _inside_hull(points[test_indices], hull)
                crossfit_prediction, crossfit_actual = _cross_fitted_predictions(
                    points, pitch, fit_indices, 2.0,
                )
                corrections = _fit_residual_corrections(crossfit_prediction, crossfit_actual)
                translation = np.asarray(corrections["translation_correction_m"])
                affine = corrections["affine_correction"]
                affine_matrix = np.asarray(affine["matrix"])
                affine_offset = np.asarray(affine["offset_m"])
                translated = prediction + translation
                affine_prediction = prediction @ affine_matrix.T + affine_offset
                residual = prediction - pitch[test_indices]
                translated_residual = translated - pitch[test_indices]
                affine_residual = affine_prediction - pitch[test_indices]
                overlay = output / f"{match}_fold_{fold}_pitch_residuals.png"
                _pitch_residual_overlay(
                    overlay, pitch, fit_indices, test_indices, prediction,
                    inside, np.linalg.norm(residual, axis=1), match, fold,
                )
                rows = []
                for local_index, index in enumerate(test_indices):
                    row = {
                        "match": match,
                        "fold": fold,
                        "point_index": int(index),
                        "pitch_key": labels[index],
                        "image_x_px": float(image[index, 0]),
                        "image_y_px": float(image[index, 1]),
                        "target_x_m": float(pitch[index, 0]),
                        "target_y_m": float(pitch[index, 1]),
                        "predicted_x_m": float(prediction[local_index, 0]),
                        "predicted_y_m": float(prediction[local_index, 1]),
                        "residual_x_m": float(residual[local_index, 0]),
                        "residual_y_m": float(residual[local_index, 1]),
                        "residual_norm_m": float(np.linalg.norm(residual[local_index])),
                        "inside_fit_hull": bool(inside[local_index]),
                        "translation_corrected_x_m": float(translated[local_index, 0]),
                        "translation_corrected_y_m": float(translated[local_index, 1]),
                        "translation_corrected_residual_norm_m": float(
                            np.linalg.norm(translated_residual[local_index]),
                        ),
                        "affine_corrected_x_m": float(affine_prediction[local_index, 0]),
                        "affine_corrected_y_m": float(affine_prediction[local_index, 1]),
                        "affine_corrected_residual_norm_m": float(
                            np.linalg.norm(affine_residual[local_index]),
                        ),
                    }
                    writer.writerow(row)
                    rows.append(row)
                before = _error_metrics(prediction, pitch[test_indices])
                translation_after = _error_metrics(translated, pitch[test_indices])
                affine_after = _error_metrics(affine_prediction, pitch[test_indices])
                inside_metrics = (
                    _error_metrics(prediction[inside], pitch[test_indices][inside])
                    if inside.any() else None
                )
                fold_summary.append({
                    "fold": fold,
                    "fitting_count": int(len(fit_indices)),
                    "withheld_count": int(len(test_indices)),
                    "withheld_inside_fit_hull": int(inside.sum()),
                    "extrapolation_fraction": float(1 - inside.mean()),
                    "baseline_all_withheld": before,
                    "baseline_inside_hull_only": inside_metrics,
                    "fitting_only_correction": corrections,
                    "heldout_after_translation_correction": translation_after,
                    "heldout_after_affine_correction": affine_after,
                    "diagnostic_overlay": str(overlay.relative_to(ROOT)),
                    "landmarks": rows,
                })
            results[match] = {
                "coordinate_transform": mapping.get("coordinate_transform", "fisheye_undistorted"),
                "source_video_sha256": mapping["video_sha256"],
                "keypoint_sha256": mapping["keypoints_sha256"],
                "intrinsics_sha256": mapping["intrinsics_sha256"],
                "folds": fold_summary,
                "interpretation": (
                    "Corrections are fitted from cross-fitted predictions of the fitting landmarks only; "
                    "the withheld spatial block is used only for evaluation. Metrics include extrapolated points "
                    "separately from points inside the fitting convex hull."
                ),
            }
    report = {
        "schema_version": "football.experimental-calibration-residual-diagnosis.1",
        "status": "DIAGNOSTIC_ONLY",
        "residual_convention": "predicted pitch coordinate minus released field landmark, x/y in metres",
        "global_validation_gate": "UNCHANGED_FAILED",
        "correction_policy": (
            "No global scalar offset is deployed. Translation and affine corrections are separately fitted "
            "on training-only cross-fitted residuals and tested on the withheld spatial block."
        ),
        "residual_csv": str(csv_path.relative_to(ROOT)),
        "matches": results,
    }
    write_json(output / "global_residual_diagnosis.json", report)
    return report


def compare_author_homography(output: Path) -> dict:
    image, pitch, labels, _, transform = _landmark_data(LOCAL_MATCH)
    principal_x = float(transform["camera"][0, 2])
    in_roi = np.flatnonzero(
        np.abs(image[:, 0] - principal_x) <= ROI_HALF_WIDTH_PX,
    )
    order = in_roi[np.argsort(pitch[in_roi, 0], kind="stable")]
    validation_indices = order[::4]
    fitting_indices = np.setdiff1d(order, validation_indices)
    rectified = transform["rectified"]
    fit_native = image[fitting_indices]
    matrix_pitch_to_image, _ = cv2.findHomography(
        pitch[fitting_indices], fit_native, 0,
    )
    if matrix_pitch_to_image is None:
        raise RuntimeError("Dataset-compatible pitch-to-image homography did not fit")
    author_inverse = np.linalg.inv(matrix_pitch_to_image)
    author_predictions = cv2.perspectiveTransform(
        image[validation_indices, None], author_inverse,
    )[:, 0]
    direct_matrix, _ = cv2.findHomography(
        fit_native, pitch[fitting_indices], 0,
    )
    direct_predictions = cv2.perspectiveTransform(
        image[validation_indices, None], direct_matrix,
    )[:, 0]
    delta = np.linalg.norm(author_predictions - direct_predictions, axis=1)
    hull = cv2.convexHull(fit_native.astype(np.float32))
    inside = _inside_hull(image[validation_indices], hull)
    fit_rectified = rectified[fitting_indices]
    rectified_pitch_to_image, _ = cv2.findHomography(
        pitch[fitting_indices], fit_rectified, 0,
    )
    if rectified_pitch_to_image is None:
        raise RuntimeError("Rectified-coordinate diagnostic homography did not fit")
    rectified_inverse = np.linalg.inv(rectified_pitch_to_image)
    rectified_predictions = cv2.perspectiveTransform(
        rectified[validation_indices, None], rectified_inverse,
    )[:, 0]
    rectified_hull = cv2.convexHull(fit_rectified.astype(np.float32))
    rectified_inside = _inside_hull(rectified[validation_indices], rectified_hull)
    predictions = []
    for local_index, index in enumerate(validation_indices):
        predictions.append({
            "point_index": int(index),
            "pitch_key": labels[index],
            "image_xy_native_px": image[index].tolist(),
            "image_xy_rectified_px": rectified[index].tolist(),
            "ground_truth_pitch_xy_m": pitch[index].tolist(),
            "author_native_inverse_homography_pitch_xy_m": author_predictions[local_index].tolist(),
            "native_direct_homography_pitch_xy_m": direct_predictions[local_index].tolist(),
            "rectified_inverse_homography_pitch_xy_m": rectified_predictions[local_index].tolist(),
            "prediction_parity_delta_m": float(delta[local_index]),
            "heldout_error_m": float(np.linalg.norm(author_predictions[local_index] - pitch[index])),
            "inside_fitting_hull": bool(inside[local_index]),
        })
    camera_json_available = False
    result = {
        "dataset_author_source": (
            "https://github.com/SoccerNet/sn-gamestate/blob/"
            "1c958345067218297d221e45e1a6405f975f83e0/"
            "sn_gamestate/calibration/bbox2pitch.py"
        ),
        "author_homography_behavior": (
            "For a list/array homography, get_bbox_pitch_homography inverts it and applies the inverse "
            "to each native-image bottom-left, bottom-right and bottom-middle point. The primary parity "
            "check below therefore fits and evaluates in native image pixels."
        ),
        "author_camera_parameter_behavior": (
            "For a dict of per-image camera parameters, Camera.unproject_point_on_planeZ0 uses the "
            "perspective camera and the dataset's 12-coefficient radial/tangential/thin-prism distortion model."
        ),
        "local_released_camera_json_parameters_available": camera_json_available,
        "local_release_has": ["K", "D", "Knew", "rvecs", "tvecs", "rms"],
        "local_release_does_not_have": [
            "per-frame baseline Camera JSON parameters",
            "author-produced per-frame homography",
            "author per-image line predictions",
        ],
        "selected_region": {
            "definition": "absolute native x distance <=650 px from the released camera's optical principal point",
            "principal_point_x_px": principal_x,
            "native_bounds_px": [principal_x - ROI_HALF_WIDTH_PX, principal_x + ROI_HALF_WIDTH_PX],
            "basis": "fixed from camera setup/optical center, not chosen from validation errors",
        },
        "fitting_count": int(len(fitting_indices)),
        "validation_count": int(len(validation_indices)),
        "in_hull_validation_count": int(inside.sum()),
        "rectified_in_hull_validation_count": int(rectified_inside.sum()),
        "author_inverse_validation_error": _error_metrics(author_predictions, pitch[validation_indices]),
        "author_inverse_validation_inside_native_hull": _error_metrics(
            author_predictions[inside], pitch[validation_indices][inside],
        ) if inside.any() else None,
        "rectified_homography_validation_error": _error_metrics(
            rectified_predictions, pitch[validation_indices],
        ),
        "rectified_homography_validation_inside_hull": _error_metrics(
            rectified_predictions[rectified_inside],
            pitch[validation_indices][rectified_inside],
        ) if rectified_inside.any() else None,
        "in_hull_author_inverse_validation_error": (
            _error_metrics(author_predictions[inside], pitch[validation_indices][inside])
            if inside.any() else None
        ),
        "local_assisted_projection_path": (
            "Undistort raw bbox bottom-middle with released K,D,Knew; fit the pitch-to-rectified-image "
            "homography on fitting landmarks and invert it for projection, following the dataset author's "
            "inverse-homography direction. Native-pixel author-path results are reported separately."
        ),
        "direct_vs_author_inverse_parity": {
            "maximum_delta_m": float(delta.max(initial=0)),
            "median_delta_m": float(np.median(delta)),
            "per_landmark": predictions,
            "interpretation": (
                "Both are fits over the same calibration correspondences, not independent validation. "
                "Parity checks the mapping direction/convention; holdout errors assess this small local fit."
            ),
        },
    }
    write_json(output / "author_projection_comparison.json", result)
    return result


def _project_image_points(points: np.ndarray, inverse_homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points[:, None], inverse_homography)[:, 0]


def _build_local_positions(
    author_comparison: dict, start_s: int, duration_s: int,
) -> tuple[list[dict], dict]:
    image, pitch, _, _, transform = _landmark_data(LOCAL_MATCH)
    principal_x = float(transform["camera"][0, 2])
    in_roi = np.flatnonzero(
        np.abs(image[:, 0] - principal_x) <= ROI_HALF_WIDTH_PX,
    )
    order = in_roi[np.argsort(pitch[in_roi, 0], kind="stable")]
    validation_indices = order[::4]
    fitting_indices = np.setdiff1d(order, validation_indices)
    rectified = transform["rectified"]
    fit_rectified = rectified[fitting_indices]
    fit_pitch = pitch[fitting_indices]
    pitch_to_rectified, _ = cv2.findHomography(fit_pitch, fit_rectified, 0)
    if pitch_to_rectified is None:
        raise RuntimeError("Local assisted calibration failed to fit")
    inverse = np.linalg.inv(pitch_to_rectified)
    hull = cv2.convexHull(fit_rectified.astype(np.float32))
    video_path = VIDEO_ROOT / LOCAL_MATCH / f"{LOCAL_MATCH}_panorama_{LOCAL_HALF}_half.mp4"
    clip_folder = CLIP_ROOT / f"{LOCAL_MATCH}_{LOCAL_HALF}_{LOCAL_START_S:05d}s_1200s"
    tracklet_path = clip_folder / "pixel_tracklets.jsonl"
    clip_manifest = _read_json(clip_folder / "clip_manifest.json")
    if file_digest(video_path) != clip_manifest["source_video_sha256"]:
        raise ValueError("Local demo video does not match the stored detection clip")
    if file_digest(tracklet_path) != clip_manifest["observations_sha256"]:
        raise ValueError("Local tracklets do not match the stored detection clip")
    if (clip_manifest["source_dimensions_px"] != [4096, 1080]
            or clip_manifest["source_fps"] != NATIVE_VIDEO_FPS
            or clip_manifest["target_fps"] != SAMPLE_FPS
            or clip_manifest["start_s"] != LOCAL_START_S
            or clip_manifest["duration_s"] != 1200):
        raise ValueError("Stored clip metadata does not match the requested source timebase")

    if start_s < LOCAL_START_S or start_s + duration_s > LOCAL_START_S + 1200:
        raise ValueError("Requested demo interval must fit inside the hashed 1500s-2700s detection clip")
    segment_start = int(start_s * NATIVE_VIDEO_FPS)
    segment_end = segment_start + duration_s * NATIVE_VIDEO_FPS
    frames: list[dict] = []
    with tracklet_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["frame_index"] < segment_start:
                continue
            if row["frame_index"] >= segment_end:
                break
            expected_time = row["frame_index"] / NATIVE_VIDEO_FPS
            if abs(float(row["time_from_video_start_s"]) - expected_time) > 1e-4:
                raise ValueError("Tracklet timestamps do not match the native frame clock")
            if frames and (
                row["frame_index"] - frames[-1]["frame_index"] != NATIVE_VIDEO_FPS // SAMPLE_FPS
                or abs(float(row["time_from_video_start_s"]) - frames[-1]["timestamp_s"] - 0.2) > 1e-4
            ):
                raise ValueError("Tracklet rows are not continuous on the requested 5 Hz grid")
            measurements, projected_detections = [], []
            for detection in row["detections"]:
                box = np.asarray(detection["bbox_xyxy_px"], dtype=np.float64)
                foot = np.asarray([[(box[0] + box[2]) / 2.0, box[3]]], dtype=np.float64)
                inside_image_roi = abs(float(foot[0, 0]) - principal_x) <= ROI_HALF_WIDTH_PX
                rect_foot = cv2.fisheye.undistortPoints(
                    foot[:, None], transform["camera"], transform["distortion"],
                    P=transform["rectified_camera"],
                )[:, 0]
                inside_fit_support = bool(_inside_hull(rect_foot, hull)[0])
                projected = _project_image_points(rect_foot, inverse)[0]
                supported = (
                    inside_image_roi and inside_fit_support and np.isfinite(projected).all()
                    and 0 <= projected[0] <= 105 and 0 <= projected[1] <= 68
                )
                record = {
                    "track": str(detection["tracklet_id"]),
                    "track_status": str(detection["link_status"]),
                    "bbox_xyxy_px": box.tolist(),
                    "footpoint_xy_px": foot[0].tolist(),
                    "pitch_xy_m": projected.tolist() if supported else None,
                    "projection_status": (
                        "EXPERIMENTAL_SUPPORTED_HULL" if supported else
                        "OUTSIDE_OPTICAL_ROI" if not inside_image_roi else
                        "OUTSIDE_FIT_HULL_OR_PITCH_BOUNDS"
                    ),
                    "detector_confidence": float(detection["detector_confidence"]),
                }
                projected_detections.append(record)
                if supported:
                    measurements.append({
                        "track": record["track"],
                        "segment": 0,
                        "team": "UNKNOWN",
                        "goalkeeper": None,
                        "position_status": "observed",
                        "xy": record["pitch_xy_m"],
                        "quality": {
                            "status": "UNVALIDATED_EXPERIMENTAL",
                            "projection": "ASSISTED_LOCAL_HOMOGRAPHY",
                            "detector_confidence": record["detector_confidence"],
                            "local_track_status": record["track_status"],
                            "position_accuracy_m": None,
                        },
                    })
            frames.append({
                "record_type": "frame",
                "frame_index": int(row["frame_index"]),
                "timestamp_s": float(row["time_from_video_start_s"]),
                "segment_id": f"{LOCAL_MATCH}_{LOCAL_HALF}_{start_s}",
                "identity_reset": bool(row.get("potential_cut_before", False)),
                "camera_cut": bool(row.get("potential_cut_before", False)),
                "measurements": measurements,
                "absent_count": int(max(0, len(projected_detections) - len(measurements))),
                "detections": projected_detections,
            })
    expected_count = duration_s * SAMPLE_FPS
    if (len(frames) != expected_count or frames[0]["frame_index"] != segment_start
            or frames[-1]["frame_index"] != segment_end - NATIVE_VIDEO_FPS // SAMPLE_FPS):
        raise ValueError(
            f"Expected {expected_count} continuous observations from native frame "
            f"{segment_start} to {segment_end - NATIVE_VIDEO_FPS // SAMPLE_FPS}; got {len(frames)}"
        )

    track_counts: dict[str, int] = {}
    for frame in frames:
        for measurement in frame["measurements"]:
            track_counts[measurement["track"]] = track_counts.get(measurement["track"], 0) + 1
    if not track_counts:
        raise ValueError("No detector footpoints fell within the locally supported projection area")
    actor_track = max(track_counts, key=lambda track: (track_counts[track], track))
    provenance = {
        "schema_version": "football.assisted-local-demo.1",
        "status": "POSITION_ONLY_EXPERIMENTAL_UNVALIDATED",
        "match": LOCAL_MATCH,
        "half": LOCAL_HALF,
        "video_path": str(video_path.relative_to(ROOT)),
        "video_sha256": file_digest(video_path),
        "source_detection_path": str(tracklet_path.relative_to(ROOT)),
        "source_detection_sha256": file_digest(tracklet_path),
        "clip_manifest_sha256": file_digest(clip_folder / "clip_manifest.json"),
        "start_time_s": start_s,
        "end_time_s_exclusive": start_s + duration_s,
        "sample_rate_hz": SAMPLE_FPS,
        "sample_count": len(frames),
        "continuous_coverage_s": duration_s,
        "pixel_detection_count": sum(len(frame["detections"]) for frame in frames),
        "projected_in_support_count": sum(
            len(frame["measurements"]) for frame in frames
        ),
        "projection_support_fraction": (
            sum(len(frame["measurements"]) for frame in frames)
            / max(1, sum(len(frame["detections"]) for frame in frames))
        ),
        "unique_source_local_tracklets": len(track_counts),
        "selected_actor_tracklet": actor_track,
        "selected_actor_observed_samples": track_counts[actor_track],
        "identity_status": "UNVERIFIED_IMAGE_SPACE_CANDIDATE",
        "team": "UNKNOWN",
        "goalkeeper_role": "UNKNOWN",
        "ball_and_possession": "UNKNOWN",
        "calibration": {
            "type": "ASSISTED_LOCAL_DEMO",
            "setup_region": author_comparison["selected_region"],
            "fitting_count": author_comparison["fitting_count"],
            "validation_count": author_comparison["validation_count"],
            "validation_inside_fit_hull": author_comparison["rectified_in_hull_validation_count"],
            "validation_error_all": author_comparison["rectified_homography_validation_error"],
            "validation_error_inside_fit_hull": author_comparison[
                "rectified_homography_validation_inside_hull"
            ],
            "global_gate_status": "FAILED_UNCHANGED",
            "position_accuracy": "UNVALIDATED",
            "online_calibration": False,
        },
        "measurement_semantics": (
            "Detector bbox bottom-middle footpoint; positions only emitted inside fixed optical ROI "
            "and fitting-landmark convex hull. Detector confidence is not metre accuracy."
        ),
    }
    return frames, {"actor_track": actor_track, "track_counts": track_counts,
                    "provenance": provenance}


def _external_jsonl(path: Path, frames: list[dict], provenance: dict) -> None:
    header = {
        "record_type": "metadata",
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "source_id": f"video-assisted-{frames[0]['segment_id']}",
        "match": LOCAL_MATCH,
        "half": LOCAL_HALF,
        "observation_kind": "exo_source",
        "sampling_seconds": 0.2,
        "missing_count_semantics": "detections outside experimental local support or absent detections; incomplete inventory",
        "coordinates": {
            "system": "pitch_top_left",
            "units": "metres",
            "axes": "x_length_y_width",
            "pitch_length_m": 105,
            "pitch_width_m": 68,
        },
        "origin": {
            "type": "video_detector_footpoints",
            "status": "UNVALIDATED_EXPERIMENTAL",
            "video": provenance["video_path"],
            "detector_and_tracklets": provenance["source_detection_path"],
        },
        "calibration": provenance["calibration"],
        "synchronization": {
            "status": "frame_index_sampled",
            "native_fps": NATIVE_VIDEO_FPS,
            "sample_stride_native_frames": 5,
        },
        "quality": {
            "position_accuracy": "UNVALIDATED",
            "detector_confidence_is_accuracy": False,
            "team_and_goalkeeper_roles": "UNKNOWN",
        },
    }
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(header, allow_nan=False, separators=(",", ":")) + "\n")
        for frame in frames:
            stream.write(json.dumps({
                key: value for key, value in frame.items() if key != "detections"
            }, allow_nan=False, separators=(",", ":")) + "\n")


def _select_action_query_index(frames: list[dict], actor_track: str) -> tuple[int | None, str]:
    actor_present = [
        any(measurement["track"] == actor_track for measurement in frame["measurements"])
        for frame in frames
    ]
    observed_indices = [index for index, present in enumerate(actor_present) if present]
    for index in range(10, len(actor_present)):
        if all(actor_present[index - 10:index + 1]):
            return index, "first supplied actor sample with an 11-sample continuous history"
    if observed_indices:
        return observed_indices[0], "first supplied actor sample; causal history is incomplete"
    return None, "no supported projected actor sample; no action query was made"


def _run_frozen_adapters(output: Path, frames: list[dict], info: dict) -> dict:
    external_path = output / "experimental_observations.jsonl"
    _external_jsonl(external_path, frames, info["provenance"])
    config = _read_json(BASELINE_CONFIG)
    experiment = ROOT / config["experiment"]
    world = ModelSession(experiment, config["world_backend"], device="cpu")
    action = ActionModelSession(ROOT / config["action_checkpoint"], device="cpu")
    source = ExternalObservationSource(external_path)
    actor_track = info["actor_track"]
    provider = SyntheticObservationProvider(
        actor_track, seed=42, fov=120, dropout=False,
    )
    engine = ReplayEngine(
        world, expiry_s=6., fallback="last_seen", action_model=action,
    )
    frames_for_replay = list(source.frames(
        int(frames[0]["frame_index"]), len(frames), stride=5,
    ))
    if len(frames_for_replay) != len(frames):
        raise ValueError("External adapter did not reproduce the continuous 5 Hz source grid")
    query_index, query_basis = _select_action_query_index(frames, actor_track)
    rows = []
    for index, frame in enumerate(frames_for_replay):
        observation = source.observe(frame, provider)
        rows.append(engine.step(
            observation, include_passing_assessment=False, action_query=index == query_index,
        ))
    output_rows = output / "experimental_g1_s4_status.jsonl"
    with output_rows.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    query_rows = [rows[query_index]] if query_index is not None else []
    reasons = sorted({
        row["action_inference"]["reason"] for row in query_rows
        if row.get("action_inference", {}).get("reason")
    })
    world_unavailable = sum(
        any("WORLD_CONTRACT_UNAVAILABLE" in warning for warning in row["belief"]["warnings"])
        for row in rows
    )
    return {
        "g1_checkpoint_loaded_once": world.load_counts["world"] == 1,
        "g1_forward_calls": world.world_forward_calls,
        "g1_explicit_contract_unavailable_samples": world_unavailable,
        "g1_block_reason": (
            "The frozen world-input contract requires known teams and goalkeeper roles over the 11-sample "
            "history; this video stream has UNKNOWN team and role, so no G1 neural forward call is valid."
        ),
        "s4_checkpoint_loaded_once": action.load_count == 1,
        "s4_explicit_query_attempts": len(query_rows),
        "s4_neural_forward_calls": action.forward_calls,
        "s4_query_frame_index": (
            int(frames[query_index]["frame_index"]) if query_index is not None else None
        ),
        "s4_query_time_s": (
            float(frames[query_index]["timestamp_s"]) if query_index is not None else None
        ),
        "s4_query_actor_observed": query_index is not None and any(
            measurement["track"] == actor_track
            for measurement in frames[query_index]["measurements"]
        ),
        "s4_query_selection_basis": query_basis,
        "s4_query_reasons": reasons,
        "s4_block_reason": (
            "The selected actor and 11-sample position history are present at the diagnostic query, but "
            "UNKNOWN team and goalkeeper roles prevent the causal player-history tensor. No event annotation "
            "or action label is invented. S4 remains a supplied-actor/time event-conditioned classifier."
        ),
        "replay_status_file": str(output_rows.relative_to(ROOT)),
        "actor_tracklet": actor_track,
        "action_output": (
            query_rows[0].get("action_inference", {}).get("status", "not_queried")
            if query_rows else "NOT_QUERIED"
        ),
    }


def _render_pitch(
    canvas: np.ndarray, detections: list[dict], bounds: tuple[int, int, int, int],
) -> None:
    x0, y0, width, height = bounds
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (245, 245, 245), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (30, 120, 30), 3)
    cv2.line(canvas, (x0 + width // 2, y0), (x0 + width // 2, y0 + height),
             (130, 130, 130), 2)
    cv2.circle(canvas, (x0 + width // 2, y0 + height // 2), round(height * .134),
               (130, 130, 130), 2)
    for detection in detections:
        xy = detection["pitch_xy_m"]
        if xy is None:
            continue
        px = x0 + round(float(xy[0]) / 105 * width)
        py = y0 + height - round(float(xy[1]) / 68 * height)
        cv2.circle(canvas, (px, py), 7, (20, 40, 230), -1)
        cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1)


def _render_video(output: Path, frames: list[dict], provenance: dict) -> Path:
    video_path = ROOT / provenance["video_path"]
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video for experimental rendering: {video_path}")
    ffmpeg = (
        ROOT / "data" / "video_review_tools" / "portable"
        / "ffmpeg-9.0.2-essentials_build" / "bin" / "ffmpeg.exe"
    )
    if not ffmpeg.is_file():
        capture.release()
        raise FileNotFoundError(f"Existing portable ffmpeg is required: {ffmpeg}")
    mp4_path = output / "assisted_local_demo.mp4"
    process = subprocess.Popen(
        [str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", "1920x1080", "-r", str(SAMPLE_FPS), "-i", "pipe:0",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
         "-crf", "24", "-movflags", "+faststart", str(mp4_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        capture.release()
        process.kill()
        raise RuntimeError("Could not create ffmpeg raw-frame input stream")
    frame_width, frame_height = 4096, 1080
    preview_frame = None
    preview_support = -1
    try:
        for index, item in enumerate(frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, item["frame_index"])
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode source frame {item['frame_index']}")
            canvas = np.full((1080, 1920, 3), (24, 24, 24), np.uint8)
            cv2.putText(
                canvas, "ASSISTED_LOCAL_DEMO | EXPERIMENTAL, UNVALIDATED POSITIONS",
                (35, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 220, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, f"{LOCAL_MATCH} {LOCAL_HALF} | {item['timestamp_s']:.1f}s | "
                "tracklet identity/team/role unverified",
                (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA,
            )
            display = cv2.resize(image, (1200, 316), interpolation=cv2.INTER_AREA)
            sx, sy = 1200 / frame_width, 316 / frame_height
            for detection in item["detections"]:
                x1, y1, x2, y2 = detection["bbox_xyxy_px"]
                color = (0, 190, 255) if detection["pitch_xy_m"] is not None else (130, 130, 130)
                cv2.rectangle(display, (round(x1 * sx), round(y1 * sy)),
                              (round(x2 * sx), round(y2 * sy)), color, 1)
                if detection["pitch_xy_m"] is not None:
                    cv2.putText(display, detection["track"][-4:],
                                (round((x1 + x2) / 2 * sx), round(y2 * sy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            canvas[110:426, 35:1235] = display
            cv2.putText(canvas, "VIDEO DETECTIONS (yellow=projected inside local support; gray=not projected)",
                        (35, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
            cv2.putText(canvas, "EXPERIMENTAL PITCH POSITIONS (not verified player locations)",
                        (1300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
            _render_pitch(canvas, item["detections"], (1300, 150, 560, 363))
            supported = sum(d["pitch_xy_m"] is not None for d in item["detections"])
            cv2.putText(canvas, f"Sample {index + 1}/{len(frames)} | "
                        f"positions {supported}/{len(item['detections'])} | "
                        f"view angle simulated; not measured gaze",
                        (35, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)
            if supported > preview_support:
                preview_support = supported
                preview_frame = canvas.copy()
            process.stdin.write(canvas.tobytes())
    except BaseException:
        capture.release()
        process.stdin.close()
        process.kill()
        process.communicate()
        raise
    else:
        process.stdin.close()
        _, stderr = process.communicate()
        capture.release()
    if process.returncode:
        raise RuntimeError(f"ffmpeg could not encode demo: {stderr.decode(errors='replace')[-2000:]}")
    if preview_frame is None or not cv2.imwrite(str(output / "preview_projected.jpg"), preview_frame):
        raise RuntimeError("Could not save a representative projected-frame preview")
    return mp4_path


def _write_html_viewer(output: Path) -> Path:
    path = output / "assisted_local_demo.html"
    path.write_text(
        """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assisted local video-to-pitch demo</title>
<style>
body { margin: 2rem auto; max-width: 1200px; padding: 0 1rem; font: 16px system-ui, sans-serif; background: #101820; color: #eef4f7; }
video { width: 100%; background: #000; }
</style>
<h1>ASSISTED_LOCAL_DEMO</h1>
<p>Experimental, unvalidated video detections and local pitch projections. No team, role, identity, ball, or possession labels are inferred.</p>
<video controls preload="metadata" poster="preview_projected.jpg">
  <source src="assisted_local_demo.mp4" type="video/mp4">
</video>
</html>
""",
        encoding="utf-8",
    )
    return path


def run(
    output: Path, start_s: int = LOCAL_START_S, duration_s: int = LOCAL_DURATION_S,
) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experimental output directory: {output}")
    if duration_s < 60 or duration_s > 120:
        raise ValueError("Experimental local replay duration must be between 60 and 120 seconds")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    diagnose_global_mappings(output)
    author_comparison = compare_author_homography(output)
    frames, info = _build_local_positions(author_comparison, start_s, duration_s)
    provenance = info["provenance"]

    write_json(output / "assisted_local_policy.json", {
        "schema_version": "football.assisted-local-demo-policy.1",
        "status": "FROZEN_FOR_EXPLORATORY_RUN",
        "region_definition": author_comparison["selected_region"],
        "calibration_split": {
            "selection": "Sort keypoints inside ROI by pitch x; hold every fourth point out, all others fit.",
            "fitting_count": author_comparison["fitting_count"],
            "validation_count": author_comparison["validation_count"],
            "no_validation_driven_region_selection": True,
        },
        "mapping": "Released fisheye K,D,Knew rectification, then pitch-to-image homography fitted on fit points; project by its inverse, matching dataset author homography convention.",
        "projection_domain": "Native image optical ROI AND rectified fit-point convex hull; abstain outside support.",
        "validation_metrics": author_comparison["rectified_homography_validation_error"],
        "inside_fit_hull_validation_count": author_comparison[
            "rectified_in_hull_validation_count"
        ],
        "status": "EXPERIMENTAL_ONLY; not global calibration certification or player-position accuracy",
    })
    write_json(output / "assisted_local_provenance.json", provenance)
    rows_path = output / "projected_detections.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, allow_nan=False, separators=(",", ":")) + "\n")
    adapter_status = _run_frozen_adapters(output, frames, info)
    video_output = _render_video(output, frames, provenance)
    viewer_output = _write_html_viewer(output)
    result = {
        "schema_version": "football.assisted-local-demo-result.1",
        "status": "POSITION_ONLY_EXPERIMENTAL_UNVALIDATED",
        "residual_diagnosis": str((output / "global_residual_diagnosis.json").relative_to(ROOT)),
        "author_projection_comparison": str((output / "author_projection_comparison.json").relative_to(ROOT)),
        "continuous_segment": {
            "start_s": start_s,
            "duration_s": duration_s,
            "sample_hz": SAMPLE_FPS,
            "samples": len(frames),
            "video": str(video_output.relative_to(ROOT)),
            "viewer": str(viewer_output.relative_to(ROOT)),
            "video_sha256": file_digest(video_output),
            "projected_rows": str(rows_path.relative_to(ROOT)),
            "projected_rows_sha256": file_digest(rows_path),
        },
        "position_quality": provenance,
        "learned_model_calls": adapter_status,
        "action_status": "UNSUPPORTED; no validated actor/team/role input or event label",
        "training_or_benchmark_admission": False,
        "global_validation_gate": "FAILED_UNCHANGED",
    }
    write_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/assisted_local_video_demo_20261027"),
    )
    parser.add_argument("--start-s", type=int, default=LOCAL_START_S)
    parser.add_argument("--duration-s", type=int, default=LOCAL_DURATION_S)
    args = parser.parse_args()
    result = run(args.output, start_s=args.start_s, duration_s=args.duration_s)
    print(json.dumps({
        "status": result["status"],
        "continuous_samples": result["continuous_segment"]["samples"],
        "video": result["continuous_segment"]["video"],
        "projected_count": result["position_quality"]["projected_in_support_count"],
        "g1_forward_calls": result["learned_model_calls"]["g1_forward_calls"],
        "s4_neural_forward_calls": result["learned_model_calls"]["s4_neural_forward_calls"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
