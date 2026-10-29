"""Config-driven detection and scoring for cross-match video-player transfer."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone

import cv2
import ijson
import numpy as np
import torch

from src.backbone.video_player_overnight import (
    TILE_COLUMNS,
    _tiles,
    detect_native,
)
from src.backbone.video_player_replay import (
    TRACK_MAX_DISTANCE_PX,
    TRACK_MAX_GAP_SAMPLES,
    _assign_tracklets,
)
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs" / "video_player_cross_match_protocol.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "video_player_cross_match_20261028_130610"
SAMPLE_STRIDE = 5
NATIVE_FPS = 25
DETECTION_CONFIDENCE = 0.20


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "football.video-player-cross-match-protocol.1":
        raise ValueError(f"Unsupported protocol schema: {path}")
    return protocol


def source_by_id(protocol: dict, match_id: str) -> dict:
    for source in (
        protocol["selection"]["development_match"],
        protocol["selection"]["evaluation_match"],
    ):
        if source["id"] == match_id:
            return source
    raise ValueError(f"Match {match_id} is not in the locked protocol")


def fit_local_projector(protocol: dict, output: Path, match_id: str) -> dict:
    source = source_by_id(protocol, match_id)
    keypoints_path = ROOT / source["calibration_keypoints"]
    intrinsics_path = ROOT / source["camera_intrinsics"]
    if file_digest(keypoints_path) != source["keypoints_sha256"]:
        raise ValueError(f"Calibration keypoints changed: {keypoints_path}")
    if file_digest(intrinsics_path) != source["camera_intrinsics_sha256"]:
        raise ValueError(f"Camera intrinsics changed: {intrinsics_path}")
    keypoints = json.loads(keypoints_path.read_text(encoding="utf-8"))
    pitch = np.asarray(
        [ast.literal_eval(label) for label in keypoints],
        dtype=np.float64,
    )
    image = np.asarray(list(keypoints.values()), dtype=np.float64)
    if image.shape != pitch.shape or image.ndim != 2 or image.shape[1] != 2:
        raise ValueError("Released setup keypoints must contain paired 2D coordinates")
    if not np.isfinite(image).all() or not np.isfinite(pitch).all():
        raise ValueError("Released setup keypoints contain non-finite coordinates")
    with np.load(intrinsics_path, allow_pickle=False) as archive:
        camera, distortion, rectified_camera = (
            archive["K"], archive["D"], archive["Knew"],
        )
    rectified = cv2.fisheye.undistortPoints(
        image[:, None], camera, distortion, P=rectified_camera,
    )[:, 0]
    principal_x = float(camera[0, 2])
    region_indices = np.flatnonzero(np.abs(image[:, 0] - principal_x) <= 650.0)
    ordered = region_indices[
        np.argsort(pitch[region_indices, 0], kind="stable")
    ]
    validation_indices = ordered[::4]
    fitting_indices = np.setdiff1d(ordered, validation_indices)
    if len(fitting_indices) < 4 or len(validation_indices) == 0:
        raise ValueError(
            f"Too few setup landmarks in the fixed local ROI for {match_id}: "
            f"{len(fitting_indices)} fitting, {len(validation_indices)} validation"
        )
    pitch_to_rectified, _ = cv2.findHomography(
        pitch[fitting_indices], rectified[fitting_indices], 0,
    )
    if pitch_to_rectified is None or not np.isfinite(pitch_to_rectified).all():
        raise RuntimeError(f"Local setup homography failed for {match_id}")
    rectified_to_pitch = np.linalg.inv(pitch_to_rectified)
    hull = cv2.convexHull(rectified[fitting_indices].astype(np.float32))
    predicted = cv2.perspectiveTransform(
        rectified[validation_indices, None], rectified_to_pitch,
    )[:, 0]
    residuals = predicted - pitch[validation_indices]
    errors = np.linalg.norm(residuals, axis=1)
    inside = np.asarray([
        cv2.pointPolygonTest(hull, tuple(map(float, rectified[index])), False) >= 0
        for index in validation_indices
    ])
    inside_errors = errors[inside]
    median = float(np.median(errors))
    p90 = float(np.percentile(errors, 90))
    gate_pass = (
        len(validation_indices) >= 50
        and median <= 2.0
        and p90 <= 5.0
    )
    result = {
        "schema_version": "football.video-player-cross-match-projector.1",
        "status": "EXPERIMENTAL_UNVALIDATED_LOCAL_PROJECTOR",
        "match_id": match_id,
        "source_video_sha256": source["video_sha256"],
        "video_dimensions_px": source["dimensions_px"],
        "native_fps": source["fps"],
        "keypoints_path": str(keypoints_path.relative_to(ROOT)),
        "keypoints_sha256": source["keypoints_sha256"],
        "intrinsics_path": str(intrinsics_path.relative_to(ROOT)),
        "intrinsics_sha256": source["camera_intrinsics_sha256"],
        "coordinate_contract": {
            "input": "raw native panorama pixel coordinates",
            "distortion": "per-recording K,D,Knew fisheye undistortPoints",
            "projection": "rectified image pixels to pitch top-left metres via inverse homography",
            "source_image_crop_or_resize": "none",
            "detector_tile_offsets": "already restored to raw native panorama coordinates",
            "pitch_axes": "x in [0,105], y in [0,68]",
        },
        "fixed_roi": {
            "rule": "abs(raw_image_x - this recording's K principal_x) <= 650 native pixels",
            "principal_x_px": principal_x,
            "half_width_px": 650.0,
            "setup_landmark_count": int(len(region_indices)),
        },
        "split": {
            "rule": "sort ROI setup landmarks by pitch x; hold every fourth, fit remaining",
            "fitting_indices": fitting_indices.astype(int).tolist(),
            "validation_indices": validation_indices.astype(int).tolist(),
            "fitting_count": int(len(fitting_indices)),
            "validation_count": int(len(validation_indices)),
            "fit_hull_rectified_px": hull[:, 0, :].astype(float).tolist(),
        },
        "validation": {
            "median_error_m": median,
            "p90_error_m": p90,
            "in_hull_count": int(inside.sum()),
            "out_of_hull_count": int((~inside).sum()),
            "in_hull_median_error_m": (
                float(np.median(inside_errors)) if len(inside_errors) else None
            ),
            "in_hull_p90_error_m": (
                float(np.percentile(inside_errors, 90)) if len(inside_errors) else None
            ),
            "residual_vectors_m": residuals.astype(float).tolist(),
            "error_m": errors.astype(float).tolist(),
            "inside_fit_hull": inside.tolist(),
            "acceptance_gate_pass": gate_pass,
            "acceptance_rule": "at least 50 held-out points, median <=2m, p90 <=5m",
        },
        "mapping": {
            "camera_K": camera.astype(float).tolist(),
            "distortion_D": distortion.astype(float).tolist(),
            "rectified_camera_Knew": rectified_camera.astype(float).tolist(),
            "rectified_to_pitch_homography": rectified_to_pitch.astype(float).tolist(),
            "fit_hull_rectified_px": hull[:, 0, :].astype(float).tolist(),
        },
        "limitations": [
            "This is setup-assisted and remains experimental; the local validation denominator is below 50.",
            "Global source-specific calibration gate remains failed; no held-out player coordinates were used.",
            "Projection does not establish player detection, identity, team, role, or tactical accuracy.",
        ],
    }
    projector_path = output / "calibration" / f"local_projector_{match_id}.json"
    if projector_path.exists():
        raise FileExistsError(f"Refusing to overwrite projector: {projector_path}")
    write_json(projector_path, result)
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in {"mapping", "split", "validation"}
    } | {
        "validation": {
            key: value for key, value in result["validation"].items()
            if key not in {"residual_vectors_m", "error_m", "inside_fit_hull"}
        },
    }, indent=2))
    return result


def project_footpoint(
    point_xy_px: list[float],
    projector: dict,
) -> tuple[list[float] | None, str]:
    mapping = projector["mapping"]
    point = np.asarray(point_xy_px, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("A footpoint must be a finite native-image xy pair")
    fixed_roi = projector["fixed_roi"]
    if abs(float(point[0]) - fixed_roi["principal_x_px"]) > fixed_roi["half_width_px"]:
        return None, "OUTSIDE_FIXED_SETUP_ROI"
    rectified = cv2.fisheye.undistortPoints(
        point[None, None],
        np.asarray(mapping["camera_K"], dtype=np.float64),
        np.asarray(mapping["distortion_D"], dtype=np.float64),
        P=np.asarray(mapping["rectified_camera_Knew"], dtype=np.float64),
    )[0, 0]
    hull = np.asarray(
        mapping["fit_hull_rectified_px"], dtype=np.float32,
    ).reshape(-1, 1, 2)
    if cv2.pointPolygonTest(hull, tuple(map(float, rectified)), False) < 0:
        return None, "OUTSIDE_FIT_LANDMARK_HULL"
    pitch = cv2.perspectiveTransform(
        rectified.reshape(1, 1, 2),
        np.asarray(mapping["rectified_to_pitch_homography"], dtype=np.float64),
    )[0, 0]
    if not np.isfinite(pitch).all() or not (0 <= pitch[0] <= 105 and 0 <= pitch[1] <= 68):
        return None, "OUTSIDE_PITCH_BOUNDS"
    return pitch.astype(float).tolist(), "UNVALIDATED_EXPERIMENTAL_LOCAL_PROJECTION"


def load_local_projector(protocol: dict, output: Path, match_id: str) -> dict:
    source = source_by_id(protocol, match_id)
    path = output / "calibration" / f"local_projector_{match_id}.json"
    projector = json.loads(path.read_text(encoding="utf-8"))
    if projector.get("schema_version") != "football.video-player-cross-match-projector.1":
        raise ValueError(f"Unsupported projector artifact: {path}")
    if (
        projector.get("match_id") != match_id
        or projector.get("source_video_sha256") != source["video_sha256"]
        or file_digest(ROOT / source["calibration_keypoints"])
        != projector.get("keypoints_sha256")
        or file_digest(ROOT / source["camera_intrinsics"])
        != projector.get("intrinsics_sha256")
    ):
        raise ValueError(f"Projector provenance no longer matches locked source: {path}")
    return projector


def validate_final_prediction_set(protocol: dict, output: Path) -> dict:
    routes = ("frozen", "adapted")
    training_provenance_path = (
        output / "detector_adaptation_5hz_run2" / "training_provenance.json"
    )
    training_provenance = json.loads(
        training_provenance_path.read_text(encoding="utf-8"),
    )
    adapted_weights = ROOT / training_provenance["best_checkpoint"]
    if file_digest(adapted_weights) != training_provenance["best_checkpoint_sha256"]:
        raise ValueError("Adapted detector checkpoint differs from its training provenance")
    frozen_weights = ROOT / protocol["preservation"]["source_detector"]
    if file_digest(frozen_weights) != protocol["preservation"]["source_detector_sha256"]:
        raise ValueError("Frozen detector checkpoint differs from the locked source hash")
    if (
        training_provenance.get("base_checkpoint_sha256")
        != protocol["preservation"]["source_detector_sha256"]
    ):
        raise ValueError("Adapted detector provenance does not name the locked base checkpoint")
    expected_detector_hashes = {
        "frozen": protocol["preservation"]["source_detector_sha256"],
        "adapted": training_provenance["best_checkpoint_sha256"],
    }
    verified = []
    for match_id, intervals in protocol["split"]["final_intervals_seconds_by_match"].items():
        source = source_by_id(protocol, match_id)
        for interval_index in range(len(intervals)):
            interval_name = f"final{interval_index}"
            start_s, end_s = intervals[interval_index]
            expected_samples = (
                end_s - start_s
            ) * (NATIVE_FPS // SAMPLE_STRIDE)
            expected_indices = [
                start_s * NATIVE_FPS + index * SAMPLE_STRIDE
                for index in range(expected_samples)
            ]
            for route in routes:
                path = prediction_path(
                    output, route, match_id, source["half"], interval_name,
                )
                metadata_path = path.with_suffix(".meta.json")
                if not path.is_file() or not metadata_path.is_file():
                    raise FileNotFoundError(
                        f"All final predictions must be frozen before scoring: {path}"
                    )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("route") != route
                    or metadata.get("match_id") != match_id
                    or metadata.get("interval") != interval_name
                    or metadata.get("sample_count") != expected_samples
                    or metadata.get("detector_sha256") != expected_detector_hashes[route]
                    or metadata.get("prediction_jsonl_sha256") != file_digest(path)
                    or metadata.get("source_video_sha256") != source["video_sha256"]
                ):
                    raise ValueError(f"Final prediction integrity check failed: {path}")
                with path.open(encoding="utf-8") as stream:
                    actual_indices = [
                        int(json.loads(line)["frame_index"])
                        for line in stream if line.strip()
                    ]
                if actual_indices != expected_indices:
                    raise ValueError(f"Final prediction frame sequence is not locked: {path}")
                verified.append({
                    "match_id": match_id,
                    "interval": interval_name,
                    "route": route,
                    "sha256": metadata["prediction_jsonl_sha256"],
                })
    return {
        "status": "ALL_FINAL_ROUTES_AND_INTERVALS_FROZEN",
        "verified_prediction_files": len(verified),
        "files": verified,
    }


def interval_by_name(protocol: dict, match_id: str, interval_name: str) -> tuple[int, int]:
    if interval_name == "development":
        if match_id != protocol["split"]["development_match"]:
            raise ValueError("The development interval is available only for D")
        return tuple(protocol["split"]["development_interval_seconds"])
    try:
        index = int(interval_name.removeprefix("final"))
    except ValueError as error:
        raise ValueError("Interval must be 'development' or 'final0', 'final1', 'final2'") from error
    intervals = protocol["split"]["final_intervals_seconds_by_match"][match_id]
    if not 0 <= index < len(intervals):
        raise ValueError(f"Unknown final interval: {interval_name}")
    return tuple(intervals[index])


def normalized_tolerance(width_px: int, reference_tolerance_px: int) -> float:
    return reference_tolerance_px * width_px / 4096.0


def _read_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
    previous_frame_index: int | None,
) -> np.ndarray:
    if previous_frame_index is None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    else:
        gap = frame_index - previous_frame_index
        if gap <= 0:
            raise ValueError("Requested frame indices must increase")
        for _ in range(gap - 1):
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame before {frame_index}")
    ok, image = capture.read()
    if not ok or image is None:
        raise RuntimeError(f"Could not decode source frame {frame_index}")
    return image


def prediction_path(output: Path, route: str, match_id: str, half: str, interval_name: str) -> Path:
    return output / "predictions" / route / f"{match_id}_{half}_{interval_name}.jsonl"


def _restore_tracker(records: list[dict]) -> tuple[dict[str, dict], int]:
    histories: dict[str, list[tuple[int, list[float]]]] = {}
    first_samples: dict[str, int] = {}
    observations: dict[str, int] = {}
    max_track_id = 0
    for sample_index, record in enumerate(records):
        for detection in record["detections"]:
            track_id = detection.get("track")
            point = detection.get("footpoint_xy_px")
            if not isinstance(track_id, str) or point is None:
                raise ValueError("Partial predictions lack tracker state")
            histories.setdefault(track_id, []).append(
                (sample_index, [float(point[0]), float(point[1])]),
            )
            first_samples.setdefault(track_id, sample_index)
            observations[track_id] = observations.get(track_id, 0) + 1
            try:
                max_track_id = max(max_track_id, int(track_id.rsplit("-", 1)[1]))
            except (ValueError, IndexError):
                raise ValueError(f"Unexpected tracker ID in partial predictions: {track_id}")
    active_tracks = {}
    last_sample = len(records) - 1
    for track_id, points in histories.items():
        sample, point = points[-1]
        if last_sample - sample > TRACK_MAX_GAP_SAMPLES:
            continue
        velocity = [0.0, 0.0]
        if len(points) > 1:
            previous_sample, previous = points[-2]
            elapsed = sample - previous_sample
            if elapsed > 0:
                velocity = [
                    float(np.clip((point[axis] - previous[axis]) / elapsed, -120.0, 120.0))
                    for axis in (0, 1)
                ]
        active_tracks[track_id] = {
            "footpoint_xy_px": point,
            "velocity_xy_px": velocity,
            "last_sample": sample,
            "observations": observations[track_id],
            "first_sample": first_samples[track_id],
            "last_detection_index": None,
        }
    return active_tracks, max_track_id + 1


def infer_segment(
    protocol: dict,
    output: Path,
    source: dict,
    interval_name: str,
    route: str,
    weights: Path,
    device: str,
    resume: bool = False,
) -> dict:
    from ultralytics import YOLO

    start_s, end_s = interval_by_name(protocol, source["id"], interval_name)
    duration_s = end_s - start_s
    if duration_s <= 0 or duration_s % 1:
        raise ValueError("Intervals must have a positive integer duration")
    video_path = ROOT / source["video"]
    weights = weights.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    weights_sha256 = file_digest(weights)
    destination = prediction_path(
        output, route, source["id"], source["half"], interval_name,
    )
    metadata_path = destination.with_suffix(".meta.json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable predictions: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists() and not resume:
        raise FileExistsError(f"Partial output exists; use --resume or preserve it: {partial}")

    expected_samples = duration_s * (NATIVE_FPS // SAMPLE_STRIDE)
    start_frame = start_s * NATIVE_FPS
    previous_frame: int | None = None
    completed = 0
    tracks: dict[str, dict] = {}
    next_track_id = 1
    existing_records: list[dict] = []
    if partial.exists():
        if not resume:
            raise FileExistsError(partial)
        with partial.open(encoding="utf-8") as existing:
            existing_records = [json.loads(line) for line in existing if line.strip()]
        if len(existing_records) > expected_samples:
            raise ValueError(f"Too many rows in partial output: {partial}")
        expected_indices = [
            start_frame + index * SAMPLE_STRIDE for index in range(len(existing_records))
        ]
        if [record.get("frame_index") for record in existing_records] != expected_indices:
            raise ValueError(f"Partial output does not match expected frame sequence: {partial}")
        if any(
            record.get("match_id") != source["id"]
            or record.get("interval") != interval_name
            or record.get("detector_sha256") != weights_sha256
            for record in existing_records
        ):
            raise ValueError("Partial predictions do not match source, interval, and model")
        completed = len(existing_records)
        if completed:
            tracks, next_track_id = _restore_tracker(existing_records)

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    if completed:
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame + completed * SAMPLE_STRIDE,
        )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if (width, height) != tuple(source["dimensions_px"]) or fps != float(source["fps"]):
        capture.release()
        raise ValueError(
            f"Video metadata changed for {source['id']}: "
            f"{width}x{height}@{fps}, expected {source['dimensions_px']}@{source['fps']}"
        )

    started = time.monotonic()
    with partial.open("a" if completed else "w", encoding="utf-8", newline="\n") as stream:
        try:
            previous_frame = None
            for sample_index in range(completed, expected_samples):
                frame_index = start_frame + sample_index * SAMPLE_STRIDE
                image = _read_frame(capture, frame_index, previous_frame)
                previous_frame = frame_index
                raw = detect_native(
                    model,
                    image,
                    confidence=DETECTION_CONFIDENCE,
                    device=device,
                )
                for detection in raw:
                    box = detection["bbox_xyxy_px"]
                    detection["footpoint_xy_px"] = [
                        (box[0] + box[2]) / 2.0, box[3],
                    ]
                next_track_id = _assign_tracklets(
                    raw,
                    tracks,
                    next_track_id,
                    sample_index,
                    max_distance_px=TRACK_MAX_DISTANCE_PX * width / 4096.0,
                )
                record = {
                    "schema_version": "football.video-player-cross-match-prediction.1",
                    "match_id": source["id"],
                    "half": source["half"],
                    "interval": interval_name,
                    "detector_sha256": weights_sha256,
                    "sample_index": sample_index,
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / NATIVE_FPS,
                    "dimensions_px": [width, height],
                    "detector_confidence_threshold": DETECTION_CONFIDENCE,
                    "detections": raw,
                }
                stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                completed = sample_index + 1
                if (sample_index + 1) % 50 == 0:
                    stream.flush()
        finally:
            capture.release()
    if completed != expected_samples:
        raise RuntimeError(f"Saved {completed}/{expected_samples} samples")
    digest = file_digest(partial)
    os.replace(partial, destination)
    metadata = {
        "schema_version": "football.video-player-cross-match-prediction-meta.1",
        "status": "FROZEN_VIDEO_ONLY_PREDICTIONS",
        "match_id": source["id"],
        "half": source["half"],
        "interval": interval_name,
        "interval_seconds": [start_s, end_s],
        "sample_count": completed,
        "sample_rate_hz": NATIVE_FPS / SAMPLE_STRIDE,
        "source_video_sha256": file_digest(video_path),
        "detector_sha256": weights_sha256,
        "prediction_jsonl_sha256": digest,
        "route": route,
        "device": device,
        "labels_read": False,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(metadata_path, metadata)
    return metadata


def _read_annotations(
    source: dict,
    frame_indices: set[int],
) -> dict[int, list[dict]]:
    annotation_path = ROOT / source["annotation"]
    if file_digest(annotation_path) != source["annotation_sha256"]:
        raise ValueError(f"Annotation source differs from the locked hash: {annotation_path}")
    image_zero = int(source["image_id_frame_zero"])
    labels: dict[int, list[dict]] = {}
    last_frame = -1
    max_frame = max(frame_indices)
    with annotation_path.open("rb") as stream:
        for item in ijson.items(stream, "annotations.item"):
            try:
                frame_index = int(item["image_id"]) - image_zero
            except (KeyError, TypeError, ValueError):
                continue
            if frame_index < last_frame:
                raise ValueError(f"Annotation frames are not ordered: {annotation_path}")
            last_frame = frame_index
            if frame_index > max_frame:
                break
            if frame_index not in frame_indices or item.get("category_id") not in (1, 2):
                continue
            box = item.get("bbox_image") or {}
            pitch = item.get("bbox_pitch") or {}
            attrs = item.get("attributes") or {}
            if not all(key in box for key in ("x", "y", "w", "h")):
                continue
            labels.setdefault(frame_index, []).append({
                "bbox_xywh_px": [
                    float(box["x"]), float(box["y"]),
                    float(box["w"]), float(box["h"]),
                ],
                "footpoint_xy_px": [
                    float(box["x"]) + float(box["w"]) / 2.0,
                    float(box["y"]) + float(box["h"]),
                ],
                "track_id": str(item.get("track_id", "")),
                "player_id": str(attrs.get("player_id", "")),
                "team": str(attrs.get("team", "UNKNOWN")),
                "role": str(attrs.get("role", "UNKNOWN")),
                "pitch_xy_centered_m": (
                    [
                        float(pitch["x_bottom_middle"]),
                        float(pitch["y_bottom_middle"]),
                    ]
                    if "x_bottom_middle" in pitch and "y_bottom_middle" in pitch
                    else None
                ),
                "category_id": int(item["category_id"]),
            })
    return labels


def _write_yolo_label_file(
    path: Path,
    tile_index: int,
    tile_left: int,
    tile_width: int,
    frame_width: int,
    frame_height: int,
    annotations: list[dict],
) -> int:
    core_width = frame_width / TILE_COLUMNS
    core_left = tile_index * core_width
    core_right = (tile_index + 1) * core_width
    rows = []
    for annotation in annotations:
        x, y, width, height = annotation["bbox_xywh_px"]
        center_x = x + width / 2.0
        if not (core_left <= center_x < core_right or (
            tile_index == TILE_COLUMNS - 1 and center_x == frame_width
        )):
            continue
        x1 = max(0.0, x - tile_left)
        x2 = min(float(tile_width), x + width - tile_left)
        y1 = max(0.0, y)
        y2 = min(float(frame_height), y + height)
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            f"0 {(x1 + x2) / (2 * tile_width):.8f} "
            f"{(y1 + y2) / (2 * frame_height):.8f} "
            f"{(x2 - x1) / tile_width:.8f} "
            f"{(y2 - y1) / frame_height:.8f}"
        )
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def _write_split_tiles(
    source: dict,
    interval: tuple[int, int],
    sample_stride: int,
    output: Path,
    split: str,
) -> dict:
    start_s, end_s = interval
    frame_indices = set(range(
        start_s * NATIVE_FPS,
        end_s * NATIVE_FPS,
        sample_stride,
    ))
    annotations = _read_annotations(source, frame_indices)
    if not annotations:
        raise ValueError(f"No fitting annotations in {source['id']} {split}")
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    video_path = ROOT / source["video"]
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    frames_written = 0
    tile_count = 0
    label_count = 0
    previous_frame = None
    try:
        for frame_index in sorted(frame_indices):
            image = _read_frame(capture, frame_index, previous_frame)
            previous_frame = frame_index
            height, width = image.shape[:2]
            if [width, height] != source["dimensions_px"]:
                raise ValueError(f"Unexpected image size in {source['id']} frame {frame_index}")
            tiles, offsets = _tiles(image)
            for tile_index, (tile, left) in enumerate(zip(tiles, offsets, strict=True)):
                stem = f"frame_{frame_index:09d}_tile_{tile_index}"
                image_path = image_dir / f"{stem}.jpg"
                label_path = label_dir / f"{stem}.txt"
                if not cv2.imwrite(str(image_path), tile):
                    raise RuntimeError(f"Could not write training tile: {image_path}")
                label_count += _write_yolo_label_file(
                    label_path,
                    tile_index,
                    left,
                    tile.shape[1],
                    width,
                    height,
                    annotations.get(frame_index, []),
                )
                tile_count += 1
            frames_written += 1
    finally:
        capture.release()
    return {
        "interval_seconds": [start_s, end_s],
        "sample_stride_frames": sample_stride,
        "source_frames_written": frames_written,
        "tile_images_written": tile_count,
        "footpoint_labels_written": label_count,
        "annotation_file_sha256": file_digest(ROOT / source["annotation"]),
        "video_file_sha256": file_digest(video_path),
    }


def prepare_fit_dataset(protocol: dict, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite transfer dataset: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = source_by_id(protocol, protocol["split"]["development_match"])
    train_interval = tuple(protocol["split"]["fitting_interval_seconds"])
    validation_interval = tuple(protocol["split"]["development_interval_seconds"])
    train = _write_split_tiles(
        source, train_interval, SAMPLE_STRIDE, output, "adaptation",
    )
    validation = _write_split_tiles(
        source, validation_interval, SAMPLE_STRIDE, output, "development",
    )

    source_root = ROOT / "outputs" / "overnight_video_player_20261028" / "training_data"
    source_images = sorted((source_root / "images" / "train").glob("*.jpg"))
    if not source_images:
        raise FileNotFoundError(f"Source-trained footpoint data is missing: {source_root}")
    adaptation_images = sorted((output / "images" / "adaptation").glob("*.jpg"))
    source_count = min(len(source_images), len(adaptation_images))
    stride = max(1, len(source_images) // source_count)
    source_subset = source_images[::stride][:source_count]
    train_list = output / "train_images.txt"
    train_list.write_text(
        "\n".join(
            str(path.resolve()).replace("\\", "/")
            for path in (*source_subset, *adaptation_images)
        ) + "\n",
        encoding="utf-8",
    )
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(
        f"train: {train_list.as_posix()}\n"
        f"val: {(output / 'images' / 'development').as_posix()}\n"
        "names:\n  0: player_footpoint\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "football.video-player-cross-match-training-data.1",
        "status": "D_FITTING_ONLY",
        "D_match": source["id"],
        "source_117093_tile_images_sampled": len(source_subset),
        "D_adaptation": train,
        "D_development": validation,
        "training_images_total": len(source_subset) + len(adaptation_images),
        "training_label_exposure": "Only source 117093 fitting labels plus 117092 [600,720] second; no D development or E annotations.",
        "validation_label_exposure": "117092 [900,1020] only; no E annotations.",
        "dataset_yaml": str(yaml_path.relative_to(ROOT)),
        "train_image_list_sha256": file_digest(train_list),
    }
    write_json(output / "dataset_provenance.json", report)
    return report


def train_adapted_detector(
    protocol: dict,
    dataset_yaml: Path,
    output: Path,
    requested_epochs: int,
    device: str,
    budget_minutes: int,
    initial_weights: Path | None = None,
) -> dict:
    from ultralytics import YOLO

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite adaptation output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if requested_epochs <= 0 or budget_minutes <= 0:
        raise ValueError("Epoch and time budgets must be positive")
    base_weights = (ROOT / protocol["preservation"]["source_detector"]).resolve()
    if file_digest(base_weights) != protocol["preservation"]["source_detector_sha256"]:
        raise ValueError("Frozen detector differs from the locked source checkpoint")
    weights = (initial_weights or base_weights).resolve()
    try:
        weights_relative = weights.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Initial weights must be stored inside the repository") from exc
    if not weights.is_file():
        raise FileNotFoundError(f"Initial detector checkpoint is missing: {weights}")
    model = YOLO(str(weights))
    deadline = datetime.fromisoformat(
        protocol["deadline"]["hard_deadline_utc"].replace("Z", "+00:00"),
    )
    final_reserve_start = deadline - timedelta(
        minutes=protocol["deadline"]["final_verification_reserved_minutes"],
    )
    budget_deadline = min(
        datetime.now(timezone.utc) + timedelta(minutes=budget_minutes),
        final_reserve_start,
    )
    stop_state = {"stopped_at_deadline": False}

    def check_deadline(trainer) -> None:
        if datetime.now(timezone.utc) >= budget_deadline:
            stop_state["stopped_at_deadline"] = True
            trainer.stop = True

    model.add_callback("on_train_batch_end", check_deadline)
    model.add_callback("on_train_epoch_end", check_deadline)
    started_at = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()
    model.train(
        data=str(dataset_yaml.resolve()),
        epochs=requested_epochs,
        imgsz=1280,
        batch=2,
        device=device,
        workers=0,
        project=str(output),
        name="detector_candidate",
        exist_ok=False,
        patience=4,
        seed=42,
        deterministic=True,
        cache=False,
        verbose=False,
    )
    best = output / "detector_candidate" / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("The bounded adaptation run did not produce best.pt")
    provenance = {
        "schema_version": "football.video-player-cross-match-training.1",
        "status": (
            "STOPPED_AT_BUDGET_DEADLINE"
            if stop_state["stopped_at_deadline"]
            else "TRAINED_SEPARATE_CANDIDATE"
        ),
        "base_checkpoint": str(base_weights.relative_to(ROOT)),
        "base_checkpoint_sha256": file_digest(base_weights),
        "starting_checkpoint": str(weights_relative),
        "starting_checkpoint_sha256": file_digest(weights),
        "best_checkpoint": str(best.relative_to(ROOT)),
        "best_checkpoint_sha256": file_digest(best),
        "dataset_yaml": str(dataset_yaml.resolve().relative_to(ROOT)),
        "dataset_provenance": json.loads(
            (dataset_yaml.parent / "dataset_provenance.json").read_text(encoding="utf-8"),
        ),
        "device": device,
        "requested_epochs": requested_epochs,
        "completed_epochs": max(
            0,
            sum(1 for _ in (
                output / "detector_candidate" / "results.csv"
            ).open(encoding="utf-8")) - 1,
        ) if (output / "detector_candidate" / "results.csv").is_file() else 0,
        "elapsed_seconds": time.monotonic() - start_monotonic,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "budget_deadline_utc": budget_deadline.isoformat(),
    }
    write_json(output / "training_provenance.json", provenance)
    return provenance


def _maximum_matches(
    predicted: list[dict],
    truth: list[dict],
    tolerance_px: float,
) -> list[tuple[int, int, float]]:
    pairs = []
    for pi, detection in enumerate(predicted):
        x, y = detection["footpoint_xy_px"]
        for gi, annotation in enumerate(truth):
            gx, gy = annotation["footpoint_xy_px"]
            distance = math.hypot(x - gx, y - gy)
            if distance <= tolerance_px:
                pairs.append((distance, pi, gi))
    by_prediction: dict[int, list[tuple[float, int]]] = {}
    for distance, pi, gi in pairs:
        by_prediction.setdefault(pi, []).append((distance, gi))
    for options in by_prediction.values():
        options.sort()
    assigned: dict[int, int] = {}

    def assign(pi: int, seen: set[int]) -> bool:
        for _, gi in by_prediction.get(pi, []):
            if gi in seen:
                continue
            seen.add(gi)
            if gi not in assigned or assign(assigned[gi], seen):
                assigned[gi] = pi
                return True
        return False

    for pi in sorted(by_prediction, key=lambda idx: by_prediction[idx][0][0]):
        assign(pi, set())
    return sorted(
        (
            pi,
            gi,
            next(d for d, candidate in by_prediction[pi] if candidate == gi),
        )
        for gi, pi in assigned.items()
    )


def score_segment(
    protocol: dict,
    output: Path,
    match_id: str,
    interval_name: str,
    route: str,
    tolerance_reference_px: int,
) -> dict:
    final_set_evidence = None
    if interval_name.startswith("final"):
        final_set_evidence = validate_final_prediction_set(protocol, output)
    source = source_by_id(protocol, match_id)
    start_s, end_s = interval_by_name(protocol, match_id, interval_name)
    pred_path = prediction_path(output, route, match_id, source["half"], interval_name)
    metadata_path = pred_path.with_suffix(".meta.json")
    if not pred_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Predictions and sidecar required before scoring: {pred_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if file_digest(pred_path) != metadata["prediction_jsonl_sha256"]:
        raise ValueError(f"Prediction file changed after it was frozen: {pred_path}")
    expected_route = metadata["route"]
    if expected_route != route:
        raise ValueError(f"Prediction route mismatch: {expected_route} != {route}")

    frames: list[dict] = []
    with pred_path.open(encoding="utf-8") as stream:
        frames = [json.loads(line) for line in stream if line.strip()]
    expected_samples = (end_s - start_s) * (NATIVE_FPS // SAMPLE_STRIDE)
    if len(frames) != expected_samples:
        raise ValueError(f"Expected {expected_samples} predictions, got {len(frames)}")
    expected_start = start_s * NATIVE_FPS
    expected_indices = [
        expected_start + i * SAMPLE_STRIDE for i in range(expected_samples)
    ]
    if [frame["frame_index"] for frame in frames] != expected_indices:
        raise ValueError("Prediction frame sequence does not match the locked protocol")

    labels = _read_annotations(source, set(expected_indices))
    matches_by_threshold: dict[int, dict[int, list[tuple[int, int, float]]]] = {
        tolerance: {} for tolerance in (20, 40)
    }
    all_detections = 0
    total_truth = 0
    for frame in frames:
        pred = frame["detections"]
        gt = labels.get(frame["frame_index"], [])
        for tolerance in matches_by_threshold:
            matches_by_threshold[tolerance][frame["frame_index"]] = _maximum_matches(
                pred,
                gt,
                normalized_tolerance(source["dimensions_px"][0], tolerance),
            )
        all_detections += len(pred)
        total_truth += len(gt)
    threshold_metrics = {}
    for tolerance, per_frame in matches_by_threshold.items():
        match_count = sum(map(len, per_frame.values()))
        unmatched_detections = all_detections - match_count
        unmatched_annotations = total_truth - match_count
        duplicate_candidates = 0
        for frame in frames:
            predictions = frame["detections"]
            truth = labels.get(frame["frame_index"], [])
            matched_detection_indices = {
                prediction_index
                for prediction_index, _, _ in per_frame[frame["frame_index"]]
            }
            matched_truth_indices = {
                truth_index for _, truth_index, _ in per_frame[frame["frame_index"]]
            }
            for prediction_index, detection in enumerate(predictions):
                if prediction_index in matched_detection_indices:
                    continue
                x, y = detection["footpoint_xy_px"]
                if any(
                    math.hypot(
                        x - annotation["footpoint_xy_px"][0],
                        y - annotation["footpoint_xy_px"][1],
                    ) <= normalized_tolerance(
                        source["dimensions_px"][0], tolerance,
                    )
                    for index, annotation in enumerate(truth)
                    if index in matched_truth_indices
                ):
                    duplicate_candidates += 1
        threshold_metrics[str(tolerance)] = {
            "footpoint_tolerance_reference_px": tolerance,
            "footpoint_tolerance_px": normalized_tolerance(
                source["dimensions_px"][0], tolerance,
            ),
            "matched_count": match_count,
            "precision": match_count / all_detections if all_detections else 0.0,
            "recall": match_count / total_truth if total_truth else 0.0,
            "unmatched_detections": unmatched_detections,
            "unmatched_annotations": unmatched_annotations,
            "unmatched_detections_near_already_matched_annotation": duplicate_candidates,
            "unmatched_detection_false_positives": unmatched_detections - duplicate_candidates,
            "matches_by_frame": {
                str(frame_index): [
                    {"detection_index": pi, "annotation_index": gi, "distance_px": dist}
                    for pi, gi, dist in pairs
                ]
                for frame_index, pairs in per_frame.items()
            },
        }

    projection_metrics = None
    projector_path = output / "calibration" / f"local_projector_{match_id}.json"
    if projector_path.is_file():
        projector = load_local_projector(protocol, output, match_id)
        projection_status_counts: dict[str, int] = {}
        projected_per_frame = []
        supported_annotation_count = 0
        projected_matched_count = 0
        projected_matched_within_support = 0
        position_errors = []
        position_residuals = []
        for frame in frames:
            frame_index = int(frame["frame_index"])
            projected: dict[int, list[float]] = {}
            for detection_index, detection in enumerate(frame["detections"]):
                point, status = project_footpoint(
                    detection["footpoint_xy_px"], projector,
                )
                projection_status_counts[status] = (
                    projection_status_counts.get(status, 0) + 1
                )
                if point is not None:
                    projected[detection_index] = point
            projected_per_frame.append(len(projected))
            truth = labels.get(frame_index, [])
            supported_annotation_indices = set()
            for annotation_index, annotation in enumerate(truth):
                _, support_status = project_footpoint(
                    annotation["footpoint_xy_px"], projector,
                )
                if support_status == "UNVALIDATED_EXPERIMENTAL_LOCAL_PROJECTION":
                    supported_annotation_indices.add(annotation_index)
            supported_annotation_count += len(supported_annotation_indices)
            for detection_index, annotation_index, _ in matches_by_threshold[40][frame_index]:
                if detection_index not in projected:
                    continue
                projected_matched_count += 1
                if annotation_index in supported_annotation_indices:
                    projected_matched_within_support += 1
                else:
                    continue
                annotation = truth[annotation_index]
                centered = annotation.get("pitch_xy_centered_m")
                if centered is None:
                    continue
                target_top_left = np.asarray(centered, dtype=np.float64) + [52.5, 34.0]
                residual = np.asarray(projected[detection_index]) - target_top_left
                position_residuals.append(residual.tolist())
                position_errors.append(float(np.linalg.norm(residual)))
        projection_metrics = {
            "status": projector["status"],
            "validation_gate_pass": projector["validation"]["acceptance_gate_pass"],
            "fixed_roi": projector["fixed_roi"],
            "support_denominators": {
                "all_annotated_player_instances": total_truth,
                "annotated_player_instances_inside_fixed_setup_support": supported_annotation_count,
                "video_detections": all_detections,
                "detections_projected_inside_support": sum(projected_per_frame),
                "pixel_matched_detections_also_projected": projected_matched_count,
                "pixel_matched_projected_within_support": projected_matched_within_support,
            },
            "within_support_recall": (
                projected_matched_within_support / supported_annotation_count
                if supported_annotation_count else None
            ),
            "coverage_relative_to_all_annotations": (
                projected_matched_count / total_truth if total_truth else 0.0
            ),
            "projection_status_counts": projection_status_counts,
            "projected_players_per_frame": {
                "minimum": min(projected_per_frame) if projected_per_frame else 0,
                "median": float(np.median(projected_per_frame)) if projected_per_frame else 0.0,
                "p90": float(np.percentile(projected_per_frame, 90)) if projected_per_frame else 0.0,
                "maximum": max(projected_per_frame) if projected_per_frame else 0,
            },
            "position_error": {
                "count": len(position_errors),
                "comparison_scope": "pixel-matched detections whose reference footpoints are inside the fixed ROI and fitting-landmark hull",
                "median_m": float(np.median(position_errors)) if position_errors else None,
                "p90_m": float(np.percentile(position_errors, 90)) if position_errors else None,
                "p95_m": float(np.percentile(position_errors, 95)) if position_errors else None,
                "maximum_m": max(position_errors) if position_errors else None,
                "mean_residual_vector_m": (
                    np.mean(position_residuals, axis=0).tolist()
                    if position_residuals else None
                ),
                "status": "UNVALIDATED_EXPERIMENTAL_DIAGNOSTIC",
            },
        }
    track_lengths: dict[str, int] = {}
    track_identity_counts: dict[str, dict[str, int]] = {}
    identity_track_counts: dict[str, set[str]] = {}
    identity_samples: dict[str, list[tuple[int, str]]] = {}
    frame_index_by_number = {
        int(frame["frame_index"]): sample_index
        for sample_index, frame in enumerate(frames)
    }
    for frame in frames:
        for detection in frame["detections"]:
            track = str(detection.get("track", ""))
            if track:
                track_lengths[track] = track_lengths.get(track, 0) + 1
    for frame_index, pairs in matches_by_threshold[40].items():
        frame = frames[frame_index_by_number[frame_index]]
        truth = labels.get(frame_index, [])
        for prediction_index, truth_index, _ in pairs:
            detection = frame["detections"][prediction_index]
            annotation = truth[truth_index]
            track = str(detection.get("track", ""))
            identity = str(annotation.get("player_id") or annotation["track_id"])
            if not track:
                continue
            counts = track_identity_counts.setdefault(track, {})
            counts[identity] = counts.get(identity, 0) + 1
            identity_track_counts.setdefault(identity, set()).add(track)
            identity_samples.setdefault(identity, []).append(
                (frame_index_by_number[frame_index], track),
            )
    purities = [
        max(counts.values()) / sum(counts.values())
        for counts in track_identity_counts.values()
    ]
    adjacent_identity_switches = 0
    for observations in identity_samples.values():
        ordered = sorted(observations)
        adjacent_identity_switches += sum(
            previous_sample + 1 == current_sample and previous_track != current_track
            for (previous_sample, previous_track), (current_sample, current_track)
            in zip(ordered, ordered[1:])
        )
    sorted_lengths = sorted(track_lengths.values())

    def quantile(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(values, percentile)) if values else None

    tracking_metrics = {
        "runtime_track_count": len(track_lengths),
        "matched_runtime_track_count": len(track_identity_counts),
        "track_length_samples": {
            "minimum": min(sorted_lengths) if sorted_lengths else None,
            "median": quantile(sorted_lengths, 50),
            "p90": quantile(sorted_lengths, 90),
            "maximum": max(sorted_lengths) if sorted_lengths else None,
        },
        "track_purity": {
            "sample_count": len(purities),
            "p10": quantile(purities, 10),
            "median": quantile(purities, 50),
            "p90": quantile(purities, 90),
            "tracks_below_0_5": sum(value < 0.5 for value in purities),
        },
        "identity_fragmentation": {
            "matched_identity_count": len(identity_track_counts),
            "runtime_tracks_per_identity_median": quantile(
                [len(value) for value in identity_track_counts.values()], 50,
            ),
            "runtime_tracks_per_identity_p90": quantile(
                [len(value) for value in identity_track_counts.values()], 90,
            ),
            "identities_split_across_multiple_runtime_tracks": sum(
                len(value) > 1 for value in identity_track_counts.values()
            ),
        },
        "adjacent_matched_identity_switches": adjacent_identity_switches,
        "identity_sequence_support": sum(map(len, identity_samples.values())),
    }
    selected_metrics = threshold_metrics[str(tolerance_reference_px)]
    result = {
        "schema_version": "football.video-player-cross-match-evaluation.1",
        "status": "ANNOTATIONS_OPENED_AFTER_PREDICTION_HASH_VERIFICATION",
        "match_id": match_id,
        "half": source["half"],
        "interval": interval_name,
        "interval_seconds": [start_s, end_s],
        "prediction_route": route,
        "prediction_sha256": metadata["prediction_jsonl_sha256"],
        "annotation_sha256": file_digest(ROOT / source["annotation"]),
        "dimensions_px": source["dimensions_px"],
        "sampling_hz": NATIVE_FPS / SAMPLE_STRIDE,
        "samples": len(frames),
        "annotation_player_instances": total_truth,
        "detection_count": all_detections,
        **{
            key: value for key, value in selected_metrics.items()
            if key != "matches_by_frame"
        },
        "threshold_results": threshold_metrics,
        "tracking": tracking_metrics,
        "projection": projection_metrics,
        "all_final_predictions_frozen_before_scoring": final_set_evidence,
        "matches_by_frame": selected_metrics["matches_by_frame"],
        "limitations": [
            "Pixel footpoint metrics only; no pitch projection was accepted by this scoring stage.",
            "Annotation player positions, identities, teams, roles, and actions are evaluation-only.",
        ],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--match", required=True)
    infer.add_argument("--interval", required=True)
    infer.add_argument("--route", default="frozen")
    infer.add_argument("--weights", type=Path)
    infer.add_argument("--device", choices=["auto", "cpu", "0"], default="auto")
    infer.add_argument("--resume", action="store_true")

    prepare = subparsers.add_parser("prepare-fit")
    prepare.add_argument("--dataset-output", type=Path, required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--match", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--dataset-yaml", type=Path, required=True)
    train.add_argument("--training-output", type=Path, required=True)
    train.add_argument("--initial-weights", type=Path)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--device", choices=["cpu", "0"], default="cpu")
    train.add_argument("--budget-minutes", type=int, default=90)

    score = subparsers.add_parser("score")
    score.add_argument("--match", required=True)
    score.add_argument("--interval", required=True)
    score.add_argument("--route", default="frozen")
    score.add_argument("--tolerance-reference-px", type=int, choices=[20, 40], default=40)

    subparsers.add_parser("validate-final")

    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    output = args.output.resolve()
    if args.command == "prepare-fit":
        print(json.dumps(
            prepare_fit_dataset(protocol, args.dataset_output.resolve()),
            indent=2,
        ))
        return 0
    if args.command == "calibrate":
        fit_local_projector(protocol, output, args.match)
        return 0
    if args.command == "train":
        print(json.dumps(
            train_adapted_detector(
                protocol,
                args.dataset_yaml.resolve(),
                args.training_output.resolve(),
                args.epochs,
                args.device,
                args.budget_minutes,
                args.initial_weights.resolve() if args.initial_weights else None,
            ),
            indent=2,
        ))
        return 0
    if args.command == "validate-final":
        print(json.dumps(validate_final_prediction_set(protocol, output), indent=2))
        return 0
    if args.command == "infer":
        source = source_by_id(protocol, args.match)
        default_weights = ROOT / protocol["preservation"]["source_detector"]
        device = (
            "0" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto"
            else args.device
        )
        if device == "0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device 0 was requested but is not available")
        metadata = infer_segment(
            protocol,
            output,
            source,
            args.interval,
            args.route,
            args.weights or default_weights,
            device,
            resume=args.resume,
        )
        print(json.dumps(metadata, indent=2))
        return 0
    result = score_segment(
        protocol,
        output,
        args.match,
        args.interval,
        args.route,
        args.tolerance_reference_px,
    )
    score_path = output / "scores" / args.route / (
        f"{args.match}_{result['half']}_{args.interval}_"
        f"{args.tolerance_reference_px}px.json"
    )
    if score_path.exists():
        raise FileExistsError(f"Refusing to overwrite score: {score_path}")
    score_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(score_path, result)
    summary = {
        key: value for key, value in result.items()
        if key not in {"matches_by_frame", "threshold_results"}
    }
    summary["threshold_results"] = {
        tolerance: {
            key: value for key, value in metrics.items()
            if key != "matches_by_frame"
        }
        for tolerance, metrics in result["threshold_results"].items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
