"""Evaluate the experimental detector, estimate player attributes, and replay video inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import ijson
import numpy as np

from src.backbone.actions import ActionModelSession
from src.backbone.external import ExternalObservationSource, VERSION as EXTERNAL_SCHEMA_VERSION
from src.backbone.models import ModelSession
from src.backbone.observation import SyntheticObservationProvider
from src.backbone.pipeline import ReplayEngine
from src.backbone.video_pitch_experimental import (
    LOCAL_MATCH,
    LOCAL_HALF,
    _inside_hull,
    _landmark_data,
)
from src.backbone.video_player_attributes import (
    K_NEIGHBORS,
    ROLE_CLASSES,
    TEAM_CLASSES,
    _attribute_metrics,
    _knn_scores,
    _maximum_footpoint_matches,
    appearance_feature,
)
from src.backbone.video_player_overnight import (
    ANNOTATION_PATH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    NATIVE_FPS,
    OUTPUT_ROOT,
    SAMPLE_STRIDE,
    TILE_COLUMNS,
    TILE_OVERLAP_PX,
    VIDEO_PATH,
    detect_native,
)
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
BEST_DETECTOR = OUTPUT_ROOT / "detector_training" / "yolo11n_footpoint" / "weights" / "best.pt"
ATTRIBUTE_MODEL = OUTPUT_ROOT / "attribute_model" / "appearance_knn.npz"
ATTRIBUTE_PROVENANCE = OUTPUT_ROOT / "attribute_model" / "appearance_classifier_provenance.json"
BASE_DETECTOR = ROOT / "yolo11n.pt"
BASELINE_CONFIG = ROOT / "configs" / "backbone_release_baseline.json"
FFMPEG = (
    ROOT / "data" / "video_review_tools" / "portable"
    / "ffmpeg-9.0.2-essentials_build" / "bin" / "ffmpeg.exe"
)
EVALUATION_SEGMENTS = ((1740, 1800), (1800, 1860))
DETECTION_CONFIDENCE = 0.20
FOOTPOINT_MATCH_TOLERANCE_PX = 40.0
TRACK_MAX_DISTANCE_PX = 180.0
TRACK_MAX_GAP_SAMPLES = 5
PROJECTOR_VERSION = 2


def _read_ground_truth(
    start_frame: int, end_frame: int,
) -> dict[int, list[dict]]:
    """Load verified labels only for the separate post-inference evaluation pass."""
    base = 3_903_000_001
    labels: dict[int, list[dict]] = {}
    with ANNOTATION_PATH.open("rb") as stream:
        for item in ijson.items(stream, "annotations.item"):
            try:
                frame_index = int(item["image_id"]) - base
            except (KeyError, TypeError, ValueError):
                continue
            if frame_index >= end_frame:
                break
            if frame_index < start_frame or frame_index % SAMPLE_STRIDE:
                continue
            if item.get("category_id") not in (1, 2) or item.get("track_id") is None:
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
                "track_id": str(item["track_id"]),
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
            })
    return labels


def _projector() -> tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    image, pitch, _, _, transform = _landmark_data(LOCAL_MATCH)
    principal_x = float(transform["camera"][0, 2])
    in_roi = np.flatnonzero(np.abs(image[:, 0] - principal_x) <= 650.0)
    order = in_roi[np.argsort(pitch[in_roi, 0], kind="stable")]
    validation = order[::4]
    fitting = np.setdiff1d(order, validation)
    pitch_to_rectified, _ = cv2.findHomography(
        pitch[fitting], transform["rectified"][fitting], 0,
    )
    if pitch_to_rectified is None:
        raise RuntimeError("The fixed regional camera mapping did not fit")
    inverse = np.linalg.inv(pitch_to_rectified)
    hull = cv2.convexHull(transform["rectified"][fitting].astype(np.float32))
    return (
        principal_x, transform["camera"], transform["distortion"],
        transform["rectified_camera"], inverse, hull,
    )


def _project_box(
    box: list[float],
    projector: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[float] | None, str]:
    principal_x, camera, distortion, rectified_camera, inverse, hull = projector
    x1, _, x2, y2 = box
    foot = np.asarray([[(x1 + x2) / 2.0, y2]], dtype=np.float64)
    if abs(float(foot[0, 0]) - principal_x) > 650.0:
        return None, "OUTSIDE_OPTICAL_ROI"
    rectified = cv2.fisheye.undistortPoints(
        foot[:, None], camera, distortion, P=rectified_camera,
    )[:, 0]
    if not bool(_inside_hull(rectified, hull)[0]):
        return None, "OUTSIDE_FIT_HULL"
    projected = cv2.perspectiveTransform(rectified[:, None], inverse)[0, 0]
    if (not np.isfinite(projected).all()
            or not (0 <= projected[0] <= 105 and 0 <= projected[1] <= 68)):
        return None, "OUTSIDE_PITCH_BOUNDS"
    return projected.astype(float).tolist(), "EXPERIMENTAL_SUPPORTED_HULL"


def _load_appearance_model() -> tuple[np.ndarray, list[str], list[str], dict]:
    if not ATTRIBUTE_MODEL.is_file() or not ATTRIBUTE_PROVENANCE.is_file():
        raise FileNotFoundError("Run video_player_attributes before the replay")
    archive = np.load(ATTRIBUTE_MODEL, allow_pickle=False)
    metadata = json.loads(ATTRIBUTE_PROVENANCE.read_text(encoding="utf-8"))
    return (
        archive["features"].astype(np.float32),
        archive["teams"].astype(str).tolist(),
        archive["roles"].astype(str).tolist(),
        metadata,
    )


def _assign_tracklets(
    detections: list[dict],
    tracks: dict[str, dict],
    next_id: int,
    sample_index: int,
    max_distance_px: float = TRACK_MAX_DISTANCE_PX,
) -> int:
    candidates = []
    for track_id, track in tracks.items():
        age = sample_index - track["last_sample"]
        if age < 1 or age > TRACK_MAX_GAP_SAMPLES:
            continue
        x, y = track["footpoint_xy_px"]
        vx, vy = track["velocity_xy_px"]
        predicted = (x + vx * age, y + vy * age)
        for detection_index, detection in enumerate(detections):
            point = detection["footpoint_xy_px"]
            distance = math.hypot(point[0] - predicted[0], point[1] - predicted[1])
            if distance <= max_distance_px * min(age, 2):
                candidates.append((distance, track_id, detection_index))
    assigned_tracks: set[str] = set()
    assigned_detections: set[int] = set()
    for _, track_id, detection_index in sorted(candidates):
        if track_id in assigned_tracks or detection_index in assigned_detections:
            continue
        track = tracks[track_id]
        point = detections[detection_index]["footpoint_xy_px"]
        dt = sample_index - track["last_sample"]
        old = track["footpoint_xy_px"]
        track["velocity_xy_px"] = [
            float(np.clip((point[0] - old[0]) / dt, -120.0, 120.0)),
            float(np.clip((point[1] - old[1]) / dt, -120.0, 120.0)),
        ]
        track["footpoint_xy_px"] = point
        track["last_sample"] = sample_index
        track["observations"] += 1
        detections[detection_index]["track"] = track_id
        assigned_tracks.add(track_id)
        assigned_detections.add(detection_index)
    for detection_index, detection in enumerate(detections):
        if detection_index in assigned_detections:
            continue
        track_id = f"video-player-{next_id:04d}"
        next_id += 1
        point = detection["footpoint_xy_px"]
        tracks[track_id] = {
            "footpoint_xy_px": point,
            "velocity_xy_px": [0.0, 0.0],
            "last_sample": sample_index,
            "observations": 1,
            "first_sample": sample_index,
            "last_detection_index": None,
        }
        detection["track"] = track_id
    for track_id in list(tracks):
        if sample_index - tracks[track_id]["last_sample"] > TRACK_MAX_GAP_SAMPLES:
            del tracks[track_id]
    return next_id


def _supported_detections(
    raw: list[dict],
    appearance_boxes: list[dict],
    image: np.ndarray,
    projector,
    train_features: np.ndarray,
    train_teams: list[str],
    train_roles: list[str],
    attr_metadata: dict,
) -> list[dict]:
    detections = []
    for item in raw:
        if item["detector_confidence"] < DETECTION_CONFIDENCE:
            continue
        box = item["bbox_xyxy_px"]
        pitch, projection_status = _project_box(box, projector)
        if pitch is None:
            continue
        detections.append({
            **item,
            "footpoint_xy_px": [(box[0] + box[2]) / 2.0, box[3]],
            "pitch_xy_m": pitch,
            "projection_status": projection_status,
            "team": 0,
            "team_prediction": "UNKNOWN",
            "team_confidence": 0.0,
            "goalkeeper": None,
            "role_prediction": "UNKNOWN",
            "role_confidence": 0.0,
            "appearance_box_xyxy_px": None,
        })
    candidate_for_appearance = [
        {"bbox_xyxy_px": detection["bbox_xyxy_px"]}
        for detection in detections
    ]
    baseline_for_appearance = [
        {
            "footpoint_xy_px": [
                (box["bbox_xyxy_px"][0] + box["bbox_xyxy_px"][2]) / 2.0,
                box["bbox_xyxy_px"][3],
            ],
            "box": box,
        }
        for box in appearance_boxes
    ]
    appearance_matches = _maximum_footpoint_matches(
        candidate_for_appearance, baseline_for_appearance, 80.0,
    )
    team_threshold = attr_metadata["team_confidence_threshold"]
    role_threshold = attr_metadata["role_confidence_threshold"]
    for candidate_index, appearance_index, _ in appearance_matches:
        detection = detections[candidate_index]
        appearance_box = baseline_for_appearance[appearance_index]["box"]
        feature = appearance_feature(image, appearance_box["bbox_xyxy_px"])[None]
        team_scores = _knn_scores(
            train_features, train_teams, feature, TEAM_CLASSES,
        )[0]
        role_scores = _knn_scores(
            train_features, train_roles, feature, ROLE_CLASSES,
        )[0]
        team_index = int(team_scores.argmax())
        role_index = int(role_scores.argmax())
        team_confidence = float(team_scores[team_index])
        role_confidence = float(role_scores[role_index])
        detection["team"] = (
            team_index + 1
            if team_threshold is not None and team_confidence >= team_threshold
            else 0
        )
        detection["team_prediction"] = TEAM_CLASSES[team_index]
        detection["team_confidence"] = team_confidence
        detection["goalkeeper"] = (
            role_index == 1
            if role_threshold is not None and role_confidence >= role_threshold
            else None
        )
        detection["role_prediction"] = ROLE_CLASSES[role_index]
        detection["role_confidence"] = role_confidence
        detection["appearance_box_xyxy_px"] = appearance_box["bbox_xyxy_px"]
    return detections


def _infer_segment(
    start_s: int,
    duration_s: int,
    detector,
    appearance_detector,
    train_features: np.ndarray,
    train_teams: list[str],
    train_roles: list[str],
    attr_metadata: dict,
) -> list[dict]:
    start_frame = start_s * NATIVE_FPS
    count = duration_s * (NATIVE_FPS // SAMPLE_STRIDE)
    capture = cv2.VideoCapture(str(VIDEO_PATH), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {VIDEO_PATH}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    projector = _projector()
    tracks: dict[str, dict] = {}
    next_track_id = 1
    frames = []
    try:
        for sample_index in range(count):
            frame_index = start_frame + sample_index * SAMPLE_STRIDE
            for _ in range(SAMPLE_STRIDE):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode source frame {frame_index}")
            if image.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
                raise ValueError(f"Unexpected source dimensions at frame {frame_index}")
            raw = detect_native(detector, image, confidence=DETECTION_CONFIDENCE)
            raw_detection_count = len(raw)
            appearance_boxes = detect_native(
                appearance_detector, image, confidence=DETECTION_CONFIDENCE,
            )
            detections = _supported_detections(
                raw, appearance_boxes, image, projector, train_features,
                train_teams, train_roles, attr_metadata,
            )
            next_track_id = _assign_tracklets(
                detections, tracks, next_track_id, sample_index,
            )
            frames.append({
                "record_type": "frame",
                "projector_version": PROJECTOR_VERSION,
                "frame_index": frame_index,
                "timestamp_s": frame_index / NATIVE_FPS,
                "segment_id": f"{LOCAL_MATCH}_{LOCAL_HALF}_{start_s}",
                "raw_detection_count": raw_detection_count,
                "candidate_raw_detections": raw,
                "baseline_person_detections": appearance_boxes,
                "detections": detections,
            })
    finally:
        capture.release()
    return frames


def _read_saved_detections(path: Path, start_s: int, duration_s: int) -> list[dict]:
    expected_count = duration_s * (NATIVE_FPS // SAMPLE_STRIDE)
    frames = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}, line {line_number}: invalid JSON") from error
            if not isinstance(frame, dict):
                raise ValueError(f"{path.name}, line {line_number}: expected an object")
            frames.append(frame)
    expected_start = start_s * NATIVE_FPS
    expected_indices = [
        expected_start + index * SAMPLE_STRIDE for index in range(expected_count)
    ]
    if len(frames) != expected_count or [
        frame.get("frame_index") for frame in frames
    ] != expected_indices:
        raise ValueError(
            f"Saved detections do not contain the exact {expected_count}-sample interval "
            f"{start_s}-{start_s + duration_s}s"
        )
    required = {
        "raw_detection_count", "candidate_raw_detections",
        "baseline_person_detections", "detections",
    }
    if any(not required.issubset(frame) for frame in frames):
        raise ValueError(f"Saved detections lack required replay fields: {path}")
    return frames


def _rebuild_saved_projection(
    frames: list[dict],
    start_s: int,
    train_features: np.ndarray,
    train_teams: list[str],
    train_roles: list[str],
    attr_metadata: dict,
) -> list[dict]:
    """Re-project complete saved detector outputs without rerunning either detector."""
    capture = cv2.VideoCapture(str(VIDEO_PATH), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen source video for saved-output repair: {VIDEO_PATH}")
    start_frame = start_s * NATIVE_FPS
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    projector = _projector()
    tracks: dict[str, dict] = {}
    next_track_id = 1
    try:
        for sample_index, frame in enumerate(frames):
            image = None
            for _ in range(SAMPLE_STRIDE):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Could not decode source frame {frame['frame_index']} during projection repair"
                    )
            frame["detections"] = _supported_detections(
                frame["candidate_raw_detections"],
                frame["baseline_person_detections"],
                image,
                projector,
                train_features,
                train_teams,
                train_roles,
                attr_metadata,
            )
            next_track_id = _assign_tracklets(
                frame["detections"], tracks, next_track_id, sample_index,
            )
            frame["projector_version"] = PROJECTOR_VERSION
    finally:
        capture.release()
    return frames


def _paired_evaluation(
    frames: list[dict], ground_truth: dict[int, list[dict]],
) -> dict:
    total_candidate_detections = total_baseline_detections = total_gt = 0
    candidate_matches = baseline_matches = supported_matches = 0
    in_support = 0
    position_errors = []
    team_true, team_scores, role_true, role_scores = [], [], [], []
    confidence_team_true, confidence_team_scores = [], []
    confidence_role_true, confidence_role_scores = [], []
    track_player: dict[str, dict[str, int]] = {}
    matched_per_frame = []
    for frame in frames:
        detections = frame["detections"]
        candidate_raw = frame["candidate_raw_detections"]
        baseline_raw = frame["baseline_person_detections"]
        labels = ground_truth.get(frame["frame_index"], [])
        total_candidate_detections += len(candidate_raw)
        total_baseline_detections += len(baseline_raw)
        total_gt += len(labels)
        candidate_frame_matches = _maximum_footpoint_matches(
            candidate_raw, labels, FOOTPOINT_MATCH_TOLERANCE_PX,
        )
        baseline_frame_matches = _maximum_footpoint_matches(
            baseline_raw, labels, FOOTPOINT_MATCH_TOLERANCE_PX,
        )
        matches = _maximum_footpoint_matches(
            [
                {"bbox_xyxy_px": detection["bbox_xyxy_px"]}
                for detection in detections
            ],
            labels,
            FOOTPOINT_MATCH_TOLERANCE_PX,
        )
        candidate_matches += len(candidate_frame_matches)
        baseline_matches += len(baseline_frame_matches)
        supported_matches += len(matches)
        matched_per_frame.append(len(matches))
        in_support += len(detections)
        for detection_index, annotation_index, _ in matches:
            detection = detections[detection_index]
            annotation = labels[annotation_index]
            if annotation["pitch_xy_centered_m"] is not None:
                truth = np.asarray(annotation["pitch_xy_centered_m"]) + [52.5, 34.0]
                position_errors.append(float(np.linalg.norm(
                    np.asarray(detection["pitch_xy_m"]) - truth,
                )))
            if annotation["team"] in TEAM_CLASSES:
                team_true.append(annotation["team"])
                team_scores.append([
                    detection["team_confidence"] if detection["team_prediction"] == "left"
                    else 1.0 - detection["team_confidence"],
                    detection["team_confidence"] if detection["team_prediction"] == "right"
                    else 1.0 - detection["team_confidence"],
                ])
                confidence_team_true.append(annotation["team"])
                confidence_team_scores.append(detection["team_confidence"])
            if annotation["role"] in ROLE_CLASSES:
                role_true.append(annotation["role"])
                role_scores.append([
                    detection["role_confidence"] if detection["role_prediction"] == "player"
                    else 1.0 - detection["role_confidence"],
                    detection["role_confidence"] if detection["role_prediction"] == "goalkeeper"
                    else 1.0 - detection["role_confidence"],
                ])
                confidence_role_true.append(annotation["role"])
                confidence_role_scores.append(detection["role_confidence"])
            identity = annotation["player_id"] or annotation["track_id"]
            track_player.setdefault(detection["track"], {})
            counts = track_player[detection["track"]]
            counts[identity] = counts.get(identity, 0) + 1
    team_prediction = _attribute_metrics(
        team_true, np.asarray(team_scores, dtype=np.float32), TEAM_CLASSES,
    ) if team_true else None
    role_prediction = _attribute_metrics(
        role_true, np.asarray(role_scores, dtype=np.float32), ROLE_CLASSES,
    ) if role_true else None
    purity = [
        max(counts.values()) / sum(counts.values())
        for counts in track_player.values()
    ]
    return {
        "denominators": {
            "continuous_samples": len(frames),
            "ground_truth_player_annotations": total_gt,
            "candidate_detector_boxes": total_candidate_detections,
            "baseline_person_boxes": total_baseline_detections,
            "supported_video_detections": in_support,
            "supported_detections_matched": supported_matches,
            "position_error_pairs": len(position_errors),
            "team_attribute_pairs": len(team_true),
            "goalkeeper_role_pairs": len(role_true),
        },
        "detection_footpoint_match_tolerance_px": FOOTPOINT_MATCH_TOLERANCE_PX,
        "detection": {
            "candidate_finetuned": {
                "precision": candidate_matches / max(total_candidate_detections, 1),
                "recall": candidate_matches / max(total_gt, 1),
                "detections": total_candidate_detections,
                "matches": candidate_matches,
                "ground_truth": total_gt,
            },
            "baseline_coco": {
                "precision": baseline_matches / max(total_baseline_detections, 1),
                "recall": baseline_matches / max(total_gt, 1),
                "detections": total_baseline_detections,
                "matches": baseline_matches,
                "ground_truth": total_gt,
            },
            "supported_finetuned": {
                "precision": supported_matches / max(in_support, 1),
                "recall": supported_matches / max(total_gt, 1),
                "detections": in_support,
                "matches": supported_matches,
                "ground_truth": total_gt,
            },
        },
        "projection_support_fraction_of_ground_truth": in_support / max(total_gt, 1),
        "position_error_m": {
            "median": float(np.median(position_errors)) if position_errors else None,
            "p90": float(np.percentile(position_errors, 90)) if position_errors else None,
            "mean": float(np.mean(position_errors)) if position_errors else None,
            "maximum": float(np.max(position_errors)) if position_errors else None,
        },
        "predicted_team": team_prediction,
        "predicted_goalkeeper_role": role_prediction,
        "tracking": {
            "matched_tracks": len(track_player),
            "median_identity_purity": float(np.median(purity)) if purity else None,
            "p10_identity_purity": float(np.percentile(purity, 10)) if purity else None,
            "tracklets_with_at_least_11_matched_samples": sum(
                sum(counts.values()) >= 11 for counts in track_player.values()
            ),
            "matched_player_observations_per_sample_p50_p90": [
                float(np.percentile(matched_per_frame, 50)),
                float(np.percentile(matched_per_frame, 90)),
            ] if matched_per_frame else [0.0, 0.0],
        },
        "status": "EXPERIMENTAL_HELDOUT_EVALUATION; no benchmark/training admission",
    }


def _build_external_rows(
    frames: list[dict],
    ground_truth: dict[int, list[dict]] | None,
    *,
    assisted_attributes: bool,
) -> list[dict]:
    output = []
    for frame in frames:
        labels = ground_truth.get(frame["frame_index"], []) if ground_truth else []
        if assisted_attributes:
            matches = _maximum_footpoint_matches(
                [{"bbox_xyxy_px": detection["bbox_xyxy_px"]}
                 for detection in frame["detections"]],
                labels,
                FOOTPOINT_MATCH_TOLERANCE_PX,
            )
            assisted_by_detection = {
                detection_index: labels[annotation_index]
                for detection_index, annotation_index, _ in matches
            }
        else:
            assisted_by_detection = {}
        measurements = []
        for index, detection in enumerate(frame["detections"]):
            if assisted_attributes:
                annotation = assisted_by_detection.get(index)
                if annotation is None:
                    continue
                team = (
                    {"left": 1, "right": 2}.get(annotation["team"], 0)
                    if annotation is not None else 0
                )
                goalkeeper = (
                    annotation["role"] == "goalkeeper"
                    if annotation is not None and annotation["role"] in ROLE_CLASSES
                    else None
                )
                attribute_source = "MATCHED_HELDOUT_ANNOTATION_ASSISTED_DIAGNOSTIC"
            else:
                team = detection["team"]
                goalkeeper = detection["goalkeeper"]
                attribute_source = "VIDEO_APPEARANCE_CLASSIFIER_OR_UNKNOWN"
            measurements.append({
                "track": detection["track"],
                "segment": 0,
                "team": team if team else "UNKNOWN",
                "goalkeeper": goalkeeper,
                "position_status": "observed",
                "xy": detection["pitch_xy_m"],
                "quality": {
                    "status": "UNVALIDATED_EXPERIMENTAL",
                    "projection": detection["projection_status"],
                    "detector_confidence": detection["detector_confidence"],
                    "team_confidence": detection["team_confidence"],
                    "role_confidence": detection["role_confidence"],
                    "attribute_source": attribute_source,
                    "position_accuracy_m": None,
                },
            })
        output.append({
            "record_type": "frame",
            "frame_index": frame["frame_index"],
            "timestamp_s": frame["timestamp_s"],
            "segment_id": frame["segment_id"],
            "identity_reset": False,
            "camera_cut": False,
            "measurements": measurements,
            "absent_count": 0,
        })
    return output


def _write_external(path: Path, rows: list[dict], start_s: int, assisted: bool) -> None:
    header = {
        "record_type": "metadata",
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "source_id": f"overnight-video-{start_s}-{'assisted' if assisted else 'autonomous'}",
        "match": LOCAL_MATCH,
        "half": LOCAL_HALF,
        "observation_kind": "exo_source",
        "sampling_seconds": 0.2,
        "missing_count_semantics": "detections outside the fixed experimental pitch support are omitted; inventory is incomplete",
        "coordinates": {
            "system": "pitch_top_left",
            "units": "metres",
            "axes": "x_length_y_width",
            "pitch_length_m": 105,
            "pitch_width_m": 68,
        },
        "origin": {
            "type": "native_resolution_video_detector_footpoints",
            "status": "UNVALIDATED_EXPERIMENTAL",
            "video_sha256": file_digest(VIDEO_PATH),
            "detector_sha256": file_digest(BEST_DETECTOR),
        },
        "calibration": {
            "type": "ASSISTED_LOCAL_DEMO",
            "global_gate_status": "FAILED_UNCHANGED",
            "region": "fixed +/-650 px around released optical principal point and inside fit-landmark hull",
            "position_accuracy": "UNVALIDATED",
        },
        "synchronization": {
            "status": "native_frame_clock",
            "native_fps": NATIVE_FPS,
            "sample_stride_native_frames": SAMPLE_STRIDE,
        },
        "quality": {
            "position_accuracy": "UNVALIDATED",
            "team_and_role": (
                "heldout annotation-assisted diagnostic only"
                if assisted else "appearance estimate with confidence abstention"
            ),
        },
    }
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(header, allow_nan=False, separators=(",", ":")) + "\n")
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def _select_actor(frames: list[dict]) -> str | None:
    counts: dict[str, int] = {}
    for frame in frames:
        for measurement in frame["measurements"]:
            counts[measurement["track"]] = counts.get(measurement["track"], 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda actor: (counts[actor], actor))


def _replay_adapter(output: Path, external_path: Path, frames: list[dict], start_s: int) -> dict:
    config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
    experiment = ROOT / config["experiment"]
    world = ModelSession(experiment, config["world_backend"], device="cpu")
    action = ActionModelSession(ROOT / config["action_checkpoint"], device="cpu")
    source = ExternalObservationSource(external_path)
    actor = _select_actor(frames)
    if actor is None:
        return {
            "actor": None,
            "g1_checkpoint_loaded_once": world.load_counts["world"] == 1,
            "g1_forward_calls": 0,
            "s4_checkpoint_loaded_once": action.load_count == 1,
            "s4_neural_forward_calls": 0,
            "status": "NO_SUPPORTED_ACTOR",
        }
    provider = SyntheticObservationProvider(actor, seed=42, fov=120, dropout=False)
    engine = ReplayEngine(world, expiry_s=6., fallback="last_seen", action_model=action)
    logical_start = start_s * NATIVE_FPS
    replay_frames = list(source.frames(logical_start, len(frames), stride=SAMPLE_STRIDE))
    if len(replay_frames) != len(frames):
        raise ValueError("The external observation adapter did not preserve the 5 Hz sample grid")
    observations = []
    for frame in replay_frames:
        observations.append(source.observe(frame, provider))
    continuous_actor = []
    current_run = []
    for index, observation in enumerate(observations):
        if observation.actor is not None:
            current_run.append(index)
        else:
            if len(current_run) > len(continuous_actor):
                continuous_actor = current_run
            current_run = []
    if len(current_run) > len(continuous_actor):
        continuous_actor = current_run
    query_index = next(
        (index for index in range(len(continuous_actor) - 10)
         if continuous_actor[index + 10] - continuous_actor[index] == 10),
        None,
    )
    query_frame = continuous_actor[query_index + 10] if query_index is not None else None
    rows = []
    for index, observation in enumerate(observations):
        rows.append(engine.step(
            observation,
            include_passing_assessment=False,
            action_query=index == query_frame,
        ))
    status_path = output / f"g1_s4_{start_s}_{'assisted' if 'assisted' in external_path.name else 'autonomous'}.jsonl"
    with status_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    queried = rows[query_frame] if query_frame is not None else None
    return {
        "actor_track": actor,
        "actor_observation_count": sum(
            observation.actor is not None for observation in observations
        ),
        "longest_continuous_actor_run_samples": len(continuous_actor),
        "g1_checkpoint_loaded_once": world.load_counts["world"] == 1,
        "g1_forward_calls": world.world_forward_calls,
        "g1_contract_unavailable_samples": sum(
            "WORLD_CONTRACT_UNAVAILABLE" in warning
            for row in rows for warning in row["belief"]["warnings"]
        ),
        "s4_checkpoint_loaded_once": action.load_count == 1,
        "s4_query_attempts": int(query_frame is not None),
        "s4_query_frame_index": (
            int(replay_frames[query_frame].index) if query_frame is not None else None
        ),
        "s4_neural_forward_calls": action.forward_calls,
        "s4_query_result": (
            queried.get("action_inference") if queried is not None else None
        ),
        "s4_query_semantics": (
            "One supplied actor/time diagnostic query; no action event annotation or outcome is provided."
        ),
        "status_file": str(status_path.relative_to(ROOT)),
    }


def _draw_pitch(canvas: np.ndarray, detections: list[dict]) -> None:
    x0, y0, width, height = 1315, 145, 560, 365
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (242, 242, 242), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (45, 130, 45), 2)
    cv2.line(canvas, (x0 + width // 2, y0), (x0 + width // 2, y0 + height), (120, 120, 120), 1)
    cv2.circle(canvas, (x0 + width // 2, y0 + height // 2), 49, (120, 120, 120), 1)
    for detection in detections:
        pitch = detection["pitch_xy_m"]
        if pitch is None:
            continue
        px = x0 + round(float(pitch[0]) / 105 * width)
        py = y0 + height - round(float(pitch[1]) / 68 * height)
        color = (0, 200, 0) if detection["team"] == 1 else (
            (255, 100, 0) if detection["team"] == 2 else (0, 180, 255)
        )
        if detection["goalkeeper"] is True:
            cv2.rectangle(canvas, (px - 6, py - 6), (px + 6, py + 6), color, -1)
        else:
            cv2.circle(canvas, (px, py), 5, color, -1)


def _render_comparison(
    output: Path, start_s: int, frames: list[dict],
) -> dict:
    capture = cv2.VideoCapture(str(VIDEO_PATH), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen source video: {VIDEO_PATH}")
    path = output / f"before_after_{start_s}.mp4"
    process = subprocess.Popen(
        [
            str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "1920x1080", "-r", "5", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "24", "-movflags", "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        capture.release()
        process.kill()
        raise RuntimeError("Could not start ffmpeg video renderer")
    try:
        for frame in frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame["frame_index"])
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode video frame {frame['frame_index']}")
            canvas = np.full((1080, 1920, 3), (24, 24, 24), np.uint8)
            cv2.putText(
                canvas, "BEFORE: COCO YOLO11n | AFTER: FOOTPOINT-FINETUNED YOLO11n",
                (32, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{LOCAL_MATCH} {LOCAL_HALF} | {frame['timestamp_s']:.1f}s | "
                "EXPERIMENTAL; local assisted calibration",
                (32, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2,
                cv2.LINE_AA,
            )
            left = cv2.resize(image, (1280, 337), interpolation=cv2.INTER_AREA)
            x_scale, y_scale = 1280 / FRAME_WIDTH, 337 / FRAME_HEIGHT
            for detection in frame["baseline_person_detections"]:
                x1, y1, x2, y2 = detection["bbox_xyxy_px"]
                cv2.rectangle(
                    left, (round(x1 * x_scale), round(y1 * y_scale)),
                    (round(x2 * x_scale), round(y2 * y_scale)), (0, 180, 255), 1,
                )
            cv2.putText(left, "BASELINE COCO PERSON BOXES", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
            right = cv2.resize(image, (1280, 337), interpolation=cv2.INTER_AREA)
            for detection in frame["detections"]:
                x1, y1, x2, y2 = detection["bbox_xyxy_px"]
                color = (0, 220, 0) if detection["team"] else (0, 180, 255)
                cv2.rectangle(
                    right, (round(x1 * x_scale), round(y1 * y_scale)),
                    (round(x2 * x_scale), round(y2 * y_scale)), color, 1,
                )
                cv2.putText(
                    right,
                    f"{detection['team_prediction'][:1].upper()}/{detection['role_prediction'][:1].upper()}",
                    (round(x1 * x_scale), max(14, round(y1 * y_scale) - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )
            cv2.putText(right, "FINE-TUNED PLAYER FOOTPOINTS + ATTRIBUTES",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 0), 2)
            canvas[110:447, 15:1295] = left
            canvas[465:802, 15:1295] = right
            _draw_pitch(canvas, frame["detections"])
            cv2.putText(
                canvas,
                "Green=team-left | Blue=team-right | Amber=attribute UNKNOWN | "
                "Square=predicted goalkeeper",
                (1315, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
            )
            cv2.putText(
                canvas,
                "Team/role estimates are experimental; unknown attributes abstain. "
                "Scores are not tactical value.",
                (1315, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1,
            )
            try:
                process.stdin.write(canvas.tobytes())
            except BrokenPipeError as error:
                stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                raise RuntimeError(f"ffmpeg renderer failed: {stderr}") from error
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg renderer exited {return_code}: {stderr}")
    finally:
        capture.release()
        if process.poll() is None:
            process.kill()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": file_digest(path),
        "duration_seconds": len(frames) / 5,
        "fps": 5,
        "frames": len(frames),
        "dimensions_px": [1920, 1080],
    }


def run_segment(start_s: int, duration_s: int, output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = (BEST_DETECTOR, ATTRIBUTE_MODEL, ATTRIBUTE_PROVENANCE, BASE_DETECTOR, FFMPEG)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing replay prerequisites: {missing}")
    if duration_s != 60 or (start_s, start_s + duration_s) not in EVALUATION_SEGMENTS:
        raise ValueError(f"Only the frozen 60-second evaluation segments are accepted: {EVALUATION_SEGMENTS}")
    if file_digest(VIDEO_PATH) != (
            "f50dfebc70743439a8c778abe06953885a50be8b47e3c9d3d1ce8d533fe315c8"):
        raise ValueError("Source video changed since the audited detector and calibration setup")
    train_features, train_teams, train_roles, attr_metadata = _load_appearance_model()
    started = time.monotonic()
    raw_path = output / f"video_detections_{start_s}.jsonl"
    if raw_path.is_file():
        frames = _read_saved_detections(raw_path, start_s, duration_s)
        inference_reused = True
        projection_rebuilt = any(
            frame.get("projector_version") != PROJECTOR_VERSION
            for frame in frames
        )
        if projection_rebuilt:
            frames = _rebuild_saved_projection(
                frames, start_s, train_features, train_teams, train_roles,
                attr_metadata,
            )
            with raw_path.open("w", encoding="utf-8", newline="\n") as stream:
                for frame in frames:
                    stream.write(json.dumps(frame, allow_nan=False, separators=(",", ":")) + "\n")
    else:
        from ultralytics import YOLO

        detector = YOLO(str(BEST_DETECTOR))
        baseline = YOLO(str(BASE_DETECTOR))
        frames = _infer_segment(
            start_s, duration_s, detector, baseline, train_features, train_teams,
            train_roles, attr_metadata,
        )
        with raw_path.open("w", encoding="utf-8", newline="\n") as stream:
            for frame in frames:
                stream.write(json.dumps(frame, allow_nan=False, separators=(",", ":")) + "\n")
        inference_reused = False
        projection_rebuilt = False

    ground_truth = _read_ground_truth(
        start_s * NATIVE_FPS,
        (start_s + duration_s) * NATIVE_FPS,
    )
    evaluation = _paired_evaluation(frames, ground_truth)
    evaluation["heldout_label_source_sha256"] = file_digest(ANNOTATION_PATH)
    evaluation["labels_read_after_video_inference"] = True
    auto_rows = _build_external_rows(frames, None, assisted_attributes=False)
    assisted_rows = _build_external_rows(
        frames, ground_truth, assisted_attributes=True,
    )
    auto_path = output / f"external_autonomous_{start_s}.jsonl"
    assisted_path = output / f"external_attribute_assisted_{start_s}.jsonl"
    _write_external(auto_path, auto_rows, start_s, assisted=False)
    _write_external(assisted_path, assisted_rows, start_s, assisted=True)

    auto_calls = _replay_adapter(output, auto_path, auto_rows, start_s)
    assisted_calls = _replay_adapter(output, assisted_path, assisted_rows, start_s)
    video = _render_comparison(
        output, start_s, frames,
    )
    result = {
        "schema_version": "football.overnight-video-player-replay.1",
        "status": "EXPERIMENTAL_HELDOUT_VIDEO_REPLAY",
        "interval_s": [start_s, start_s + duration_s],
        "continuous_samples": len(frames),
        "continuous_coverage_seconds": duration_s,
        "sample_rate_hz": NATIVE_FPS / SAMPLE_STRIDE,
        "source_video_sha256": file_digest(VIDEO_PATH),
        "detector_sha256": file_digest(BEST_DETECTOR),
        "attribute_classifier_sha256": file_digest(ATTRIBUTE_MODEL),
        "detector_confidence_threshold": DETECTION_CONFIDENCE,
        "attribute_abstention_thresholds": {
            "team": attr_metadata["team_confidence_threshold"],
            "role": attr_metadata["role_confidence_threshold"],
        },
        "heldout_evaluation": evaluation,
        "autonomous_video_estimated_route": {
            "attributes": "appearance estimates with class-specific calibrated abstention; otherwise UNKNOWN",
            "external_observation_file": str(auto_path.relative_to(ROOT)),
            "adapter_calls": auto_calls,
            "annotation_attributes_used_in_inference": False,
            "position_status": "EXPERIMENTAL; local setup calibration only",
        },
        "attribute_assisted_diagnostic": {
            "attributes": "heldout annotation attributes assigned only to geometrically matched video detections; unmatched tracks stay UNKNOWN",
            "external_observation_file": str(assisted_path.relative_to(ROOT)),
            "adapter_calls": assisted_calls,
            "heldout_annotation_assisted": True,
            "not_autonomous_or_benchmark_eligible": True,
            "unmatched_detections_policy": "Excluded from this explicitly assisted diagnostic only; retained in autonomous route and raw evaluation."
        },
        "video": video,
        "raw_video_detections": str(raw_path.relative_to(ROOT)),
        "elapsed_seconds": time.monotonic() - started,
        "video_inference_reused_from_complete_raw_output": inference_reused,
        "projection_rebuilt_without_rerunning_detectors": projection_rebuilt,
        "limitations": [
            "Global video-to-pitch validation remains failed; only the fixed local assisted region is used.",
            "Tracks are local geometric associations and are not certified player identities.",
            "Appearance team/role predictions may abstain; unknown attributes block G1/S4.",
            "Player visibility is simulated, not measured gaze.",
            "S4 is a supplied actor/time event-conditioned classifier; no event label or outcome is provided.",
            "Scores are not tactical value or best-action recommendations.",
            "Ball and possession remain unknown.",
            "Predictions are experimental and are not added to training or benchmark corpora.",
        ],
    }
    result_path = output / f"replay_result_{start_s}.json"
    write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-s", type=int, choices=[segment[0] for segment in EVALUATION_SEGMENTS], required=True)
    parser.add_argument("--duration-s", type=int, default=60)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "replays")
    args = parser.parse_args()
    result = run_segment(args.start_s, args.duration_s, args.output)
    print(json.dumps({
        "status": result["status"],
        "interval_s": result["interval_s"],
        "detection": result["heldout_evaluation"]["detection"],
        "position_error_m": result["heldout_evaluation"]["position_error_m"],
        "g1_forward_calls_autonomous": result["autonomous_video_estimated_route"]["adapter_calls"]["g1_forward_calls"],
        "g1_forward_calls_assisted": result["attribute_assisted_diagnostic"]["adapter_calls"]["g1_forward_calls"],
        "s4_forward_calls_autonomous": result["autonomous_video_estimated_route"]["adapter_calls"]["s4_neural_forward_calls"],
        "s4_forward_calls_assisted": result["attribute_assisted_diagnostic"]["adapter_calls"]["s4_neural_forward_calls"],
        "video": result["video"]["path"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
