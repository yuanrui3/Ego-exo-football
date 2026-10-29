"""Fit and evaluate a small appearance-only team/goalkeeper classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.backbone.video_player_overnight import (
    ANNOTATION_PATH,
    OUTPUT_ROOT,
    TRAIN_FRAME_RANGE,
    VALIDATION_FRAME_RANGE,
    VIDEO_PATH,
    _read_annotation_windows,
    detect_native,
)
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
BASE_WEIGHTS = ROOT / "yolo11n.pt"
MATCH_DISTANCE_PX = 40.0
TRAIN_SAMPLE_STRIDE = 10
K_NEIGHBORS = 11
TEAM_CLASSES = ("left", "right")
ROLE_CLASSES = ("player", "goalkeeper")


def _footpoint(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, box[3]


def _maximum_footpoint_matches(
    detections: list[dict], annotations: list[dict], max_distance: float,
) -> list[tuple[int, int, float]]:
    distances = []
    for detection_index, detection in enumerate(detections):
        point = _footpoint(detection["bbox_xyxy_px"])
        for annotation_index, annotation in enumerate(annotations):
            target = annotation["footpoint_xy_px"]
            distance = float(np.hypot(point[0] - target[0], point[1] - target[1]))
            if distance <= max_distance:
                distances.append((distance, detection_index, annotation_index))
    by_detection: dict[int, list[tuple[float, int]]] = {}
    for distance, detection_index, annotation_index in distances:
        by_detection.setdefault(detection_index, []).append((distance, annotation_index))
    for candidates in by_detection.values():
        candidates.sort()
    annotation_to_detection: dict[int, int] = {}

    def assign(detection_index: int, seen: set[int]) -> bool:
        for _, annotation_index in by_detection.get(detection_index, []):
            if annotation_index in seen:
                continue
            seen.add(annotation_index)
            current = annotation_to_detection.get(annotation_index)
            if current is None or assign(current, seen):
                annotation_to_detection[annotation_index] = detection_index
                return True
        return False

    for detection_index in sorted(
        by_detection,
        key=lambda index: (
            by_detection[index][0][0] if by_detection[index] else float("inf"),
            index,
        ),
    ):
        assign(detection_index, set())
    matches = []
    for annotation_index, detection_index in annotation_to_detection.items():
        distance = next(
            distance for distance, candidate in by_detection[detection_index]
            if candidate == annotation_index
        )
        matches.append((detection_index, annotation_index, distance))
    return sorted(matches, key=lambda match: match[0])


def appearance_feature(image: np.ndarray, bbox_xyxy: list[float]) -> np.ndarray:
    """Describe the upper-body jersey region while excluding most green field pixels."""
    x1, y1, x2, y2 = bbox_xyxy
    width, height = x2 - x1, y2 - y1
    left = max(0, int(round(x1 + 0.20 * width)))
    right = min(image.shape[1], int(round(x2 - 0.20 * width)))
    top = max(0, int(round(y1 + 0.12 * height)))
    bottom = min(image.shape[0], int(round(y1 + 0.62 * height)))
    if right <= left or bottom <= top:
        return np.zeros(104, dtype=np.float32)
    crop = image[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    green_field = (
        (hsv[:, :, 0] >= 32) & (hsv[:, :, 0] <= 98)
        & (hsv[:, :, 1] >= 55)
    )
    foreground = np.where(green_field, 0, 255).astype(np.uint8)
    hue_sat = cv2.calcHist(
        [hsv], [0, 1], foreground, [16, 4], [0, 180, 0, 256],
    ).reshape(-1)
    lab_hist = cv2.calcHist(
        [lab], [1, 2], foreground, [6, 6], [0, 256, 0, 256],
    ).reshape(-1)
    hue_sat /= max(float(hue_sat.sum()), 1.0)
    lab_hist /= max(float(lab_hist.sum()), 1.0)
    value = hsv[:, :, 2].astype(np.float32)
    summary = np.asarray([
        float(np.mean(value)) / 255.0,
        float(np.std(value)) / 255.0,
        float(np.mean(hsv[:, :, 1])) / 255.0,
        float(np.mean(green_field)),
    ], dtype=np.float32)
    return np.concatenate((hue_sat, lab_hist, summary)).astype(np.float32)


def _appearance_samples(start: int, end: int) -> tuple[np.ndarray, list[str], list[str], dict]:
    from ultralytics import YOLO

    if not BASE_WEIGHTS.is_file():
        raise FileNotFoundError(f"Official COCO appearance-box weights are missing: {BASE_WEIGHTS}")
    model = YOLO(str(BASE_WEIGHTS))
    annotations_by_frame = _read_annotation_windows({"selected": (start, end)})
    if not annotations_by_frame:
        raise ValueError(f"No tracked annotations in attribute interval {start}:{end}")
    capture = cv2.VideoCapture(str(VIDEO_PATH), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {VIDEO_PATH}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    features, teams, roles = [], [], []
    last_sample_by_player: dict[str, int] = {}
    total_detections = total_matches = 0
    try:
        for frame_index in range(start, end):
            valid, image = capture.read()
            if not valid:
                raise RuntimeError(f"Could not decode source frame {frame_index}")
            if frame_index % TRAIN_SAMPLE_STRIDE:
                continue
            detections = detect_native(model, image, confidence=0.05)
            frame_annotations = annotations_by_frame.get(frame_index, [])
            matches = _maximum_footpoint_matches(
                detections, frame_annotations, MATCH_DISTANCE_PX,
            )
            total_detections += len(detections)
            total_matches += len(matches)
            for detection_index, annotation_index, _ in matches:
                annotation = frame_annotations[annotation_index]
                team, role = annotation["team"], annotation["role"]
                if team not in TEAM_CLASSES or role not in ROLE_CLASSES:
                    continue
                player_key = annotation["player_id"] or annotation["track_id"]
                if frame_index - last_sample_by_player.get(player_key, -10_000) < 50:
                    continue
                features.append(appearance_feature(
                    image, detections[detection_index]["bbox_xyxy_px"],
                ))
                teams.append(team)
                roles.append(role)
                last_sample_by_player[player_key] = frame_index
    finally:
        capture.release()
    if not features:
        raise ValueError("No appearance samples could be paired with verified fitting annotations")
    return np.stack(features), teams, roles, {
        "interval_frame_indices": [start, end],
        "frame_sampling_stride": TRAIN_SAMPLE_STRIDE,
        "detector_weight_sha256": file_digest(BASE_WEIGHTS),
        "detector_confidence_threshold": 0.05,
        "footpoint_match_threshold_px": MATCH_DISTANCE_PX,
        "detections": total_detections,
        "matched_detection_annotation_pairs": total_matches,
        "appearance_samples_after_track_downsampling": len(features),
        "team_counts": {label: teams.count(label) for label in TEAM_CLASSES},
        "role_counts": {label: roles.count(label) for label in ROLE_CLASSES},
    }


def _knn_scores(
    train_features: np.ndarray,
    train_labels: list[str],
    query_features: np.ndarray,
    classes: tuple[str, ...],
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    if not len(train_features):
        raise ValueError("Attribute classifier has no fitting samples")
    k = min(k, len(train_features))
    class_counts = {label: max(1, train_labels.count(label)) for label in classes}
    class_balance = {
        label: len(train_labels) / (len(classes) * count)
        for label, count in class_counts.items()
    }
    scores = np.zeros((len(query_features), len(classes)), dtype=np.float32)
    train_norms = np.einsum("ij,ij->i", train_features, train_features)
    for index, feature in enumerate(query_features):
        distances = np.maximum(
            train_norms + float(feature @ feature) - 2 * (train_features @ feature),
            0.0,
        )
        nearest = np.argpartition(distances, k - 1)[:k]
        nearest = nearest[np.argsort(distances[nearest])]
        weights = 1.0 / (np.sqrt(distances[nearest]) + 0.02)
        for neighbor, weight in zip(nearest, weights, strict=True):
            label = train_labels[int(neighbor)]
            scores[index, classes.index(label)] += (
                float(weight) * class_balance[label]
            )
        total = float(scores[index].sum())
        if total:
            scores[index] /= total
    return scores


def _attribute_metrics(
    expected: list[str], scores: np.ndarray, classes: tuple[str, ...],
) -> dict:
    prediction_indices = scores.argmax(axis=1)
    predictions = [classes[index] for index in prediction_indices]
    confidence = scores.max(axis=1)
    result = {
        "count": len(expected),
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(expected))),
        "class_counts": {label: expected.count(label) for label in classes},
        "threshold_results": {},
        "selected_confidence_threshold": None,
    }
    chosen = None
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())
        accuracy = (
            float(np.mean(np.asarray(predictions)[accepted] == np.asarray(expected)[accepted]))
            if accepted_count else None
        )
        predicted_counts = {
            label: sum(1 for i, pred in enumerate(predictions)
                       if accepted[i] and pred == label)
            for label in classes
        }
        precision_by_class = {
            label: (
                sum(1 for i, pred in enumerate(predictions)
                    if accepted[i] and pred == label and expected[i] == label)
                / predicted_counts[label]
                if predicted_counts[label] else None
            )
            for label in classes
        }
        result["threshold_results"][str(threshold)] = {
            "accepted": accepted_count,
            "coverage": accepted_count / max(1, len(expected)),
            "accuracy": accuracy,
            "predicted_counts": predicted_counts,
            "precision_by_class": precision_by_class,
        }
        active_precisions = [
            value for label, value in precision_by_class.items()
            if predicted_counts[label] >= 10 and value is not None
        ]
        if (accepted_count >= 25 and accuracy is not None and accuracy >= 0.90
                and len(active_precisions) == len(classes)
                and all(value >= 0.80 for value in active_precisions)):
            chosen = threshold
            break
    result["selected_confidence_threshold"] = chosen
    return result


def fit(output: Path) -> dict:
    """Train on 1500-1620 s; use 1620-1680 s only for abstention calibration."""
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite attribute-model output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_features, train_teams, train_roles, train_counts = _appearance_samples(
        *TRAIN_FRAME_RANGE,
    )
    validation_features, validation_teams, validation_roles, validation_counts = (
        _appearance_samples(*VALIDATION_FRAME_RANGE)
    )
    team_scores = _knn_scores(
        train_features, train_teams, validation_features, TEAM_CLASSES,
    )
    role_scores = _knn_scores(
        train_features, train_roles, validation_features, ROLE_CLASSES,
    )
    team_metrics = _attribute_metrics(validation_teams, team_scores, TEAM_CLASSES)
    role_metrics = _attribute_metrics(validation_roles, role_scores, ROLE_CLASSES)
    model_path = output / "appearance_knn.npz"
    np.savez_compressed(
        model_path,
        features=train_features,
        teams=np.asarray(train_teams, dtype="U16"),
        roles=np.asarray(train_roles, dtype="U16"),
    )
    summary = {
        "schema_version": "football.video-player-appearance-classifier.1",
        "status": "EXPERIMENTAL_AUTONOMOUS_ATTRIBUTES",
        "feature_definition": (
            "HSV hue/saturation and Lab chroma histograms from the upper-body crop of "
            "COCO-person boxes; excludes most green pixels. No pitch coordinates, jersey "
            "numbers, player IDs, or evaluation labels are model inputs."
        ),
        "label_mapping": {
            "team_left": 1,
            "team_right": 2,
            "role_player": False,
            "role_goalkeeper": True,
        },
        "fitting": train_counts,
        "validation": {
            **validation_counts,
            "team_metrics": team_metrics,
            "role_metrics": role_metrics,
        },
        "k_neighbors": K_NEIGHBORS,
        "training_attributes_are_appearance_only": True,
        "team_confidence_threshold": team_metrics["selected_confidence_threshold"],
        "role_confidence_threshold": role_metrics["selected_confidence_threshold"],
        "classifier_npz": str(model_path.relative_to(ROOT)),
        "classifier_sha256": file_digest(model_path),
        "annotation_source_sha256": file_digest(ANNOTATION_PATH),
        "source_video_sha256": file_digest(VIDEO_PATH),
        "feature_samples_npz_contains_only_fitting_labels": True,
    }
    write_json(output / "appearance_classifier_provenance.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=OUTPUT_ROOT / "attribute_model",
    )
    args = parser.parse_args()
    print(json.dumps(fit(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
