"""Create conservative image-space tracklet proposals from existing pilot detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
MAX_LINK_DISTANCE_PX = 260.0
CUT_THRESHOLD = 0.55


def pixel_to_normalized(x_px: float, y_px: float, width_px: int, height_px: int) -> tuple[float, float]:
    if width_px <= 0 or height_px <= 0:
        raise ValueError("Pixel image dimensions must be positive")
    if not np.isfinite([x_px, y_px]).all() or not (0 <= x_px <= width_px and 0 <= y_px <= height_px):
        raise ValueError("Pixel coordinates must be finite and inside the image bounds")
    return x_px / width_px, y_px / height_px


def _histogram(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (320, 160), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist


def _shot_boundaries(video_path: Path, frame_indices: list[int], threshold: float) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open existing pilot video: {video_path}")
    histograms = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            valid, frame = capture.read()
            if not valid:
                raise RuntimeError(f"Could not decode sampled pilot frame {frame_index}")
            histograms.append(_histogram(frame))
    finally:
        capture.release()
    transitions = []
    for index in range(len(histograms)):
        distance = None if index == 0 else float(cv2.compareHist(
            histograms[index - 1], histograms[index], cv2.HISTCMP_BHATTACHARYYA,
        ))
        transitions.append({
            "frame_index": frame_indices[index],
            "potential_cut_before": bool(distance is not None and distance >= threshold),
            "histogram_distance_from_previous_sample": distance,
        })
    return transitions


def _box(detection: dict) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, detection["bbox_xyxy_px"])
    return x1, y1, x2, y2


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-12, left_area + right_area - intersection)


def associate_tracklets(
    records: list[dict],
    cut_flags: list[bool] | None = None,
    *,
    max_distance_px: float = MAX_LINK_DISTANCE_PX,
) -> tuple[list[dict], list[dict]]:
    if cut_flags is None:
        cut_flags = [False] * len(records)
    if len(cut_flags) != len(records):
        raise ValueError("Each sampled frame requires exactly one cut-boundary flag")
    if max_distance_px <= 0:
        raise ValueError("Maximum association distance must be positive")
    next_id = 1
    previous: list[dict] = []
    linked_records, tracklets = [], {}
    for frame_number, source_record in enumerate(records):
        if cut_flags[frame_number] or (frame_number and (
                source_record["frame_index"] - records[frame_number - 1]["frame_index"] > 25)):
            previous = []
        current = []
        for detection_index, source_detection in enumerate(source_record.get("detections", [])):
            detection = dict(source_detection)
            detection["frame_index"] = int(source_record["frame_index"])
            detection["time_from_video_start_s"] = float(source_record["time_from_video_start_s"])
            detection["detection_index"] = detection_index
            detection["tracklet_id"] = None
            detection["link_status"] = "NEW_TRACKLET"
            current.append(detection)

        possible = []
        for previous_index, old in enumerate(previous):
            for current_index, new in enumerate(current):
                distance = float(np.linalg.norm(
                    np.asarray(_center(_box(old))) - np.asarray(_center(_box(new)))
                ))
                if distance <= max_distance_px:
                    overlap = _iou(_box(old), _box(new))
                    possible.append((distance, -overlap, previous_index, current_index))
        possible.sort()
        ambiguous_current: set[int] = set()
        for current_index in range(len(current)):
            candidate_distances = sorted(
                item[0] for item in possible if item[3] == current_index
            )
            if len(candidate_distances) > 1 and candidate_distances[1] - candidate_distances[0] < 25.0:
                ambiguous_current.add(current_index)

        used_previous: set[int] = set()
        used_current: set[int] = set()
        for distance, negative_overlap, previous_index, current_index in possible:
            if (previous_index in used_previous or current_index in used_current
                    or current_index in ambiguous_current):
                continue
            tracklet_id = previous[previous_index]["tracklet_id"]
            current[current_index]["tracklet_id"] = tracklet_id
            current[current_index]["link_status"] = "UNVERIFIED_ADJACENT_SAMPLE_MATCH"
            used_previous.add(previous_index)
            used_current.add(current_index)
            tracklets[tracklet_id]["observations"].append({
                "frame_index": current[current_index]["frame_index"],
                "time_from_video_start_s": current[current_index]["time_from_video_start_s"],
                "detection_index": current_index,
                "match_distance_px": distance,
                "match_iou": -negative_overlap,
            })
        for current_index, detection in enumerate(current):
            if current_index in ambiguous_current:
                detection["link_status"] = "AMBIGUOUS; NEW_TRACKLET"
            if detection["tracklet_id"] is None:
                tracklet_id = f"pixel-tracklet-{next_id:04d}"
                next_id += 1
                detection["tracklet_id"] = tracklet_id
                tracklets[tracklet_id] = {
                    "tracklet_id": tracklet_id,
                    "status": "UNVERIFIED_IMAGE_SPACE_CANDIDATE",
                    "observations": [{
                        "frame_index": detection["frame_index"],
                        "time_from_video_start_s": detection["time_from_video_start_s"],
                        "detection_index": current_index,
                    }],
                }
        linked_records.append({
            **source_record,
            "detections": current,
            "association_scope": "adjacent sampled frames only; no interpolation across empty samples or cuts",
        })
        previous = current if current else []
    return linked_records, list(tracklets.values())


def _write_tracklet_overlays(
    original_pilot: Path,
    output: Path,
    linked_records: list[dict],
    image_width: int,
    image_height: int,
) -> list[str]:
    overlay_paths = []
    for record in linked_records:
        source_overlay = original_pilot / f"overlay_{record['frame_index']}.jpg"
        if not source_overlay.is_file():
            continue
        image = cv2.imread(str(source_overlay))
        if image is None:
            raise RuntimeError(f"Could not read existing pilot overlay: {source_overlay}")
        scale_x = image.shape[1] / image_width
        scale_y = image.shape[0] / image_height
        for detection in record["detections"]:
            x1, y1, x2, y2 = _box(detection)
            p1 = (round(x1 * scale_x), round(y1 * scale_y))
            p2 = (round(x2 * scale_x), round(y2 * scale_y))
            cv2.rectangle(image, p1, p2, (255, 180, 0), 2)
            cv2.putText(
                image, detection["tracklet_id"],
                (p1[0], max(18, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 180, 0), 1, cv2.LINE_AA,
            )
        target = output / f"tracklets_{record['frame_index']}.jpg"
        if not cv2.imwrite(str(target), image):
            raise RuntimeError(f"Could not save tracklet overlay: {target}")
        overlay_paths.append(target.name)
    return overlay_paths


def build_extension(pilot_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing video-pilot extension: {output}")
    manifest = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("pixel_observations", {}).get("status") != "OBTAINED":
        raise ValueError("The existing pilot does not contain real pixel detections")
    detections_path = pilot_dir / "pixel_observations.jsonl"
    records = [
        json.loads(line) for line in detections_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or any(record.get("track_ids") is not None for record in records):
        raise ValueError("Expected the original detection-only pilot without assigned identities")
    output.mkdir(parents=True)
    transitions = _shot_boundaries(
        Path(manifest["source_video"]["path"]),
        [int(record["frame_index"]) for record in records],
        CUT_THRESHOLD,
    )
    cut_flags = [row["potential_cut_before"] for row in transitions]
    linked_records, tracklets = associate_tracklets(records, cut_flags)
    for record in linked_records:
        frame_transition = next(row for row in transitions if row["frame_index"] == record["frame_index"])
        record["shot_boundary_evidence"] = frame_transition
    (output / "pixel_tracklets.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in linked_records),
        encoding="utf-8",
    )
    write_json(output / "tracklet_summary.json", {
        "status": "PIXEL_TRACKLET_CANDIDATES_ONLY",
        "tracklets": tracklets,
        "shot_boundary_method": (
            "Heuristic Bhattacharyya distance on low-resolution HSV histograms between the existing 1Hz samples; "
            "threshold is conservative but not an independently validated shot detector."
        ),
        "potential_cut_threshold": CUT_THRESHOLD,
        "maximum_adjacent_link_distance_px": MAX_LINK_DISTANCE_PX,
        "identity_semantics": "Unverified image-space candidates, not persistent athlete identities.",
        "team_assignments": "UNKNOWN",
        "pitch_positions": None,
        "calibrated_position_accuracy": None,
        "new_video_downloads": False,
        "detector_rerun": False,
    })
    width, height = map(int, records[0]["dimensions_px"])
    overlays = _write_tracklet_overlays(pilot_dir, output, linked_records, width, height)
    calibration = {
        "schema_version": "football.pixel-coordinate-sidecar.0.1",
        "image_dimensions_px": [width, height],
        "source_coordinate_system": "continuous pixel coordinates, origin at top-left, x right, y down",
        "normalized_image_coordinates": {
            "status": "SUPPORTED",
            "formula": ["x_px / width_px", "y_px / height_px"],
            "units": "dimensionless image coordinates in [0, 1]",
        },
        "pitch_coordinates": {
            "status": "BLOCKED",
            "transform": None,
            "reason": "No accepted per-camera/seam mapping; image-space normalization is not a pitch projection.",
        },
        "synchronization": "NOT_VALIDATED",
        "eligible_for_pitch_backbone": False,
    }
    write_json(output / "pixel_calibration_sidecar.json", calibration)
    write_json(output / "manifest.json", {
        "schema_version": "football.video-pilot-extension.0.4",
        "status": "PIXEL_TRACKLET_CANDIDATES_READY_PITCH_ADAPTER_BLOCKED",
        "source_pilot_manifest": str((pilot_dir / "manifest.json").resolve()),
        "source_pilot_manifest_sha256": file_digest(pilot_dir / "manifest.json"),
        "source_pixel_detections_sha256": file_digest(detections_path),
        "source_video_sha256": manifest["source_video"]["sha256"],
        "frames": len(linked_records),
        "detections": sum(len(row["detections"]) for row in linked_records),
        "tracklet_candidates": len(tracklets),
        "potential_cut_frames": [row["frame_index"] for row in transitions if row["potential_cut_before"]],
        "association_scope": "Adjacent 1Hz detections within heuristic shots only; empty frames and potential cuts break tracks.",
        "identity_claim": "NONE; image-space candidate tracklets only.",
        "overlay_paths": overlays,
        "calibration_sidecar": "pixel_calibration_sidecar.json",
        "pitch_adapter": "BLOCKED",
        "no_new_downloads": True,
        "detector_rerun": False,
    })
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot = args.pilot if args.pilot.is_absolute() else ROOT / args.pilot
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(build_extension(pilot.resolve(), output.resolve()), indent=2))


if __name__ == "__main__":
    main()
