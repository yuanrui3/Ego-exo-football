"""Bounded genuine pixel detections; no annotation coordinates or unverified projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from src.backbone.external import ExternalObservationSource, VERSION
from src.trajectory_dataset import file_digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def run(output: Path):
    from ultralytics import YOLO, __version__
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Preserve the completed video pilot; use a new directory")
    source = ROOT / "data" / "video_review_v1_2" / "videos" / "117092" / "117092_panorama_1st_half.mp4"
    weights = ROOT / "outputs" / "hidden_defender_20261019" / "video" / "weights" / "yolo11n-pose.pt"
    diagnostic = ROOT / "outputs" / "hidden_defender_20261019" / "video" / "alignment_smoke" / "field_calibration.json"
    if not weights.is_file() or not source.is_file():
        raise FileNotFoundError("Pilot requires existing local video and existing YOLO nano pose weights; downloads forbidden")
    cv2.setNumThreads(4)
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 4])
    if not capture.isOpened():
        raise RuntimeError("Existing local video could not be opened")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video has no valid frame rate")
    records = []
    try:
        for second in range(12):
            if time.perf_counter() - started > 20 * 60:
                raise TimeoutError("Video pilot time budget exceeded; no alternate architecture attempted")
            frame_index = round((60 + second) * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            valid, image = capture.read()
            if not valid:
                raise RuntimeError(f"Could not decode actual video frame {frame_index}")
            height, width = image.shape[:2]
            tiles, offsets = [], []
            for column in range(4):
                left = max(0, int(column * width / 4) - 64)
                right = min(width, int((column + 1) * width / 4) + 64)
                tiles.append(image[:, left:right])
                offsets.append(left)
            predictions = model.predict(tiles, imgsz=1280, conf=.15, device="cpu",
                                        verbose=False, max_det=100, save=False)
            detections = []
            for tile, prediction, offset in zip(range(4), predictions, offsets):
                for box, confidence in zip(prediction.boxes.xyxy.cpu().numpy(),
                                           prediction.boxes.conf.cpu().numpy()):
                    xyxy = box.astype(float).tolist()
                    xyxy[0] += offset
                    xyxy[2] += offset
                    detections.append({"bbox_xyxy_px": xyxy, "detector_confidence": float(confidence), "tile": tile})
            # Tile overlaps may duplicate boxes: deterministic image-space NMS, no identity assignment.
            if detections:
                boxes = [[d["bbox_xyxy_px"][0], d["bbox_xyxy_px"][1],
                          d["bbox_xyxy_px"][2] - d["bbox_xyxy_px"][0],
                          d["bbox_xyxy_px"][3] - d["bbox_xyxy_px"][1]] for d in detections]
                keep = cv2.dnn.NMSBoxes(boxes, [d["detector_confidence"] for d in detections], .15, .5)
                detections = [detections[int(i)] for i in np.asarray(keep).reshape(-1)]
            for detection in detections:
                x1, y1, x2, y2 = map(round, detection["bbox_xyxy_px"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 210, 255), 2)
            cv2.putText(image, f"PIXEL DETECTIONS ONLY | frame {frame_index} | no pitch/identity/team claims",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 210, 255), 2)
            if second in (0, 5, 11):
                overlay = output / f"overlay_{frame_index}.jpg"
                shown = cv2.resize(image, (2048, round(height * 2048 / width)))
                if not cv2.imwrite(str(overlay), shown):
                    raise RuntimeError(f"Could not save diagnostic overlay {overlay}")
            records.append({
                "record_type": "pixel_observations", "frame_index": frame_index,
                "time_from_video_start_s": frame_index / fps,
                "decoder_next_frame_index": capture.get(cv2.CAP_PROP_POS_FRAMES),
                "decoder_position_ms": capture.get(cv2.CAP_PROP_POS_MSEC),
                "dimensions_px": [width, height], "detections": detections,
                "track_ids": None, "team_assignments": "UNKNOWN", "pitch_positions": None,
                "calibrated_position_accuracy": None,
            })
            print(f"Pixel pilot {second + 1}/12: {len(detections)} detections", flush=True)
    finally:
        capture.release()
    (output / "pixel_observations.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records), encoding="utf-8")
    unsupported = output / "unsupported_pixels.jsonl"
    unsupported.write_text(json.dumps({
        "record_type": "metadata", "schema_version": VERSION,
        "coordinates": {"units": "pixels", "system": "image_top_left"},
    }) + "\n", encoding="utf-8")
    try:
        ExternalObservationSource(unsupported)
    except ValueError as error:
        rejection = str(error)
    else:
        raise AssertionError("Pitch backbone incorrectly accepted pixel-only observations")
    write_json(output / "manifest.json", {
        "status": "PIXEL_ONLY_PITCH_ADAPTER_BLOCKED", "execution_seconds": time.perf_counter() - started,
        "source_video": {"path": str(source), "sha256": file_digest(source), "fps": fps,
                         "selection": "Fixed 60..72s interval, 12 image samples at 1Hz; no annotations used by detector",
                         "sampled_times_s": [row["time_from_video_start_s"] for row in records]},
        "detector": {"implementation": "ultralytics", "version": __version__, "opencv_version": cv2.__version__,
                     "weights": str(weights), "sha256": file_digest(weights), "license": "AGPL-3.0 upstream terms",
                     "device": "cpu", "imgsz": 1280, "tile_columns": 4, "confidence_threshold": .15,
                     "tile_overlap_px": 64, "nms_iou": .5, "new_downloads": False},
        "pixel_observations": {"status": "OBTAINED", "frames": len(records),
                               "detections": sum(len(row["detections"]) for row in records)},
        "camera_to_pitch_projection": {"status": "BLOCKED", "reason":
            "No accepted compatible per-camera/seam pixel-to-pitch mapping; exploratory panorama thirds are not physical cameras"},
        "projection_and_sync_independently_validated": {"status": "NOT_VALIDATED", "reason":
            "Decoded video ordinal is not independent synchronization; no validated detector ground-foot correspondence"},
        "other_blockers": ["No source-local identity continuity across frames/cuts", "Team and goalkeeper roles UNKNOWN"],
        "prior_calibration_evidence": {"path": str(diagnostic), "sha256": file_digest(diagnostic),
                                      "content": json.loads(diagnostic.read_text())},
        "backbone_rejection": {"status": "PASSED", "error": rejection},
        "no_gsr_substitution": True, "no_new_homography": True, "no_pitch_observations_fed_to_backbone": True,
    })
    print(output / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
