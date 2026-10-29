"""Run bounded detection and conservative image-space tracklets on local footage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from src.backbone.video_tracklets import CUT_THRESHOLD, associate_tracklets
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]


def _tiles(image: np.ndarray, columns: int, overlap: int) -> tuple[list[np.ndarray], list[int]]:
    width = image.shape[1]
    tiles, offsets = [], []
    for column in range(columns):
        left = max(0, int(column * width / columns) - overlap)
        right = min(width, int((column + 1) * width / columns) + overlap)
        tiles.append(image[:, left:right])
        offsets.append(left)
    return tiles, offsets


def _detect_frame(model, image: np.ndarray, *, columns: int, overlap: int,
                  confidence: float, device: str) -> list[dict]:
    tiles, offsets = _tiles(image, columns, overlap)
    predictions = model.predict(
        tiles, imgsz=1280, conf=confidence, device=device, verbose=False, max_det=100, save=False,
    )
    detections = []
    for tile_index, (prediction, offset) in enumerate(zip(predictions, offsets, strict=True)):
        for box, score, class_id in zip(
            prediction.boxes.xyxy.cpu().numpy(),
            prediction.boxes.conf.cpu().numpy(),
            prediction.boxes.cls.cpu().numpy(),
            strict=True,
        ):
            xyxy = box.astype(float).tolist()
            xyxy[0] += offset
            xyxy[2] += offset
            class_name = model.names.get(int(class_id), str(int(class_id)))
            detections.append({
                "bbox_xyxy_px": xyxy,
                "detector_confidence": float(score),
                "class_id": int(class_id),
                "class_name": str(class_name),
                "tile": tile_index,
            })
    if not detections:
        return []
    boxes = [[d["bbox_xyxy_px"][0], d["bbox_xyxy_px"][1],
              d["bbox_xyxy_px"][2] - d["bbox_xyxy_px"][0],
              d["bbox_xyxy_px"][3] - d["bbox_xyxy_px"][1]] for d in detections]
    keep = cv2.dnn.NMSBoxes(
        boxes, [d["detector_confidence"] for d in detections], confidence, 0.5,
    )
    return [detections[int(index)] for index in np.asarray(keep).reshape(-1)]


def _annotate(image: np.ndarray, detections: list[dict], target: Path, frame_index: int) -> None:
    shown = image.copy()
    for item in detections:
        x1, y1, x2, y2 = map(round, item["bbox_xyxy_px"])
        cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 210, 255), 2)
    cv2.putText(
        shown, f"PIXEL DETECTIONS ONLY | frame {frame_index} | teams/pitch positions UNKNOWN",
        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2,
    )
    height, width = shown.shape[:2]
    shown = cv2.resize(shown, (2048, round(height * 2048 / width)))
    if not cv2.imwrite(str(target), shown):
        raise RuntimeError(f"Could not write video diagnostic overlay: {target}")


def run(config_path: Path, output: Path, *, root: Path = ROOT) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "football.backbone-integration.0.5":
        raise ValueError("Video ingestion requires the v0.5 integration configuration")
    settings = config["video_ingestion"]
    clips = settings["clips"]
    if len(clips) != 3 or len({clip["match"] for clip in clips}) < 2:
        raise ValueError("Video workload requires three configured clips from at least two matches")
    fps_target = int(settings["target_fps"])
    duration_s = int(settings["clip_seconds"])
    if fps_target != 5 or duration_s != 60:
        raise ValueError("The bounded v0.5 video workload is fixed at 60 seconds and 5 FPS")
    expected_frames = fps_target * duration_s
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing video workload: {output}")
    output.mkdir(parents=True)
    video_root = root / config["source_paths"]["video_root"]
    weight_path = root / config["source_paths"]["video_weights"]
    if not weight_path.is_file():
        raise FileNotFoundError(f"Existing configured detector weights are unavailable: {weight_path}")
    from ultralytics import YOLO, __version__ as ultralytics_version

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv2.setNumThreads(4)
    model = YOLO(str(weight_path))
    started = time.monotonic()
    clip_records = []
    total_frames, total_decoded, total_detections = 0, 0, 0
    for clip in clips:
        match, half = str(clip["match"]), str(clip["half"])
        if half not in ("1st", "2nd"):
            raise ValueError(f"Invalid video half: {half}")
        video_path = video_root / match / f"{match}_panorama_{half}_half.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Configured local video is missing: {video_path}")
        capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open configured local video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start_s = float(clip["start_s"])
        if not np.isfinite(fps) or fps <= 0 or start_s < 0:
            capture.release()
            raise ValueError(f"Invalid frame-rate or clip start metadata for {video_path}")
        indices = [round((start_s + i / fps_target) * fps) for i in range(expected_frames)]
        if indices[0] < 0 or indices[-1] >= frame_count:
            capture.release()
            raise ValueError(f"Configured 60-second clip exceeds local video duration: {video_path}")

        clip_id = f"{match}_{half}_{int(start_s):05d}s"
        clip_dir = output / clip_id
        clip_dir.mkdir()
        records, image_histograms = [], []
        decoded = 0
        try:
            for sample_index, frame_index in enumerate(indices):
                if time.monotonic() - started > float(settings["max_runtime_minutes"]) * 60:
                    raise TimeoutError("Configured v0.5 video workload exceeded its runtime budget")
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                valid, image = capture.read()
                if not valid:
                    raise RuntimeError(f"Could not decode configured video frame {frame_index}")
                decoded += 1
                image_histograms.append(_histogram(image))
                detections = _detect_frame(
                    model, image,
                    columns=int(settings["tile_columns"]),
                    overlap=int(settings["tile_overlap_px"]),
                    confidence=float(settings["detector_confidence"]),
                    device=device,
                )
                distance = None if sample_index == 0 else float(cv2.compareHist(
                    image_histograms[-2], image_histograms[-1], cv2.HISTCMP_BHATTACHARYYA,
                ))
                records.append({
                    "record_type": "pixel_observations",
                    "frame_index": frame_index,
                    "time_from_video_start_s": frame_index / fps,
                    "dimensions_px": [int(image.shape[1]), int(image.shape[0])],
                    "detections": detections,
                    "potential_cut_before": bool(distance is not None and distance >= CUT_THRESHOLD),
                    "histogram_distance_from_previous_sample": distance,
                    "track_ids": None,
                    "team_assignments": "UNKNOWN",
                    "pitch_positions": None,
                    "calibrated_position_accuracy": None,
                })
                if sample_index in (0, expected_frames // 2, expected_frames - 1):
                    _annotate(image, detections, clip_dir / f"overlay_{sample_index:03d}.jpg", frame_index)
                total_detections += len(detections)
                if (sample_index + 1) % 25 == 0:
                    write_json(output / "progress.json", {
                        "status": "RUNNING",
                        "active_clip": clip_id,
                        "clip_inference_frames": sample_index + 1,
                        "total_inference_frames": total_frames + sample_index + 1,
                        "target_inference_frames": len(clips) * expected_frames,
                        "actual_decoded_frames": total_decoded + decoded,
                        "detections": total_detections,
                    })
        finally:
            capture.release()
        cut_flags = [row["potential_cut_before"] for row in records]
        linked, tracklets = associate_tracklets(records, cut_flags)
        observations_path = clip_dir / "pixel_tracklets.jsonl"
        observations_path.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in linked),
            encoding="utf-8",
        )
        write_json(clip_dir / "tracklet_summary.json", {
            "status": "UNVERIFIED_IMAGE_SPACE_CANDIDATES",
            "tracklets": tracklets,
            "pitch_projection": "UNSUPPORTED",
            "identity_claim": "NONE",
        })
        clip_records.append({
            "clip_id": clip_id,
            "match": match,
            "half": half,
            "start_s": start_s,
            "duration_s": duration_s,
            "target_fps": fps_target,
            "source_video": str(video_path.relative_to(root)),
            "source_video_sha256": file_digest(video_path),
            "source_video_fps": fps,
            "video_frame_count": frame_count,
            "decoded_inference_frames": decoded,
            "inference_frames": len(records),
            "detection_count": sum(len(row["detections"]) for row in linked),
            "tracklet_count": len(tracklets),
            "tracklet_link_status": "UNVERIFIED_ADJACENT_SAMPLE_MATCH",
            "observations_path": str(observations_path.relative_to(output)),
            "observations_sha256": file_digest(observations_path),
        })
        total_frames += len(records)
        total_decoded += decoded
    if total_frames != len(clips) * expected_frames or total_frames < 900:
        raise RuntimeError(f"Expected at least 900 actual inferred frames, received {total_frames}")
    manifest = {
        "schema_version": "football.video-ingestion.0.5",
        "status": "PIXEL_DETECTION_AND_TRACKLET_CANDIDATES_COMPLETE",
        "execution_seconds": time.monotonic() - started,
        "source_matches": sorted({clip["match"] for clip in clips}),
        "detector": {
            "implementation": "ultralytics",
            "version": ultralytics_version,
            "opencv_version": cv2.__version__,
            "weights": str(weight_path.relative_to(root)),
            "weights_sha256": file_digest(weight_path),
            "device": device,
            "imgsz": 1280,
            "tile_columns": int(settings["tile_columns"]),
            "tile_overlap_px": int(settings["tile_overlap_px"]),
            "confidence_threshold": float(settings["detector_confidence"]),
            "new_downloads": False,
        },
        "clips": clip_records,
        "totals": {
            "decoded_inference_frames": total_decoded,
            "inference_frames": total_frames,
            "detections": total_detections,
            "tracklet_candidates": sum(row["tracklet_count"] for row in clip_records),
        },
        "limitations": [
            "Pixel coordinates only; no camera-to-pitch transform or physical positions",
            "Image-space links are unverified and do not establish player identity",
            "Team and goalkeeper roles remain unknown",
            "No raw-video output was used as a tracking-cache replacement",
        ],
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "progress.json", {"status": manifest["status"], "totals": manifest["totals"]})
    return manifest


def _histogram(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (320, 160), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/backbone_v0_5.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else args.root / args.config
    output = args.output if args.output.is_absolute() else args.root / args.output
    result = run(config.resolve(), output.resolve(), root=args.root.resolve())
    print(json.dumps({"status": result["status"], "totals": result["totals"],
                      "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
