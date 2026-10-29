"""Fit a separate video player-footpoint detector from verified fitting annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import ijson
import numpy as np

from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "overnight_video_player_20261028"
VIDEO_PATH = ROOT / "data" / "video_review_v1_2" / "videos" / "117093" / "117093_panorama_1st_half.mp4"
ANNOTATION_PATH = ROOT / "data" / "117093_1st.json"
IMAGE_ID_FRAME_ZERO = 3_903_000_001
FRAME_WIDTH = 4096
FRAME_HEIGHT = 1080
NATIVE_FPS = 25
SAMPLE_STRIDE = 5
TILE_COLUMNS = 4
TILE_OVERLAP_PX = 64

# These are fitting and validation intervals only. Final evaluation starts at 1740 s.
TRAIN_FRAME_RANGE = (1500 * NATIVE_FPS, 1620 * NATIVE_FPS)
VALIDATION_FRAME_RANGE = (1620 * NATIVE_FPS, 1680 * NATIVE_FPS)


def _read_annotation_windows(windows: dict[str, tuple[int, int]]) -> dict[int, list[dict]]:
    """Read only tracked player/goalkeeper annotations in the requested sample windows."""
    last_frame = -1
    annotations: dict[int, list[dict]] = {}
    max_end = max(end for _, end in windows.values())
    with ANNOTATION_PATH.open("rb") as stream:
        for item in ijson.items(stream, "annotations.item"):
            image_id = item.get("image_id")
            try:
                frame_index = int(image_id) - IMAGE_ID_FRAME_ZERO
            except (TypeError, ValueError):
                continue
            if frame_index < last_frame:
                raise ValueError("Annotation image IDs are not in chronological order")
            last_frame = frame_index
            if frame_index >= max_end:
                break
            if frame_index % SAMPLE_STRIDE:
                continue
            if not any(start <= frame_index < end for start, end in windows.values()):
                continue
            if item.get("category_id") not in (1, 2) or item.get("track_id") is None:
                continue
            box = item.get("bbox_image")
            if not isinstance(box, dict):
                continue
            x, y, width, height = (float(box[key]) for key in ("x", "y", "w", "h"))
            if (width <= 0 or height <= 0 or x < 0 or y < 0
                    or x + width > FRAME_WIDTH or y + height > FRAME_HEIGHT):
                continue
            attributes = item.get("attributes") or {}
            pitch = item.get("bbox_pitch") or {}
            point = None
            if "x_bottom_middle" in pitch and "y_bottom_middle" in pitch:
                point = (
                    float(pitch["x_bottom_middle"]),
                    float(pitch["y_bottom_middle"]),
                )
            annotations.setdefault(frame_index, []).append({
                "bbox_xywh_px": [x, y, width, height],
                "footpoint_xy_px": [x + width / 2.0, y + height],
                "track_id": str(item["track_id"]),
                "player_id": str(attributes.get("player_id", "")),
                "team": str(attributes.get("team", "UNKNOWN")),
                "role": str(attributes.get("role", "UNKNOWN")),
                "pitch_xy_centered_m": list(point) if point is not None else None,
            })
    return annotations


def _tiles(image):
    width = image.shape[1]
    tiles, offsets = [], []
    for column in range(TILE_COLUMNS):
        left = max(0, int(column * width / TILE_COLUMNS) - TILE_OVERLAP_PX)
        right = min(
            width,
            int((column + 1) * width / TILE_COLUMNS) + TILE_OVERLAP_PX,
        )
        tiles.append(image[:, left:right])
        offsets.append(left)
    return tiles, offsets


def detect_native(
    model,
    image,
    *,
    confidence: float = 0.05,
    device: int | str = 0,
) -> list[dict]:
    """Run tiled person detection and restore boxes to native panorama coordinates."""
    tiles, offsets = _tiles(image)
    predictions = model.predict(
        tiles, imgsz=1280, conf=confidence, device=device, verbose=False, max_det=100,
    )
    detections = []
    for prediction, offset in zip(predictions, offsets, strict=True):
        boxes = prediction.boxes
        for box, score, class_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            strict=True,
        ):
            if int(class_id) != 0:
                continue
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "bbox_xyxy_px": [x1 + offset, y1, x2 + offset, y2],
                "detector_confidence": float(score),
                "class_id": 0,
                "class_name": "person",
            })
    if not detections:
        return []
    boxes_xywh = [
        [
            item["bbox_xyxy_px"][0],
            item["bbox_xyxy_px"][1],
            item["bbox_xyxy_px"][2] - item["bbox_xyxy_px"][0],
            item["bbox_xyxy_px"][3] - item["bbox_xyxy_px"][1],
        ]
        for item in detections
    ]
    keep = cv2.dnn.NMSBoxes(
        boxes_xywh,
        [item["detector_confidence"] for item in detections],
        confidence,
        0.5,
    )
    indices = np.asarray(keep).reshape(-1)
    return [detections[int(index)] for index in indices]


def _tile_for_center(x: float) -> int:
    return min(TILE_COLUMNS - 1, max(0, int(x / (FRAME_WIDTH / TILE_COLUMNS))))


def _write_yolo_labels(
    label_path: Path, tile_index: int, tile_left: int, tile_width: int,
    frame_annotations: list[dict],
) -> int:
    rows = []
    core_width = FRAME_WIDTH / TILE_COLUMNS
    core_left = tile_index * core_width
    core_right = (tile_index + 1) * core_width
    for annotation in frame_annotations:
        x, y, width, height = annotation["bbox_xywh_px"]
        center_x = x + width / 2.0
        if _tile_for_center(center_x) != tile_index:
            continue
        if not (core_left <= center_x < core_right or (
                tile_index == TILE_COLUMNS - 1 and center_x == FRAME_WIDTH)):
            continue
        x1 = max(0.0, x - tile_left)
        x2 = min(float(tile_width), x + width - tile_left)
        y1 = max(0.0, y)
        y2 = min(float(FRAME_HEIGHT), y + height)
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            f"0 {(x1 + x2) / (2 * tile_width):.8f} "
            f"{(y1 + y2) / (2 * FRAME_HEIGHT):.8f} "
            f"{(x2 - x1) / tile_width:.8f} "
            f"{(y2 - y1) / FRAME_HEIGHT:.8f}"
        )
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def prepare(output: Path) -> dict:
    """Create a small tiled YOLO dataset without modifying source annotations."""
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prepared training data: {output}")
    if not VIDEO_PATH.is_file() or not ANNOTATION_PATH.is_file():
        raise FileNotFoundError("The configured local video or verified annotation file is missing")
    if file_digest(VIDEO_PATH) != (
            "f50dfebc70743439a8c778abe06953885a50be8b47e3c9d3d1ce8d533fe315c8"):
        raise ValueError("The configured panorama differs from the audited source video")
    output.mkdir(parents=True, exist_ok=True)
    windows = {
        "train": TRAIN_FRAME_RANGE,
        "val": VALIDATION_FRAME_RANGE,
    }
    labels = _read_annotation_windows(windows)
    if not labels:
        raise ValueError("No tracked player labels were found in the fitting intervals")

    image_dirs = {split: output / "images" / split for split in windows}
    label_dirs = {split: output / "labels" / split for split in windows}
    for folder in (*image_dirs.values(), *label_dirs.values()):
        folder.mkdir(parents=True)

    capture = cv2.VideoCapture(str(VIDEO_PATH), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {VIDEO_PATH}")
    start = TRAIN_FRAME_RANGE[0]
    end = VALIDATION_FRAME_RANGE[1]
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    image_counts = {split: 0 for split in windows}
    object_counts = {split: 0 for split in windows}
    sampled_frames = {split: 0 for split in windows}
    started = time.monotonic()
    try:
        for frame_index in range(start, end):
            valid, image = capture.read()
            if not valid:
                raise RuntimeError(f"Could not decode source frame {frame_index}")
            if frame_index % SAMPLE_STRIDE:
                continue
            split = next(
                (name for name, (left, right) in windows.items()
                 if left <= frame_index < right),
                None,
            )
            if split is None:
                continue
            tiles, offsets = _tiles(image)
            sampled_frames[split] += 1
            frame_annotations = labels.get(frame_index, [])
            for tile_index, (tile, left) in enumerate(zip(tiles, offsets, strict=True)):
                stem = f"frame_{frame_index:06d}_tile_{tile_index}"
                image_path = image_dirs[split] / f"{stem}.jpg"
                if not cv2.imwrite(
                    str(image_path), tile, [cv2.IMWRITE_JPEG_QUALITY, 92],
                ):
                    raise RuntimeError(f"Could not write training tile: {image_path}")
                label_path = label_dirs[split] / f"{stem}.txt"
                object_counts[split] += _write_yolo_labels(
                    label_path, tile_index, left, tile.shape[1], frame_annotations,
                )
                image_counts[split] += 1
    finally:
        capture.release()

    expected_frames = {
        name: (right - left) // SAMPLE_STRIDE
        for name, (left, right) in windows.items()
    }
    if sampled_frames != expected_frames:
        raise RuntimeError(f"Unexpected sampling coverage: {sampled_frames} != {expected_frames}")

    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: player_footpoint\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "football.video-player-footpoint-training-data.1",
        "status": "FITTING_AND_DEVELOPMENT_VALIDATION_ONLY",
        "source_video": str(VIDEO_PATH.relative_to(ROOT)),
        "source_video_sha256": file_digest(VIDEO_PATH),
        "annotation_source": str(ANNOTATION_PATH.relative_to(ROOT)),
        "annotation_source_sha256": file_digest(ANNOTATION_PATH),
        "timebase": {
            "native_fps": NATIVE_FPS,
            "sampling_stride_frames": SAMPLE_STRIDE,
            "sample_rate_hz": NATIVE_FPS / SAMPLE_STRIDE,
            "train_seconds": [1500, 1620],
            "validation_seconds": [1620, 1680],
            "final_evaluation_seconds": [1740, 1800],
            "second_segment_seconds": [1800, 1860],
        },
        "label_contract": (
            "One class collapses annotated player and goalkeeper categories. "
            "bbox_image is used as the annotated ground-contact localization box, "
            "not a full-body detection box. bbox_pitch and team/role labels are not "
            "included in YOLO training targets."
        ),
        "image_preprocessing": {
            "native_dimensions_px": [FRAME_WIDTH, FRAME_HEIGHT],
            "tile_columns": TILE_COLUMNS,
            "tile_overlap_px": TILE_OVERLAP_PX,
            "tile_offsets_restored_at_inference": True,
            "source_resize_or_crop": False,
            "image_format": "JPEG quality 92",
        },
        "dataset_yaml": str(dataset_yaml.relative_to(ROOT)),
        "sampled_frames": sampled_frames,
        "tile_images": image_counts,
        "player_footpoint_labels": object_counts,
        "preparation_seconds": time.monotonic() - started,
        "evaluation_annotations_are_not_materialized": True,
    }
    write_json(output / "dataset_provenance.json", metadata)
    return metadata


def train(
    dataset_yaml: Path,
    output: Path,
    *,
    epochs: int = 30,
    batch: int = 8,
    image_size: int = 1280,
) -> dict:
    """Fine-tune a separate YOLO11n player-footpoint detector."""
    from ultralytics import YOLO, __version__ as ultralytics_version

    dataset_yaml, output = dataset_yaml.resolve(), output.resolve()
    provenance_path = dataset_yaml.parent / "dataset_provenance.json"
    if not dataset_yaml.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Prepare the fitting-only dataset before training")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a detector training run: {output}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The configured overnight detector training requires CUDA")
    output.mkdir(parents=True, exist_ok=True)
    base_weights = ROOT / "yolo11n.pt"
    if not base_weights.is_file():
        base_weights = Path("yolo11n.pt").resolve()
    if not base_weights.is_file():
        raise FileNotFoundError("Official Ultralytics yolo11n.pt weights are unavailable")
    model = YOLO(str(base_weights))
    started = time.monotonic()
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        patience=8,
        imgsz=image_size,
        batch=batch,
        workers=4,
        device=0,
        project=str(output),
        name="yolo11n_footpoint",
        exist_ok=False,
        seed=42,
        deterministic=True,
        close_mosaic=4,
        cache=False,
        amp=True,
        plots=False,
        verbose=True,
    )
    run_dir = output / "yolo11n_footpoint"
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise RuntimeError("Training returned without producing best.pt")
    summary = {
        "schema_version": "football.video-player-footpoint-detector-training.1",
        "status": "EXPERIMENTAL_SEPARATE_FROM_BACKBONE",
        "base_weights": str(base_weights.relative_to(ROOT)),
        "base_weights_sha256": file_digest(base_weights),
        "best_weights": str(best_weights.relative_to(ROOT)),
        "best_weights_sha256": file_digest(best_weights),
        "dataset_yaml": str(dataset_yaml.relative_to(ROOT)),
        "dataset_provenance_sha256": file_digest(provenance_path),
        "ultralytics_version": ultralytics_version,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "hyperparameters": {
            "epochs_requested": epochs,
            "batch": batch,
            "image_size": image_size,
            "device": 0,
            "seed": 42,
            "workers": 4,
        },
        "training_seconds": time.monotonic() - started,
        "training_results_csv": str((run_dir / "results.csv").relative_to(ROOT)),
    }
    write_json(output / "training_provenance.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--output", type=Path,
        default=OUTPUT_ROOT / "training_data",
    )
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--dataset-yaml", type=Path,
        default=OUTPUT_ROOT / "training_data" / "dataset.yaml",
    )
    train_parser.add_argument(
        "--output", type=Path,
        default=OUTPUT_ROOT / "detector_training",
    )
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--image-size", type=int, default=1280)
    args = parser.parse_args()
    result = (
        prepare(args.output) if args.command == "prepare"
        else train(
            args.dataset_yaml, args.output, epochs=args.epochs,
            batch=args.batch, image_size=args.image_size,
        )
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
