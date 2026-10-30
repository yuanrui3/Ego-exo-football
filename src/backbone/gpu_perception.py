"""Isolated GPU perception experiment; never changes the frozen backbone."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import torch

from src.backbone.video_player_cross_match import (
    _maximum_matches,
    _read_annotations,
    _read_frame,
    load_protocol,
)
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "outputs" / "video_player_cross_match_20261028_130610"
DEFAULT_OUTPUT = ROOT / "outputs" / "gp_20261029_125507"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_records() -> dict[str, dict]:
    protocol = load_protocol(HISTORICAL / "protocol.json")
    sources = {
        row["id"]: row for name, row in protocol["selection"].items()
        if name in ("development_match", "evaluation_match")
    }
    sources["117093"] = {
        "id": "117093", "half": "1st", "dimensions_px": [4096, 1080],
        "video": "data/video_review_v1_2/videos/117093/117093_panorama_1st_half.mp4",
        "annotation": "data/117093_1st.json", "image_id_frame_zero": 3903000001,
        "annotation_sha256": "ceb740ea03b445e27d45f49e7001795f03dbd95eaebde85304e42fd9d0581b19",
        "video_sha256": "f50dfebc70743439a8c778abe06953885a50be8b47e3c9d3d1ce8d533fe315c8",
    }
    return sources


def tile_bounds(width: int, height: int, scheme: str = "consistent") -> list[tuple[int, int, int, int]]:
    rows = 2 if scheme == "consistent" and height > 1280 else 1
    bounds = []
    for row in range(rows):
        top = max(0, int(row * height / rows) - (96 if rows > 1 else 0))
        bottom = min(height, int((row + 1) * height / rows) + (96 if rows > 1 else 0))
        for column in range(4):
            left = max(0, int(column * width / 4) - 64)
            right = min(width, int((column + 1) * width / 4) + 64)
            bounds.append((left, top, right, bottom))
    return bounds


def tile_labels(truth: list[dict], bounds: tuple[int, int, int, int]) -> tuple[list[list[float]], dict]:
    left, top, right, bottom = bounds
    rows, counts = [], Counter()
    for annotation in truth:
        x, y, width, height = annotation["bbox_xywh_px"]
        foot_x, foot_y = x + width / 2, y + height
        if not (left <= foot_x < right and top <= foot_y <= bottom):
            continue
        x1, y1 = max(x, left), max(y, top)
        x2, y2 = min(x + width, right), min(y + height, bottom)
        if x2 <= x1 or y2 <= y1:
            counts["discarded_nonpositive"] += 1
            continue
        counts["clipped"] += int(x1 != x or y1 != y or x2 != x + width or y2 != y + height)
        rows.append([0, (x1 + x2 - 2 * left) / (2 * (right - left)),
                     (y1 + y2 - 2 * top) / (2 * (bottom - top)),
                     (x2 - x1) / (right - left), (y2 - y1) / (bottom - top)])
    return rows, dict(counts)


def restore_boxes(boxes: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    restored = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    restored[:, [0, 2]] += bounds[0]
    restored[:, [1, 3]] += bounds[1]
    return restored


def historical_data_audit(output: Path) -> dict:
    result = {}
    for name, folder in {
        "source_5hz": ROOT / "outputs/overnight_video_player_20261028/training_data",
        "D_1hz": HISTORICAL / "adaptation_dataset",
        "D_5hz": HISTORICAL / "adaptation_dataset_5hz",
    }.items():
        splits = {}
        for split in (folder / "images").iterdir():
            images = sorted(split.glob("*.jpg"))
            labels = [folder / "labels" / split.name / (p.stem + ".txt") for p in images]
            missing = [str(p) for p in labels if not p.is_file()]
            if missing:
                raise FileNotFoundError(missing[:5])
            counts = [len(p.read_text().splitlines()) for p in labels]
            frames = sorted({int(p.stem.split("_")[1]) for p in images})
            splits[split.name] = {
                "unique_frames": len(frames), "frame_stride": dict(Counter(np.diff(frames).tolist())),
                "tile_images": len(images), "label_rows": sum(counts),
                "empty_labels": sum(n == 0 for n in counts), "missing_labels": len(missing),
                "caches": [{"path": str(p.relative_to(ROOT)), "sha256": file_digest(p)}
                           for p in (folder / "labels").glob("*.cache")],
            }
        result[name] = splits
    result["discrepancy_resolution"] = (
        "Both versions exist. D 1Hz has 120 frames/480 tiles/2640 labels per split; "
        "D 5Hz has 600 frames/2400 tiles/13200 labels. The historical 5286/2640 "
        "validator statement describes the older 1Hz population, not the later 5Hz fit. "
        "Current disk counts are authoritative; historical reports are not rewritten."
    )
    write_json(output / "historical_data_audit.json", result)
    return result


def draw_boxes(image: np.ndarray, boxes: list, color: tuple[int, int, int]) -> None:
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        cv2.circle(image, (round((x1 + x2) / 2), round(y2)), 3, color, -1)


def prepare(output: Path) -> dict:
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.augment import LetterBox
    from ultralytics.data.utils import check_det_dataset

    destination = output / "data_v1"
    if destination.exists():
        raise FileExistsError(destination)
    if shutil.disk_usage(output).free < 15 * 2**30:
        raise RuntimeError("Need at least 15 GiB free before preparing new images")
    historical_data_audit(output)
    destination.mkdir()
    sources = source_records()
    windows = {
        "117093": {"fit": [1500, 1620], "dev": [1620, 1680]},
        "117092": {"fit": [600, 720], "dev": [900, 1020]},
    }
    manifest = {
        "created_utc": now(), "sampling_hz": 5, "native_fps": 25,
        "sources": sources, "windows": windows, "scheme": "consistent",
        "selection_exposure": "117093 development already used historically; 118576 excluded",
        "target": "annotated ground-contact box, bottom-center footpoint, NOT full-body person",
        "overlap_policy": "Every positive-area box with visible bottom-center in a tile is labeled; native NMS removes overlap duplicates.",
        "channel_order": "OpenCV BGR tiles; Ultralytics loader converts BGR to RGB and normalizes uint8 by 255.",
        "splits": {}, "examples": [],
    }
    image_lists = {"fit": [], "dev": []}
    for match_id, splits in windows.items():
        source = sources[match_id]
        frame_sets = {
            split: set(range(start * 25, end * 25, 5))
            for split, (start, end) in splits.items()
        }
        assert not frame_sets["fit"] & frame_sets["dev"]
        truth_by_frame = _read_annotations(source, set.union(*frame_sets.values()))
        video = ROOT / source["video"]
        if file_digest(video) != source["video_sha256"]:
            raise ValueError(f"Video hash changed: {video}")
        source["annotation_sha256"] = file_digest(ROOT / source["annotation"])
        cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        try:
            for split, indices in frame_sets.items():
                image_dir = destination / "images" / f"{match_id}_{split}"
                label_dir = destination / "labels" / f"{match_id}_{split}"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                counts = Counter()
                sizes = {"consistent": [], "historical": []}
                native_frames = []
                selected = set(sorted(indices)[::max(1, len(indices) // 6)][:6])
                previous = None
                digest_rows = []
                for frame_index in sorted(indices):
                    image = _read_frame(cap, frame_index, previous)
                    previous = frame_index
                    if list(image.shape[1::-1]) != source["dimensions_px"]:
                        raise ValueError(f"Native dimensions changed for {match_id}")
                    truth = truth_by_frame.get(frame_index, [])
                    counts["raw_annotations"] += len(truth)
                    bounds_list = tile_bounds(*source["dimensions_px"])
                    retained = Counter()
                    frame_record = {"frame_index": frame_index, "tiles": []}
                    for tile_index, bounds in enumerate(bounds_list):
                        left, top, right, bottom = bounds
                        tile = image[top:bottom, left:right]
                        rows, flags = tile_labels(truth, bounds)
                        counts.update(flags)
                        stem = f"f{frame_index:07d}_t{tile_index}"
                        image_path = image_dir / (stem + ".jpg")
                        label_path = label_dir / (stem + ".txt")
                        if not cv2.imwrite(str(image_path), tile):
                            raise OSError(f"Cannot write {image_path}")
                        label_path.write_text("".join(
                            " ".join(f"{value:.8f}" for value in row) + "\n" for row in rows
                        ), encoding="utf-8")
                        digest_rows.append([label_path.name, file_digest(label_path)])
                        image_lists[split].append(image_path.resolve())
                        counts["tile_images"] += 1
                        counts["parsed_labels"] += len(rows)
                        counts["empty_label_files"] += int(not rows)
                        frame_record["tiles"].append({
                            "path": str(image_path.relative_to(ROOT)), "bounds": list(bounds),
                        })
                        for k, annotation in enumerate(truth):
                            px, py = annotation["footpoint_xy_px"]
                            if left <= px < right and top <= py <= bottom and tile_labels([annotation], bounds)[0]:
                                retained[k] += 1
                        scale = min(1280 / tile.shape[0], 1280 / tile.shape[1])
                        sizes["consistent"].extend([[r[3] * tile.shape[1] * scale,
                                                      r[4] * tile.shape[0] * scale] for r in rows])
                    for bounds in tile_bounds(*source["dimensions_px"], scheme="historical"):
                        rows, _ = tile_labels(truth, bounds)
                        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
                        scale = min(1280 / w, 1280 / h)
                        sizes["historical"].extend([[r[3] * w * scale, r[4] * h * scale] for r in rows])
                    counts["unique_retained_annotations"] += len(retained)
                    counts["overlap_duplicate_labels"] += sum(max(0, n - 1) for n in retained.values())
                    counts["discarded_annotations"] += len(truth) - len(retained)
                    native_frames.append(frame_record)
                    if frame_index in selected:
                        positive = [(i, len(tile_labels(truth, b)[0])) for i, b in enumerate(bounds_list)]
                        ti = max(positive, key=lambda pair: pair[1])[0]
                        bounds = bounds_list[ti]
                        left, top, right, bottom = bounds
                        tile = image[top:bottom, left:right].copy()
                        rows, _ = tile_labels(truth, bounds)
                        model_input = LetterBox(new_shape=(1280, 1280), auto=False)(image=tile)
                        scale = min(1280 / tile.shape[0], 1280 / tile.shape[1])
                        pad_x = round((1280 - round(tile.shape[1] * scale)) / 2 - 0.1)
                        pad_y = round((1280 - round(tile.shape[0] * scale)) / 2 - 0.1)
                        local = np.array([[(r[1] - r[3]/2)*tile.shape[1], (r[2] - r[4]/2)*tile.shape[0],
                                           (r[1] + r[3]/2)*tile.shape[1], (r[2] + r[4]/2)*tile.shape[0]] for r in rows])
                        transformed = local * scale + np.array([pad_x, pad_y, pad_x, pad_y])
                        restored = restore_boxes((transformed - [pad_x, pad_y, pad_x, pad_y]) / scale, bounds)
                        assert np.allclose(restored, restore_boxes(local, bounds), atol=1e-6)
                        native = image.copy()
                        draw_boxes(native, [[a["bbox_xywh_px"][0], a["bbox_xywh_px"][1],
                                             a["bbox_xywh_px"][0]+a["bbox_xywh_px"][2],
                                             a["bbox_xywh_px"][1]+a["bbox_xywh_px"][3]] for a in truth], (0, 255, 0))
                        draw_boxes(native, restored.tolist(), (0, 165, 255))
                        cv2.rectangle(native, (left, top), (right, bottom), (255, 255, 0), 3)
                        draw_boxes(model_input, transformed.tolist(), (0, 255, 0))
                        panel = np.concatenate([cv2.resize(native, (960, 640)), cv2.resize(model_input, (640, 640))], axis=1)
                        cv2.putText(panel, f"{match_id} {split} frame {frame_index}: native / exact letterbox input; orange=restored",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                        example_path = output / "examples" / f"{match_id}_{split}_{frame_index}.jpg"
                        example_path.parent.mkdir(exist_ok=True)
                        if not cv2.imwrite(str(example_path), panel):
                            raise OSError(example_path)
                        manifest["examples"].append(str(example_path.relative_to(ROOT)))
                report = {
                    **dict(counts), "unique_source_frames": len(indices), "frame_stride": 5,
                    "timestamp_spacing_seconds": .2, "first_frame": min(indices), "last_frame": max(indices),
                    "effective_target_wh_px": {
                        scheme: {"p10": np.percentile(values, 10, axis=0).tolist(),
                                 "median": np.median(values, axis=0).tolist(),
                                 "p90": np.percentile(values, 90, axis=0).tolist()}
                        for scheme, values in sizes.items()
                    },
                    "label_hashes": digest_rows, "frames": native_frames,
                }
                manifest["splits"][f"{match_id}_{split}"] = report
        finally:
            cap.release()
    for split, paths in image_lists.items():
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate paths in {split}")
        (destination / f"{split}.txt").write_text(
            "\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8",
        )
    yaml = destination / "dataset.yaml"
    yaml.write_text(
        f"path: {destination.resolve().as_posix()}\ntrain: fit.txt\nval: dev.txt\nnames:\n  0: player_footpoint\n",
        encoding="utf-8",
    )
    write_json(output / "data_manifest_draft.json", manifest)
    return audit_prepared(output)


def audit_prepared(output: Path, manifest: dict | None = None) -> dict:
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    destination = output / "data_v1"
    yaml = destination / "dataset.yaml"
    if manifest is None:
        sources = source_records()
        windows = {"117093": {"fit": [1500, 1620], "dev": [1620, 1680]},
                   "117092": {"fit": [600, 720], "dev": [900, 1020]}}
        manifest = {
            "created_utc": now(), "sources": sources, "windows": windows, "sampling_hz": 5,
            "scheme": "consistent", "splits": {},
            "examples": [str(p.relative_to(ROOT)) for p in sorted((output / "examples").glob("*.jpg"))],
            "target": "Ground-contact localization boxes, not full-body person; bottom-center footpoint",
            "selection_exposure": "117093 dev previously examined. 118576 excluded from fitting/selection.",
            "overlap_policy": "All positive-area boxes with visible bottom-center are labeled in each overlapping tile; native NMS suppresses duplicate predictions.",
            "channel_order": "OpenCV BGR to Ultralytics RGB, divide by 255.",
        }
        for match_id, splits in windows.items():
            source = sources[match_id]
            indices = set.union(*(set(range(a * 25, b * 25, 5)) for a, b in splits.values()))
            truth = _read_annotations(source, indices)
            for split, (start, end) in splits.items():
                frames, counts, sizes = [], Counter(), {"consistent": [], "historical": []}
                label_hashes, image_hashes = [], []
                for index in range(start * 25, end * 25, 5):
                    frame_truth = truth.get(index, [])
                    bounds = tile_bounds(*source["dimensions_px"])
                    frame = {"frame_index": index, "tiles": []}
                    retained = Counter()
                    counts["raw_annotations"] += len(frame_truth)
                    for ti, bound in enumerate(bounds):
                        path = destination / "images" / f"{match_id}_{split}" / f"f{index:07d}_t{ti}.jpg"
                        label = destination / "labels" / f"{match_id}_{split}" / f"f{index:07d}_t{ti}.txt"
                        if not path.is_file() or not label.is_file():
                            raise FileNotFoundError(f"Missing tile/label: {path}")
                        rows, flags = tile_labels(frame_truth, bound)
                        actual = [list(map(float, line.split())) for line in label.read_text().splitlines()]
                        if len(rows) != len(actual) or (rows and not np.allclose(rows, actual, atol=1e-7)):
                            raise ValueError(f"Prepared target mismatch: {label}")
                        counts.update(flags)
                        counts["tile_images"] += 1
                        counts["parsed_labels"] += len(actual)
                        counts["empty_label_files"] += int(not actual)
                        unique = len(np.unique(np.asarray(actual, dtype=np.float32), axis=0)) if actual else 0
                        counts["exact_duplicate_rows"] += len(actual) - unique
                        counts["expected_unique_loader_labels"] += unique
                        l, t, r, b = bound
                        for k, annotation in enumerate(frame_truth):
                            x, y = annotation["footpoint_xy_px"]
                            if l <= x < r and t <= y <= b and tile_labels([annotation], bound)[0]:
                                retained[k] += 1
                        ratio = min(1280 / (r - l), 1280 / (b - t))
                        sizes["consistent"].extend([[row[3] * (r - l) * ratio, row[4] * (b - t) * ratio]
                                                    for row in rows])
                        frame["tiles"].append({"path": str(path.relative_to(ROOT)), "bounds": list(bound)})
                        label_hashes.append([label.name, file_digest(label)])
                        image_hashes.append([path.name, file_digest(path)])
                    for bound in tile_bounds(*source["dimensions_px"], scheme="historical"):
                        rows, _ = tile_labels(frame_truth, bound)
                        w, h = bound[2] - bound[0], bound[3] - bound[1]
                        ratio = min(1280 / w, 1280 / h)
                        sizes["historical"].extend([[row[3] * w * ratio, row[4] * h * ratio] for row in rows])
                    counts["unique_retained_annotations"] += len(retained)
                    counts["overlap_duplicate_labels"] += sum(max(0, n - 1) for n in retained.values())
                    counts["discarded_annotations"] += len(frame_truth) - len(retained)
                    frames.append(frame)
                manifest["splits"][match_id + "_" + split] = {
                    **dict(counts), "unique_source_frames": len(frames), "frame_stride": 5,
                    "timestamp_spacing_seconds": .2, "frames": frames,
                    "label_hashes": label_hashes, "image_hashes": image_hashes,
                    "effective_target_wh_px": {
                        scheme: {"p10": np.percentile(values, 10, axis=0).tolist(),
                                 "median": np.median(values, axis=0).tolist(),
                                 "p90": np.percentile(values, 90, axis=0).tolist()}
                        for scheme, values in sizes.items()
                    },
                }
    data = check_det_dataset(str(yaml.resolve()), autodownload=False)
    config = get_cfg(overrides={"imgsz": 1280, "cache": False, "workers": 0, "mosaic": 0.0})
    for split in ("fit", "dev"):
        dataset = build_yolo_dataset(config, data["train" if split == "fit" else "val"],
                                     batch=8, data=data, mode="val", rect=False, stride=32)
        delivered = sum(len(row["cls"]) for row in dataset.labels)
        expected = sum(row.get("expected_unique_loader_labels", row["parsed_labels"])
                       for key, row in manifest["splits"].items()
                       if key.endswith("_" + split))
        if delivered != expected:
            raise ValueError(f"Loader removed labels: {split}: {delivered} != {expected}")
        manifest[f"{split}_loader"] = {
            "images": len(dataset), "labels": delivered, "unique_disk_labels": expected,
            "raw_disk_label_rows": sum(row["parsed_labels"] for key, row in manifest["splits"].items()
                                      if key.endswith("_" + split)),
            "list_sha256": file_digest(destination / f"{split}.txt"),
        }
    manifest["dataset_yaml_sha256"] = file_digest(yaml)
    write_json(output / "data_manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key.endswith("_loader")}), flush=True)
    return manifest


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED: no CPU fallback")
    torch.cuda.set_device(0)
    torch.cuda.synchronize()


def enable_detector_gradients(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".dfl." not in name)


def merged_detections(model, tiles: list[np.ndarray], bounds: list, device=0) -> list[dict]:
    predictions = model.predict(tiles, imgsz=1280, conf=.20, iou=.5, max_det=100,
                                device=device, verbose=False, rect=False)
    result = []
    for prediction, offset in zip(predictions, bounds, strict=True):
        boxes = restore_boxes(prediction.boxes.xyxy.cpu().numpy(), offset)
        for box, score, cls in zip(boxes, prediction.boxes.conf.cpu().tolist(),
                                   prediction.boxes.cls.cpu().tolist(), strict=True):
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = box.tolist()
            result.append({"bbox_xyxy_px": [x1, y1, x2, y2],
                           "footpoint_xy_px": [(x1 + x2) / 2, y2],
                           "detector_confidence": score, "class_id": 0})
    if not result:
        return []
    indices = cv2.dnn.NMSBoxes(
        [[r["bbox_xyxy_px"][0], r["bbox_xyxy_px"][1],
          r["bbox_xyxy_px"][2] - r["bbox_xyxy_px"][0],
          r["bbox_xyxy_px"][3] - r["bbox_xyxy_px"][1]] for r in result],
        [r["detector_confidence"] for r in result], .20, .5,
    )
    return [result[int(index)] for index in np.asarray(indices).reshape(-1)]


def aggregate_scores(frames: list[dict], labels: dict, width: int, sampling_hz: float = 5.) -> dict:
    if not math.isfinite(sampling_hz) or sampling_hz <= 0:
        raise ValueError("Sampling frequency must be positive")
    scores = {}
    total_truth = sum(len(labels.get(f["frame_index"], [])) for f in frames)
    total_pred = sum(len(f["detections"]) for f in frames)
    for tolerance in (20, 40):
        matched, duplicates, transitions, switches = 0, 0, 0, 0
        previous, last_matched, identity_run = {}, {}, {}
        identity_tracks = {}
        gap_returns, same_track_returns = 0, 0
        runs, run = [], Counter()
        history_opportunities, complete = 0, 0
        matched_history_complete = 0
        observed_counts = Counter()
        for sample_index, frame in enumerate(frames):
            detections, truth = frame["detections"], labels.get(frame["frame_index"], [])
            pairs = _maximum_matches(detections, truth, tolerance * width / 4096)
            matched += len(pairs)
            paired_pred, paired_gt = {pi for pi, _, _ in pairs}, {gi for _, gi, _ in pairs}
            for pi, detection in enumerate(detections):
                if pi not in paired_pred and any(
                    math.dist(detection["footpoint_xy_px"], truth[gi]["footpoint_xy_px"]) <= tolerance * width / 4096
                    for gi in paired_gt
                ):
                    duplicates += 1
            current, current_identity_run = {}, {}
            for pi, gi, _ in pairs:
                track = detections[pi].get("track")
                if track is None:
                    continue
                identity = truth[gi]["track_id"]
                current[identity] = track
                current_identity_run[identity] = (
                    identity_run.get(identity, 0) + 1 if previous.get(identity) == track else 1
                )
                matched_history_complete += int(current_identity_run[identity] >= 11)
                identity_tracks.setdefault(identity, set()).add(track)
                if identity in last_matched and sample_index - last_matched[identity][0] > 1:
                    gap_returns += 1
                    same_track_returns += int(last_matched[identity][1] == track)
                last_matched[identity] = (sample_index, track)
                if identity in previous:
                    transitions += 1
                    switches += previous[identity] != track
            previous = current
            identity_run = current_identity_run
            present = {d["track"] for d in detections if "track" in d}
            for track in set(run) - present:
                runs.append(run.pop(track))
            for track in present:
                run[track] += 1
                observed_counts[track] += 1
                history_opportunities += 1
                complete += int(run[track] >= 11)
        runs.extend(run.values())
        precision = matched / total_pred if total_pred else 0
        recall = matched / total_truth if total_truth else 0
        scores[str(tolerance)] = {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "matches": matched, "truth_instances": total_truth, "detections": total_pred,
            "misses": total_truth - matched, "false_positives_including_duplicates": total_pred - matched,
            "duplicates": duplicates, "frames": len(frames),
            "matched_adjacent_transitions": transitions, "identity_switches": switches,
            "switch_rate": switches / transitions if transitions else None,
            "matched_returns_after_gap": gap_returns,
            "same_runtime_track_returns_after_gap": same_track_returns,
            "continuity_after_matched_gap_fraction": same_track_returns / gap_returns if gap_returns else None,
            "fragmentation_extra_tracks_per_identity": sum(max(0, len(t) - 1) for t in identity_tracks.values()),
            "continuous_track_samples_median": float(np.median(runs)) if runs else None,
            "continuous_track_runs": len(runs),
            "continuous_track_length_samples_histogram": dict(sorted(Counter(runs).items())),
            "continuous_track_duration_seconds_median": float(np.median(runs) - 1) / sampling_hz if runs else None,
            "covered_sample_time_seconds_median": float(np.median(runs)) / sampling_hz if runs else None,
            "sampling_hz": sampling_hz,
            "backbone_history_compatible_sampling": sampling_hz == 5,
            "eleven_sample_span_seconds": 10 / sampling_hz,
            "complete_11_sample_history_observations": complete,
            "all_track_observation_opportunities": history_opportunities,
            "complete_11_sample_history_fraction": complete / history_opportunities if history_opportunities else None,
            "identity_consistent_matched_history_observations": matched_history_complete,
            "truth_instance_complete_history_fraction": matched_history_complete / total_truth if total_truth else None,
        }
    counts = [len(f["detections"]) for f in frames]
    return {"thresholds": scores, "per_frame_player_counts": counts,
            "per_frame_count_median": float(np.median(counts)) if counts else None}


def development_scores(output: Path, weights: Path, *, save_name: str | None = None) -> dict:
    from ultralytics import YOLO

    manifest = json.loads((output / "data_manifest.json").read_text())
    model = YOLO(str(weights))
    result, saved_predictions = {}, {}
    for match_id in ("117093", "117092"):
        source = manifest["sources"][match_id]
        records = manifest["splits"][match_id + "_dev"]["frames"]
        predictions = []
        for record in records:
            tiles = [cv2.imread(str(ROOT / tile["path"])) for tile in record["tiles"]]
            if any(tile is None for tile in tiles):
                raise OSError("Missing prepared development tile")
            detections = merged_detections(model, tiles, [tile["bounds"] for tile in record["tiles"]])
            predictions.append({"frame_index": record["frame_index"], "detections": detections})
        labels = _read_annotations(source, {r["frame_index"] for r in records})
        result[match_id] = aggregate_scores(predictions, labels, source["dimensions_px"][0])
        saved_predictions[match_id] = predictions
    summary = {match: value["thresholds"]["40"] for match, value in result.items()}
    result["mean_per_match_f1"] = float(np.mean([m["f1"] for m in summary.values()]))
    result["weights_sha256"] = file_digest(weights)
    result["created_utc"] = now()
    if save_name:
        write_json(output / save_name, result)
        write_json(output / save_name.replace(".json", "_predictions.json"), saved_predictions)
    del model
    torch.cuda.empty_cache()
    return result


def detector_smoke(output: Path) -> dict:
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from torch.utils.data import DataLoader

    if (output / "detector_smoke.json").exists():
        raise FileExistsError("Completed smoke evidence is immutable; choose a new experiment root")
    require_cuda()
    torch.manual_seed(42)
    data = check_det_dataset(str((output / "data_v1/dataset.yaml").resolve()), autodownload=False)
    config = get_cfg(overrides={"imgsz": 1280, "workers": 0, "cache": False, "mosaic": 0.,
                                "mixup": 0., "scale": .25, "fliplr": .5})
    dataset = build_yolo_dataset(config, data["train"], batch=16, data=data,
                                 mode="train", rect=False, stride=32)
    by_match = [[i for i, path in enumerate(dataset.im_files) if match in path]
                for match in ("117093", "117092")]
    mixed_indices = [index for pair in zip(*by_match) for index in pair]
    weights = ROOT / json.loads((HISTORICAL / "protocol.json").read_text())["preservation"]["source_detector"]
    benchmarks = []
    for batch_size in (4, 8, 16):
        model = YOLO(str(weights)).model.to("cuda:0").train()
        model.args = config
        enable_detector_gradients(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=128.)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=mixed_indices, num_workers=0,
                            collate_fn=dataset.collate_fn)
        iterator = iter(loader)
        before = next(model.parameters()).detach().clone()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        losses = []
        for _ in range(10):
            batch = next(iterator)
            batch = {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            batch["img"] = batch["img"].float() / 255.
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, _ = model(batch)
                loss = loss.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite actual detector loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError("Nonfinite or missing detector gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            losses.append(float(loss))
        elapsed = time.perf_counter() - start
        changed = float((before - next(model.parameters())).abs().max())
        if changed <= 0:
            raise RuntimeError("Detector optimizer made no parameter update")
        benchmark = {
            "batch": batch_size, "optimizer_steps": 10, "images_per_second": batch_size * 10 / elapsed,
            "elapsed_seconds": elapsed, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "losses": losses, "parameter_change": changed,
        }
        benchmarks.append(benchmark)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate(), "optimizer_steps": 10}
        torch.save(state, output / f"smoke_b{batch_size}.pt")
        del iterator, loader, model, optimizer, scaler
        torch.cuda.empty_cache()
    chosen = max((b for b in benchmarks if b["peak_allocated_mib"] < 16000),
                 key=lambda b: b["images_per_second"])
    batch_size = chosen["batch"]
    model = YOLO(str(weights)).model.to("cuda:0").train()
    model.args = config
    enable_detector_gradients(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=128.)
    state = torch.load(output / f"smoke_b{batch_size}.pt", map_location="cuda:0", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    loader = DataLoader(dataset, batch_size=batch_size, sampler=mixed_indices, num_workers=0,
                        collate_fn=dataset.collate_fn)
    iterator = iter(loader)
    for _ in range(10):
        next(iterator)
    before = next(model.parameters()).detach().clone()
    for _ in range(10):
        batch = next(iterator)
        batch = {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        batch["img"] = batch["img"].float() / 255.
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = model(batch)
            loss = loss.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Resume produced nonfinite detector loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError("Resume produced nonfinite gradients")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
    assert float((before - next(model.parameters())).abs().max()) > 0
    inference = YOLO(str(weights))
    inference.model.load_state_dict(model.state_dict())
    sample = cv2.imread(str(ROOT / json.loads((output / "data_manifest.json").read_text())["splits"]["117093_dev"]["frames"][0]["tiles"][0]["path"]))
    predictions = inference.predict(sample, device=0, imgsz=1280, verbose=False)
    validation = inference.val(data=str((output / "data_v1/dataset.yaml").resolve()),
                               device=0, imgsz=1280, batch=batch_size, workers=0,
                               plots=False, verbose=False, project=str(output), name="smoke_val")
    report = {
        "status": "PASS", "created_utc": now(), "device": "cuda:0", "amp": True,
        "actual_optimizer_steps": 40, "benchmarks": benchmarks, "selected_batch": batch_size,
        "checkpoint_reload_and_nms": "PASS", "resume_additional_steps": 10,
        "resume_scope": "Disposable smoke model/optimizer/AMP/RNG restored; sequential loader advanced 10 batches. Not proof of exact full-trainer continuation.",
        "inference_boxes": len(predictions[0].boxes),
        "validation_box_metrics": validation.results_dict,
        "initialization_for_main_fit": "Frozen source checkpoint, NOT any smoke-trained weights",
    }
    write_json(output / "detector_smoke.json", report)
    return report


def inspect_loader_examples(output: Path) -> dict:
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    manifest = json.loads((output / "data_manifest.json").read_text())
    data = check_det_dataset(str((output / "data_v1/dataset.yaml").resolve()), autodownload=False)
    config = get_cfg(overrides={"imgsz": 1280, "workers": 0, "cache": False, "mosaic": 0.,
                                "mixup": 0., "scale": .25, "fliplr": .5})
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    examples = []
    for split in ("fit", "dev"):
        dataset = build_yolo_dataset(config, data["train" if split == "fit" else "val"],
                                     batch=8, data=data, mode="train" if split == "fit" else "val",
                                     rect=False, stride=32)
        paths = {str(Path(path).resolve()): index for index, path in enumerate(dataset.im_files)}
        for match in ("117093", "117092"):
            records = manifest["splits"][match + "_" + split]["frames"]
            panels = []
            for record in records[::max(1, len(records) // 6)][:6]:
                tile = max(record["tiles"], key=lambda r: len(
                    (ROOT / r["path"].replace("images", "labels")).with_suffix(".txt").read_text().splitlines()
                ))
                sample = dataset[paths[str((ROOT / tile["path"]).resolve())]]
                image = sample["img"].permute(1, 2, 0).numpy()[:, :, ::-1].copy()
                h, w = image.shape[:2]
                boxes = sample["bboxes"].numpy()
                xyxy = np.array([[(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]
                                 for cx, cy, bw, bh in boxes])
                draw_boxes(image, xyxy, (0, 255, 0))
                cv2.putText(image, f"{match} {split} frame {record['frame_index']} actual loader; RGB tensor",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                path = output / "loader_examples" / f"{match}_{split}_{record['frame_index']}.jpg"
                path.parent.mkdir(exist_ok=True)
                if not cv2.imwrite(str(path), image):
                    raise OSError(path)
                examples.append({"path": str(path.relative_to(ROOT)), "source_tile": tile,
                                 "shape": list(sample["img"].shape), "normalized_boxes": boxes.tolist(),
                                 "classes": sample["cls"].tolist(), "training_augmentation": split == "fit"})
                panels.append(cv2.resize(image, (400, 400)))
            contact = np.concatenate([np.concatenate(panels[i:i+3], axis=1) for i in (0, 3)])
            cv2.imwrite(str(output / f"{match}_{split}_actual_loader.jpg"), contact)
    report = {"status": "INSPECTED_LOADER_EXAMPLES", "seed": 42, "examples": examples,
              "scope": "Actual YOLODataset image tensor and delivered labels; fit includes affine, HSV and flips. Separate native/letterbox examples verify coordinate restoration."}
    write_json(output / "loader_examples.json", report)
    return report


def train(output: Path) -> dict:
    from src.backbone.amp_guard import AmpGuard
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    if (output / "recipe.json").exists():
        raise FileExistsError("Recipe already locked; use explicit checkpoint resume diagnostics")
    require_cuda()
    smoke = json.loads((output / "detector_smoke.json").read_text())
    if smoke["status"] != "PASS":
        raise RuntimeError("Actual detector smoke gate is not PASS")
    protocol = json.loads((HISTORICAL / "protocol.json").read_text())
    weights = ROOT / protocol["preservation"]["source_detector"]
    baseline = development_scores(output, weights, save_name="source_development.json")
    recipe = {
        "created_utc": now(), "initialization": str(weights.relative_to(ROOT)),
        "initialization_sha256": file_digest(weights), "seed": 42, "epochs": 40,
        "optimizer": "AdamW", "lr0": .001, "lrf": .01, "weight_decay": .0005,
        "batch": smoke["selected_batch"], "imgsz": 1280, "workers": 0, "amp": True,
        "mosaic": 0., "scale": .25, "amp_initial_scale": 128., "patience_balanced_epochs": 8,
        "selection": "Maximize mean per-match development F1 at confidence .20 and width-scaled 40px; each match recall >= frozen source recall minus .02. No E labels.",
        "operating_point": {"confidence": .2, "native_nms_iou": .5, "tolerance_reference_px": [20, 40]},
        "maximum_fit_seconds": 4 * 3600,
        "hard_fit_stop_utc": "2026-10-29T20:55:07+00:00",
        "dataset_manifest_sha256": file_digest(output / "data_manifest.json"),
        "worker_source_sha256": file_digest(Path(__file__)),
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "history": "117093 dev previously examined; 118576 retrospective evaluation excluded from selection.",
        "scheme": "Consistent-scale 4 horizontal by 2 vertical tiles for 1906-high input; 4 by 1 for 1080-high input. Native geometry preserved.",
    }
    if (output / "recipe.json").exists():
        raise FileExistsError("Recipe already locked; use explicit resume rather than re-fitting")
    write_json(output / "recipe.json", recipe)
    started = time.monotonic()
    status = {"optimizer_steps": 0, "best_f1": baseline["mean_per_match_f1"],
              "selected": str(weights.relative_to(ROOT)), "candidate_best_f1": -1.,
              "epochs_without_gain": 0, "epochs": []}

    class AuditedTrainer(DetectionTrainer):
        def optimizer_step(self):
            if self.device.type != "cuda":
                raise RuntimeError("CUDA device required")
            self.update_guard.update(self.model, self.optimizer, self.scaler, self.loss,
                                     ema=self.ema, context=self.guard_context)
            status["optimizer_steps"] = self.update_guard.counters.committed
            status["update_accounting"] = self.update_guard.state_dict()

        def preprocess_batch(self, batch):
            self.guard_context = {"fitting_samples": list(batch.get("im_file", [])),
                                  "epoch": self.epoch, "augmentation": "Resolved native YOLO training transforms",
                                  "rng": "Local first_abnormal_rng.pt"}
            self.update_guard.note_minibatch(torch.zeros((), device=self.device))
            return super().preprocess_batch(batch)

        def _handle_nan_recovery(self, epoch):
            if not torch.isfinite(self.loss).all():
                raise RuntimeError("Nonfinite loss; stopping rather than retrying")
            return False

    def batch_end(trainer):
        if trainer.batch_size != recipe["batch"]:
            raise RuntimeError("Trainer changed batch size; abort rather than silently retry")
        if not torch.isfinite(trainer.loss).all():
            raise RuntimeError("Nonfinite detector loss")
        if (time.monotonic() - started > recipe["maximum_fit_seconds"]
                or datetime.now(timezone.utc) >= datetime.fromisoformat(recipe["hard_fit_stop_utc"])):
            trainer.stop = True

    def started_callback(trainer):
        if trainer.device.type != "cuda" or type(trainer.optimizer).__name__ != "AdamW":
            raise RuntimeError("Resolved trainer does not match CUDA/AdamW recipe")
        initial_lrs = [group["initial_lr"] for group in trainer.optimizer.param_groups]
        if not all(abs(lr - recipe["lr0"]) < 1e-10 for lr in initial_lrs):
            raise RuntimeError(f"Resolved learning rate mismatch: {initial_lrs}")
        trainer.scaler = torch.amp.GradScaler("cuda", enabled=bool(trainer.amp), init_scale=128.)
        def record_event(event):
            with (output / "step_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if event.get("first_abnormal"):
                torch.save({"torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                            "numpy_rng": np.random.get_state(), "python_rng": random.getstate()},
                           output / "first_abnormal_rng.pt")
        trainer.update_guard = AmpGuard(event_sink=record_event)
        trainer.guard_context = {}
        write_json(output / "resolved_trainer.json", {
            "args": vars(trainer.args), "device": str(trainer.device),
            "optimizer": type(trainer.optimizer).__name__, "initial_lrs": initial_lrs,
            "gpu": torch.cuda.get_device_name(), "train_batches": len(trainer.train_loader),
        })

    def save_callback(trainer):
        checkpoint = output / "last_training.pt"
        shutil.copy2(trainer.last, checkpoint)
        torch.save({
            "scheduler": trainer.scheduler.state_dict(), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "loader_generator": trainer.train_loader.generator.get_state(),
            "optimizer_steps": status["optimizer_steps"],
            "epoch": trainer.epoch, "sampler": type(trainer.train_loader.sampler).__name__,
            "resume_limitation": "Native checkpoints serialize EMA, not live model; sampler iteration state not serialized. Use weights-only-restart terminology, not exact continuation.",
        }, output / "last_runtime_state.pt")
        metrics = development_scores(output, checkpoint)
        recalls_ok = all(
            metrics[m]["thresholds"]["40"]["recall"] >= baseline[m]["thresholds"]["40"]["recall"] - .02
            for m in ("117093", "117092")
        )
        gain = recalls_ok and metrics["mean_per_match_f1"] > status["best_f1"] + .0001
        record = {"epoch": trainer.epoch + 1, "optimizer_steps": status["optimizer_steps"],
                  "eligible": recalls_ok, "mean_f1": metrics["mean_per_match_f1"],
                  "matches": {m: metrics[m]["thresholds"] for m in ("117093", "117092")},
                  "box_metrics": trainer.metrics, "elapsed_seconds": time.monotonic() - started}
        status["epochs"].append(record)
        if metrics["mean_per_match_f1"] > status["candidate_best_f1"]:
            candidate = output / "candidate_best.pt"
            shutil.copy2(checkpoint, candidate)
            status.update(candidate_best_f1=metrics["mean_per_match_f1"],
                          candidate_best=str(candidate.relative_to(ROOT)))
        if gain:
            candidate = output / "selected.pt"
            shutil.copy2(checkpoint, candidate)
            status.update(best_f1=metrics["mean_per_match_f1"], selected=str(candidate.relative_to(ROOT)),
                          epochs_without_gain=0)
            write_json(output / "selected_development.json", metrics)
        else:
            status["epochs_without_gain"] += 1
        write_json(output / "training_progress.json", status)
        print(f"BALANCED epoch={trainer.epoch + 1} F1={metrics['mean_per_match_f1']:.5f} eligible={recalls_ok} selected={gain}", flush=True)
        if status["epochs_without_gain"] >= recipe["patience_balanced_epochs"]:
            trainer.stop = True

    model = YOLO(str(weights))
    model.add_callback("on_train_start", started_callback)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_model_save", save_callback)
    model.train(
        trainer=AuditedTrainer, data=str((output / "data_v1/dataset.yaml").resolve()),
        epochs=40, imgsz=1280, batch=recipe["batch"], device=0, workers=0,
        project=str(output), name="fit", exist_ok=False, seed=42, deterministic=True,
        optimizer="AdamW", lr0=.001, lrf=.01, weight_decay=.0005, warmup_epochs=3.,
        mosaic=0., mixup=0., scale=.25, close_mosaic=0, nbs=recipe["batch"],
        cache=False, amp=True, patience=100, plots=True, verbose=False, save_period=1,
    )
    status.update(status="TRAINED_EXPERIMENTAL", finished_utc=now(),
                  elapsed_seconds=time.monotonic() - started,
                  selected_sha256=file_digest(ROOT / status["selected"]),
                  release_promotion=False)
    write_json(output / "training_summary.json", status)
    diagnostic_weights = (status["selected"] if (ROOT / status["selected"]).resolve() != weights.resolve()
                          else status["candidate_best"])
    development_scores(output, ROOT / diagnostic_weights, save_name="candidate_development.json")
    return status


def predict_video(output: Path, source: dict, interval: list[int], route: str,
                  weights: Path, *, scheme: str, stride: int = 5) -> dict:
    from ultralytics import YOLO
    from src.backbone.video_player_replay import _assign_tracklets

    require_cuda()
    if len(interval) != 2 or interval[0] < 0 or interval[1] <= interval[0] or stride <= 0:
        raise ValueError("A positive source interval and sampling stride are required")
    destination = output / "predictions" / route / f"{source['id']}_{interval[0]}_{interval[1]}.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / source["video"]), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source['video']}")
    dimensions = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    if dimensions != source["dimensions_px"] or abs(cap.get(cv2.CAP_PROP_FPS) - 25) > .001:
        raise ValueError(f"Video clock/geometry mismatch: {source['id']}")
    if interval[1] * 25 > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        raise ValueError("Requested interval extends beyond the source video")
    model = YOLO(str(weights))
    digest = file_digest(weights)
    previous, tracks, next_id = None, {}, 1
    started = time.monotonic()
    try:
        with destination.with_suffix(".partial").open("w", encoding="utf-8") as stream:
            for sample, index in enumerate(range(interval[0] * 25, interval[1] * 25, stride)):
                image = _read_frame(cap, index, previous)
                previous = index
                bounds = tile_bounds(*dimensions, scheme=scheme)
                tiles = [image[t:b, l:r] for l, t, r, b in bounds]
                detections = merged_detections(model, tiles, bounds)
                next_id = _assign_tracklets(detections, tracks, next_id, sample,
                                            max_distance_px=180 * dimensions[0] / 4096)
                record = {
                    "schema_version": "football.video-player-cross-match-prediction.1",
                    "match_id": source["id"], "half": source["half"],
                    "interval": f"{interval[0]}_{interval[1]}", "sample_index": sample,
                    "frame_index": index, "timestamp_s": index / 25,
                    "dimensions_px": dimensions, "detector_sha256": digest,
                    "detector_confidence_threshold": .20, "detections": detections,
                }
                stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    finally:
        cap.release()
    destination.with_suffix(".partial").replace(destination)
    metadata = {
        "status": "FROZEN_VIDEO_ONLY_PREDICTIONS", "match_id": source["id"],
        "half": source["half"], "route": route, "interval_seconds": interval,
        "source_video_sha256": source["video_sha256"], "detector_sha256": digest,
        "prediction_jsonl_sha256": file_digest(destination), "labels_read": False,
        "scheme": scheme, "sampling_hz": 25 / stride, "elapsed_seconds": time.monotonic() - started,
        "sample_count": len(range(interval[0] * 25, interval[1] * 25, stride)),
        "created_utc": now(), "device": "cuda:0",
    }
    write_json(destination.with_suffix(".meta.json"), metadata)
    del model
    torch.cuda.empty_cache()
    return {"path": str(destination.relative_to(ROOT)), **metadata}


def score_predictions(output: Path, record: dict, source: dict) -> dict:
    from src.backbone.video_player_cross_match import load_local_projector, project_footpoint

    path = ROOT / record["path"]
    if file_digest(path) != record["prediction_jsonl_sha256"]:
        raise ValueError(f"Sealed predictions changed: {path}")
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    sampling_hz = record.get("sampling_hz", record.get("sample_rate_hz", 5.))
    if not math.isfinite(sampling_hz) or sampling_hz <= 0:
        raise ValueError("Prediction sampling frequency must be finite and positive")
    stride = 25 / sampling_hz
    if not float(stride).is_integer():
        raise ValueError("Prediction sampling does not match the native frame clock")
    start, end = record["interval_seconds"]
    expected = list(range(start * 25, end * 25, int(stride)))
    if [frame["frame_index"] for frame in frames] != expected:
        raise ValueError(f"Prediction coverage is not the complete locked interval: {path}")
    labels = _read_annotations(source, {frame["frame_index"] for frame in frames})
    result = aggregate_scores(frames, labels, source["dimensions_px"][0],
                              sampling_hz)
    projector_file = HISTORICAL / "calibration" / f"local_projector_{source['id']}.json"
    projected, errors, supported_truth = 0, [], 0
    if projector_file.is_file():
        protocol = load_protocol(HISTORICAL / "protocol.json")
        projector = load_local_projector(protocol, HISTORICAL, source["id"])
        for frame in frames:
            truth = labels.get(frame["frame_index"], [])
            support = [project_footpoint(a["footpoint_xy_px"], projector)[0] is not None for a in truth]
            supported_truth += sum(support)
            positions = [project_footpoint(d["footpoint_xy_px"], projector)[0] for d in frame["detections"]]
            projected += sum(p is not None for p in positions)
            pairs = _maximum_matches(frame["detections"], truth, 40 * source["dimensions_px"][0] / 4096)
            for pi, gi, _ in pairs:
                target = truth[gi].get("pitch_xy_centered_m")
                if positions[pi] is not None and support[gi] and target is not None:
                    errors.append(math.dist(positions[pi], [target[0] + 52.5, target[1] + 34]))
    result["projection"] = {
        "status": "UNVALIDATED/EXPERIMENTAL", "global_calibration_pass": False,
        "projected": projected,
        "unprojected": result["thresholds"]["40"]["detections"] - projected,
        "supported_truth_instances": supported_truth, "pixel_matched_in_support_with_pitch_truth": len(errors),
        "conditional_median_m": float(np.median(errors)) if errors else None,
        "conditional_p90_m": float(np.percentile(errors, 90)) if errors else None,
        "conditional_errors_m": errors,
    }
    result["provenance"] = record
    return result


def evaluate(output: Path) -> dict:
    protocol = load_protocol(HISTORICAL / "protocol.json")
    sources = source_records()
    training = json.loads((output / "training_summary.json").read_text())
    continued = HISTORICAL / "detector_adaptation_continued_20261028/detector_candidate/weights/best.pt"
    retained_source = (ROOT / training["selected"]).resolve() == (ROOT / protocol["preservation"]["source_detector"]).resolve()
    candidate_path = (training["selected"] if not retained_source
                      else training["candidate_best"])
    diagnostic_path = output / "candidate_development.json"
    if not diagnostic_path.is_file() or json.loads(diagnostic_path.read_text())["weights_sha256"] != file_digest(ROOT / candidate_path):
        development_scores(output, ROOT / candidate_path, save_name="evaluated_candidate_development.json")
    routes = {
        "gpu_selected": (ROOT / candidate_path, "consistent"),
        "continued": (continued, "historical"),
    }
    sealed = []
    for route, (weights, scheme) in routes.items():
        for match_id in ("117092", "118576"):
            for interval in protocol["split"]["final_intervals_seconds_by_match"][match_id]:
                sealed.append(predict_video(output, sources[match_id], interval, route, weights, scheme=scheme))
    for match_id in ("117092", "118576"):
        for index, interval in enumerate(protocol["split"]["final_intervals_seconds_by_match"][match_id]):
            path = HISTORICAL / "predictions/frozen" / f"{match_id}_{sources[match_id]['half']}_final{index}.jsonl"
            meta = json.loads(path.with_suffix(".meta.json").read_text())
            if file_digest(path) != meta["prediction_jsonl_sha256"]:
                raise ValueError("Historical frozen predictions changed")
            sealed.append({"path": str(path.relative_to(ROOT)), "route": "frozen", "interval_seconds": interval,
                           "match_id": match_id, **{k: v for k, v in meta.items() if k != "route"}})
    for match_id in ("117092", "118576"):
        sealed.append(predict_video(output, sources[match_id], [300, 420], "pretrained_person",
                                    ROOT / "yolo11n.pt", scheme="historical", stride=25))
    write_json(output / "evaluation_seal.json", {
        "created_utc": now(), "status": "ALL_PREDICTIONS_HASHED_BEFORE_NEW_SCORING",
        "exposure": "All six D/E intervals are retrospective diagnostics, not untouched test data.",
        "control": "Pretrained COCO person bottom-center proxy scored at pixel footpoint tolerance; never full-body-versus-footpoint box IoU.",
        "files": sealed,
    })
    scores = []
    for record in sealed:
        score = score_predictions(output, record, sources[record["match_id"]])
        target = output / "scores" / record["route"] / f"{record['match_id']}_{record['interval_seconds'][0]}.json"
        write_json(target, score)
        scores.append({"path": str(target.relative_to(ROOT)), "route": record["route"],
                       "match_id": record["match_id"], "interval": record["interval_seconds"],
                       "thresholds": score["thresholds"], "projection": score["projection"]})
    summary = {"status": "RETROSPECTIVE_EVALUATION_COMPLETE", "created_utc": now(), "scores": scores,
               "candidate_passed_development_retention": not retained_source,
               "selection_note": "gpu_selected identifies the best newly trained balanced-development diagnostic; source is retained if retention constraints failed."}
    write_json(output / "evaluation_summary.json", summary)
    return summary


def integrate(output: Path) -> dict:
    from src.backbone.video_player_cross_match_replay import (
        build_external_observations, render_video_comparison, replay_route,
    )

    protocol = load_protocol(HISTORICAL / "protocol.json")
    sources = source_records()
    evaluation = json.loads((output / "evaluation_seal.json").read_text())
    association = output / "association_seal.json"
    if association.is_file():
        replacements = json.loads(association.read_text())["files"]
        evaluation["files"] = [
            next((new for new in replacements
                  if new["match_id"] == record["match_id"]
                  and new["interval_seconds"] == record["interval_seconds"]), record)
            if record["route"] == "gpu_selected" else record
            for record in evaluation["files"]
        ]
    write_json(output / "integration_input_seal.json", {
        "created_utc": now(), "association": "development-selected global" if association.is_file() else "original greedy",
        "files": evaluation["files"],
    })
    results = []
    for match_id in ("117092", "118576"):
        destination = output / f"r{match_id}"
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir()
        local_protocol = json.loads(json.dumps(protocol))
        local_protocol["split"]["development_match"] = match_id
        local_protocol["split"]["development_interval_seconds"] = [300, 420]
        write_json(destination / "protocol.json", local_protocol)
        (destination / "calibration").mkdir()
        shutil.copy2(HISTORICAL / "calibration" / f"local_projector_{match_id}.json",
                     destination / "calibration" / f"local_projector_{match_id}.json")
        for route, published in (("adapted", "gpu_selected"), ("frozen", "frozen")):
            record = next(row for row in evaluation["files"] if row["route"] == published
                          and row["match_id"] == match_id and row["interval_seconds"] == [300, 420])
            source_path = ROOT / record["path"]
            if file_digest(source_path) != record["prediction_jsonl_sha256"]:
                raise ValueError("Integration prediction seal failed")
            prediction = destination / "predictions" / route / f"{match_id}_{sources[match_id]['half']}_development.jsonl"
            prediction.parent.mkdir(parents=True)
            with prediction.open("w", encoding="utf-8") as stream:
                for line in source_path.read_text().splitlines():
                    row = json.loads(line)
                    row["interval"] = "development"
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            write_json(prediction.with_suffix(".meta.json"), {
                **record, "interval": "development", "route": route,
                "sample_count": 600, "prediction_jsonl_sha256": file_digest(prediction),
                "derived_from_sealed_prediction_sha256": record["prediction_jsonl_sha256"],
            })
        builds = []
        replays = []
        for route in ("adapted", "frozen"):
            builds.append(build_external_observations(local_protocol, destination, match_id, "development",
                                                     route, assisted_attributes=False, device="0"))
        for route in ("adapted", "frozen"):
            replays.append(replay_route(local_protocol, destination, match_id, "development",
                                        route, assisted_attributes=False))
            render_video_comparison(local_protocol, destination, match_id, "development", route,
                                    duration_seconds=120)
        results.append({"match": match_id, "builds": builds, "replays": replays})
        write_json(output / "integration_progress.json", {"results": results, "updated_utc": now()})
    summary = {"status": "EXPERIMENTAL_INTEGRATION_COMPLETE", "results": results, "finished_utc": now()}
    write_json(output / "integration_summary.json", summary)
    return summary


def verify(output: Path) -> dict:
    protocol = json.loads((HISTORICAL / "protocol.json").read_text())
    release = json.loads((ROOT / "configs/backbone_release_manifest.json").read_text())
    pinned = {
        "configs/backbone_release_baseline.json": protocol["preservation"]["release_config_sha256"],
        protocol["preservation"]["source_detector"]: protocol["preservation"]["source_detector_sha256"],
        **{row["checkpoint"]: row["sha256"] for key, row in release["models"].items()
           if key in ("default_world", "default_action")},
    }
    for path, digest in pinned.items():
        if file_digest(ROOT / path) != digest:
            raise ValueError(f"Frozen release changed: {path}")
    manifest = json.loads((output / "data_manifest.json").read_text())
    recipe = json.loads((output / "recipe.json").read_text())
    if file_digest(output / "worker_training_snapshot.py") != recipe["worker_source_sha256"]:
        raise ValueError("Archived training source does not match the locked recipe")
    if file_digest(output / "data_manifest.json") != recipe["dataset_manifest_sha256"]:
        raise ValueError("Data manifest changed after recipe lock")
    for match_id in ("117093", "117092"):
        source = manifest["sources"][match_id]
        if file_digest(ROOT / source["annotation"]) != source["annotation_sha256"]:
            raise ValueError(f"Human annotation changed: {match_id}")
        fit = {r["frame_index"] for r in manifest["splits"][match_id + "_fit"]["frames"]}
        dev = {r["frame_index"] for r in manifest["splits"][match_id + "_dev"]["frames"]}
        if fit & dev:
            raise ValueError("Source frame split leakage")
    for source in manifest["sources"].values():
        if file_digest(ROOT / source["annotation"]) != source["annotation_sha256"]:
            raise ValueError(f"Annotation changed: {source['id']}")
    for split, report in manifest["splits"].items():
        for folder, field, suffix in (("labels", "label_hashes", ".txt"), ("images", "image_hashes", ".jpg")):
            for name, digest in report[field]:
                path = output / "data_v1" / folder / split / Path(name).with_suffix(suffix)
                if file_digest(path) != digest:
                    raise ValueError(f"Prepared data changed after audit: {path}")
    listed = (output / "data_v1/fit.txt").read_text() + (output / "data_v1/dev.txt").read_text()
    if any(match in listed for match in ("118576", "128057")):
        raise ValueError("Evaluation match entered fitting/selection lists")
    additional = output / "additional_match_lock.json"
    if additional.is_file():
        extra = json.loads(additional.read_text())
        if file_digest(ROOT / extra["annotation"]) != extra["annotation_sha256"]:
            raise ValueError("Additional-match annotation changed")
    seal = json.loads((output / "evaluation_seal.json").read_text())
    for record in seal["files"]:
        path = ROOT / record["path"]
        if file_digest(path) != record["prediction_jsonl_sha256"]:
            raise ValueError(f"Sealed predictions changed: {path}")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for detection in row["detections"]:
                if {"player_id", "reference_id", "pitch_xy_centered_m", "team", "role"} & detection.keys():
                    raise ValueError("Reference attributes or coordinates in detector runtime output")
    integration = json.loads((output / "integration_summary.json").read_text())
    if any(build["annotation_attributes_used"] for match in integration["results"] for build in match["builds"]):
        raise ValueError("Assisted attributes entered autonomous integration")
    tests = ["tests/test_gpu_perception.py", "tests/test_video_player_cross_match.py",
             "tests/test_video_player_cross_match_replay.py", "tests/test_video_player_overnight.py",
             "tests/test_regional_pitch_mapping.py", "tests/test_video_player_replay.py",
             "tests/test_video_pitch_experimental.py", "tests/test_backbone_release.py",
             "tests/test_video_review_server.py"]
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests],
                               cwd=ROOT, capture_output=True, text=True, timeout=300)
    (output / "regressions.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Regressions failed; see {output / 'regressions.txt'}")
    summary = {"status": "PASS", "created_utc": now(), "frozen_hashes_preserved": pinned,
               "annotation_hashes_preserved": True, "evaluation_labels_excluded_from_fit_selection": True,
               "training_source_snapshot_sha256": recipe["worker_source_sha256"],
               "final_source_sha256": {
                   str(path.relative_to(ROOT)): file_digest(path)
                   for path in (ROOT / "src/backbone/gpu_perception.py",
                                ROOT / "src/backbone/gpu_perception_post.py",
                                ROOT / "src/backbone/video_review_server.py")
               },
               "regression_summary": completed.stdout.strip().splitlines()[-1]}
    write_json(output / "verification.json", summary)
    return summary


def finalize_interrupted(output: Path) -> dict:
    if (output / "training_summary.json").exists():
        raise FileExistsError("Training outcome already recorded")
    log = output / "training.log"
    log_bytes = log.read_bytes()
    log_text = log_bytes.decode("utf-16" if log_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8")
    if "RuntimeError: Nonfinite detector gradients; no automatic recovery" not in log_text:
        raise ValueError("This recovery only accepts the recorded nonfinite-gradient termination")
    progress = json.loads((output / "training_progress.json").read_text())
    recipe = json.loads((output / "recipe.json").read_text())
    checkpoint = torch.load(output / "last_training.pt", map_location="cpu", weights_only=False)
    if checkpoint["epoch"] + 1 != progress["epochs"][-1]["epoch"]:
        raise ValueError("Completed epoch and preserved checkpoint disagree")
    if not all(torch.isfinite(parameter).all() for parameter in checkpoint["ema"].parameters()):
        raise ValueError("Preserved checkpoint parameters are not finite")
    selected = ROOT / progress["selected"]
    if not selected.is_file():
        raise FileNotFoundError(selected)
    failed_utc = datetime.fromtimestamp(log.stat().st_mtime, timezone.utc)
    progress.update(
        status="INTERRUPTED_NONFINITE_GRADIENTS_CHECKPOINT_EVALUATION_ONLY",
        finished_utc=failed_utc.isoformat(),
        elapsed_seconds=(failed_utc - datetime.fromisoformat(recipe["created_utc"])).total_seconds(),
        selected_sha256=file_digest(selected), release_promotion=False,
        optimizer_step_count_semantics="Confirmed completed-epoch lower bound; updates in the interrupted next epoch were not journaled.",
        interrupted_epoch=checkpoint["epoch"] + 2,
        failure="Nonfinite detector gradients; original process terminated without automatic recovery.",
        failure_log_sha256=file_digest(log),
        recovery="No training restart. Evaluate the already selected finite checkpoint; CUDA inference health checked separately.",
    )
    write_json(output / "training_summary.json", progress)
    return progress


def run(output: Path) -> dict:
    stages = {"smoke": detector_smoke, "train": train, "evaluate": evaluate,
              "integrate": integrate, "verify": verify}
    journal = {"started_utc": now(), "status": "RUNNING", "stages": []}
    for name, function in stages.items():
        journal["active_stage"] = name
        write_json(output / "worker_status.json", journal)
        try:
            result = function(output)
        except Exception as exc:
            journal.update(status="BLOCKED", error=str(exc), traceback=traceback.format_exc(), finished_utc=now())
            write_json(output / "worker_status.json", journal)
            raise
        journal["stages"].append({"stage": name, "finished_utc": now(), "status": result.get("status", "COMPLETE")})
        write_json(output / "worker_status.json", journal)
    journal.update(status="COMPLETE", finished_utc=now())
    write_json(output / "worker_status.json", journal)
    return journal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("command", choices=["prepare", "audit", "inspect-loader", "smoke", "train", "finalize-interrupted", "evaluate", "integrate", "verify", "run"])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    {"prepare": prepare, "audit": audit_prepared, "inspect-loader": inspect_loader_examples,
     "smoke": detector_smoke, "train": train, "finalize-interrupted": finalize_interrupted,
     "evaluate": evaluate, "integrate": integrate, "verify": verify, "run": run}[args.command](output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
