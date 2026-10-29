"""Resumable, chunked video detection for the v0.6 integration workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from src.backbone.video_ingest import _annotate, _detect_frame, _histogram
from src.backbone.video_tracklets import CUT_THRESHOLD, associate_tracklets
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
FIT_MATCHES = {"117092", "117093", "118576", "118578", "128057", "132831"}
DEVELOPMENT_MATCH = "132877"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _fingerprint(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sampled_frame_indices(start_s: float, duration_s: int, fps: float, target_fps: int) -> list[int]:
    samples = duration_s * target_fps
    indices = [round((start_s + index / target_fps) * fps) for index in range(samples)]
    if len(indices) != len(set(indices)):
        raise ValueError("Target sampling produces duplicate native source frames")
    return indices


def _clamped_duration_s(start_s: float, requested_duration_s: int, fps: float, frame_count: int) -> int:
    available_s = max(0.0, frame_count / fps - start_s)
    return min(requested_duration_s, math.floor(available_s + 1e-9))


def _chunk_rows(
    folder: Path,
    chunk_index: int,
    fingerprint: str,
    expected_frames: list[int],
) -> tuple[list[dict] | None, dict | None]:
    chunk = folder / "chunks" / f"chunk_{chunk_index:04d}.jsonl"
    completion = folder / "chunks" / f"chunk_{chunk_index:04d}.json"
    if not completion.is_file():
        return None, None
    record = _read_json(completion)
    if record.get("fingerprint") != fingerprint:
        raise ValueError(f"Existing chunk configuration differs; preserve and use a new output: {chunk}")
    if not chunk.is_file() or file_digest(chunk) != record.get("rows_sha256"):
        raise ValueError(f"Completed video chunk failed its hash check: {chunk}")
    rows = _read_jsonl(chunk)
    if [int(row["frame_index"]) for row in rows] != expected_frames:
        raise ValueError(f"Completed video chunk frame index mismatch: {chunk}")
    return rows, record


def _validate_config(config: dict) -> list[tuple[str, dict]]:
    if config.get("schema_version") != "football.backbone-video-workload.0.6":
        raise ValueError("Unsupported v0.6 video workload configuration")
    workload = config["workload"]
    if workload["target_fps"] != 5 or workload["chunk_samples"] < 25:
        raise ValueError("Video workload requires 5Hz sampling and chunks of at least five seconds")
    clips = [
        ("fitting", clip) for clip in config["fitting_clips"]
    ] + [
        ("development", clip) for clip in config["development_clips"]
    ]
    if len(config["fitting_clips"]) != 6 or len(config["development_clips"]) != 2:
        raise ValueError("The frozen workload is six fitting excerpts and two development excerpts")
    if len({str(clip["match"]) for split, clip in clips if split == "fitting"}) < 4:
        raise ValueError("Fitting excerpts must span at least four matches")
    intervals: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
    for split, clip in clips:
        match = str(clip["match"])
        if clip["half"] not in ("1st", "2nd") or float(clip["start_s"]) < 0:
            raise ValueError(f"Invalid video half or negative excerpt start: {clip}")
        if split == "fitting" and match not in FIT_MATCHES:
            raise ValueError(f"Non-fitting match assigned to fitting video: {match}")
        if split == "development" and match != DEVELOPMENT_MATCH:
            raise ValueError(f"Development video must use the declared development match: {match}")
        if int(clip["duration_s"]) != (1200 if split == "fitting" else 600):
            raise ValueError(f"Incorrect excerpt duration for {split}: {clip}")
        key = (str(clip["match"]), str(clip["half"]))
        start = float(clip["start_s"])
        end = start + int(clip["duration_s"])
        for old_start, old_end, old_split in intervals.setdefault(key, []):
            if max(start, old_start) < min(end, old_end):
                raise ValueError(f"Overlapping or reused excerpts in {key}: {old_split} and {split}")
        intervals[key].append((start, end, split))
    return clips


def _run_clip(
    split: str,
    clip: dict,
    *,
    config: dict,
    config_sha: str,
    root: Path,
    output: Path,
    model,
    weights_sha: str,
    started: float,
    max_runtime_s: float,
    totals: dict,
    resume: bool,
) -> dict:
    workload = config["workload"]
    match, half = str(clip["match"]), str(clip["half"])
    video_path = root / config["source_paths"]["video_root"] / match / f"{match}_panorama_{half}_half.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Configured local video is missing: {video_path}")
    source_sha = file_digest(video_path)
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open configured local video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_s, requested_duration_s = float(clip["start_s"]), int(clip["duration_s"])
    duration_s = _clamped_duration_s(start_s, requested_duration_s, fps, frame_count)
    if duration_s <= 0:
        capture.release()
        raise ValueError(f"Excerpt start lies beyond actual half boundary: {match} {half} at {start_s}s")
    indices = _sampled_frame_indices(start_s, duration_s, fps, int(workload["target_fps"]))

    clip_id = f"{match}_{half}_{int(start_s):05d}s_{duration_s:04d}s"
    clip_folder = output / split / clip_id
    clip_folder.mkdir(parents=True, exist_ok=True)
    if not resume and any(clip_folder.iterdir()):
        capture.release()
        raise FileExistsError(f"Refusing to overwrite an existing excerpt: {clip_folder}")

    all_rows: list[dict] = []
    previous_histogram: np.ndarray | None = None
    chunk_size = int(workload["chunk_samples"])
    overlays = {0, len(indices) // 2, len(indices) - 1}
    clip_started = time.monotonic()
    clip_new_frames = 0
    clip_reused_frames = 0
    try:
        for chunk_index, start in enumerate(range(0, len(indices), chunk_size)):
            expected_frames = indices[start:start + chunk_size]
            fingerprint = _fingerprint({
                "schema": "football.video-chunk.0.6",
                "config_sha256": config_sha,
                "source_sha256": source_sha,
                "weights_sha256": weights_sha,
                "clip_id": clip_id,
                "chunk_index": chunk_index,
                "frame_indices": expected_frames,
                "preprocessing": workload["detector"],
            })
            cached_rows, chunk_meta = _chunk_rows(
                clip_folder, chunk_index, fingerprint, expected_frames,
            )
            if cached_rows is not None:
                all_rows.extend(cached_rows)
                previous_histogram = np.asarray(chunk_meta["last_histogram"], dtype=np.float32).reshape(16, 8)
                totals["reused_chunks"] += 1
                totals["reused_inference_frames"] += len(cached_rows)
                clip_reused_frames += len(cached_rows)
                continue
            if time.monotonic() - started >= max_runtime_s:
                raise TimeoutError("V0.6 video workload paused at a completed chunk boundary")

            chunk_rows = []
            for offset, frame_index in enumerate(expected_frames):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                valid, image = capture.read()
                if not valid:
                    raise RuntimeError(f"Could not decode configured source frame {frame_index}")
                histogram = _histogram(image)
                distance = None if previous_histogram is None else float(cv2.compareHist(
                    previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA,
                ))
                detections = _detect_frame(
                    model, image,
                    columns=int(workload["detector"]["tile_columns"]),
                    overlap=int(workload["detector"]["tile_overlap_px"]),
                    confidence=float(workload["detector"]["confidence_threshold"]),
                    device=workload["device"],
                )
                sample_index = start + offset
                if sample_index in overlays:
                    _annotate(image, detections, clip_folder / f"overlay_{sample_index:05d}.jpg", frame_index)
                chunk_rows.append({
                    "record_type": "pixel_observations",
                    "sample_index": sample_index,
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
                previous_histogram = histogram

            chunk_path = clip_folder / "chunks" / f"chunk_{chunk_index:04d}.jsonl"
            _atomic_jsonl(chunk_path, chunk_rows)
            chunk_meta = {
                "fingerprint": fingerprint,
                "rows_sha256": file_digest(chunk_path),
                "first_frame": expected_frames[0],
                "last_frame": expected_frames[-1],
                "frame_count": len(chunk_rows),
                "last_histogram": previous_histogram.reshape(-1).tolist(),
                "status": "COMPLETE",
            }
            _atomic_json(clip_folder / "chunks" / f"chunk_{chunk_index:04d}.json", chunk_meta)
            all_rows.extend(chunk_rows)
            totals["new_inference_frames"] += len(chunk_rows)
            clip_new_frames += len(chunk_rows)
            write_json(output / "progress.json", {
                "status": "RUNNING",
                "active_clip": clip_id,
                "active_split": split,
                "completed_chunks_in_clip": chunk_index + 1,
                "new_inference_frames": totals["new_inference_frames"],
                "reused_inference_frames": totals["reused_inference_frames"],
                "target_fitting_frames": 36000,
                "target_development_frames": 6000,
                "elapsed_seconds": time.monotonic() - started,
            })
    finally:
        capture.release()

    if len(all_rows) != len(indices):
        raise RuntimeError(f"Chunk assembly count mismatch in {clip_id}")
    all_rows.sort(key=lambda row: int(row["sample_index"]))
    linked, tracklets = associate_tracklets(
        all_rows, [bool(row["potential_cut_before"]) for row in all_rows],
    )
    linked_path = clip_folder / "pixel_tracklets.jsonl"
    _atomic_jsonl(linked_path, linked)
    totals["detections"] += sum(len(row["detections"]) for row in linked)
    detections_per_frame = [len(row["detections"]) for row in linked]
    confidences = [
        float(detection["detector_confidence"])
        for row in linked for detection in row["detections"]
    ]
    clip_manifest = {
        "schema_version": "football.video-excerpt.0.6",
        "status": "PIXEL_DETECTION_AND_UNVERIFIED_TRACKLETS_COMPLETE",
        "split": split,
        "match": match,
        "half": half,
        "start_s": start_s,
        "duration_s": duration_s,
        "requested_duration_s": requested_duration_s,
        "coverage_shortfall_s": requested_duration_s - duration_s,
        "target_fps": int(workload["target_fps"]),
        "source_video": str(video_path.relative_to(root)),
        "source_video_sha256": source_sha,
        "source_fps": fps,
        "source_frame_count": frame_count,
        "source_dimensions_px": linked[0]["dimensions_px"],
        "first_sample_time_s": linked[0]["time_from_video_start_s"],
        "last_sample_time_s": linked[-1]["time_from_video_start_s"],
        "unique_source_frames": len({row["frame_index"] for row in linked}),
        "inference_frames": len(linked),
        "new_inference_frames": clip_new_frames,
        "reused_inference_frames": clip_reused_frames,
        "detections": sum(detections_per_frame),
        "empty_detection_frame_fraction": float(np.mean(np.asarray(detections_per_frame) == 0)),
        "detections_per_frame_p50_p90": [
            float(np.percentile(detections_per_frame, 50)),
            float(np.percentile(detections_per_frame, 90)),
        ],
        "confidence_p50_p90": [
            float(np.percentile(confidences, 50)) if confidences else None,
            float(np.percentile(confidences, 90)) if confidences else None,
        ],
        "potential_cut_count": sum(bool(row["potential_cut_before"]) for row in linked),
        "tracklet_candidates": len(tracklets),
        "tracklet_status": "UNVERIFIED_IMAGE_SPACE_CANDIDATES",
        "pitch_projection": "UNSUPPORTED",
        "team_and_goalkeeper_roles": "UNKNOWN",
        "observations_path": str(linked_path.relative_to(output)),
        "observations_sha256": file_digest(linked_path),
        "execution_seconds_this_invocation": time.monotonic() - clip_started,
        "chunk_count": len(list((clip_folder / "chunks").glob("chunk_*.json"))),
        "overlays": sorted(path.name for path in clip_folder.glob("overlay_*.jpg")),
    }
    _atomic_json(clip_folder / "tracklet_summary.json", {
        "status": "UNVERIFIED_IMAGE_SPACE_CANDIDATES",
        "identity_claim": "NONE",
        "pitch_projection": "UNSUPPORTED",
        "tracklets": tracklets,
    })
    _atomic_json(clip_folder / "clip_manifest.json", clip_manifest)
    totals[f"{split}_frames"] += len(linked)
    totals[f"{split}_unique_source_frames"] += len({row["frame_index"] for row in linked})
    return clip_manifest


def run(config_path: Path, output: Path, *, root: Path = ROOT, resume: bool = False) -> dict:
    config = _read_json(config_path)
    clips = _validate_config(config)
    workload = config["workload"]
    root = root.resolve()
    output = output.resolve()
    config_sha = file_digest(config_path)
    weights_path = root / config["source_paths"]["detector_weights"]
    if not weights_path.is_file():
        raise FileNotFoundError(f"Existing configured detector weights are unavailable: {weights_path}")
    weights_sha = file_digest(weights_path)
    if output.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite an existing v0.6 video run: {output}")
        state_path = output / "run_state.json"
        if not state_path.is_file() or _read_json(state_path).get("config_sha256") != config_sha:
            raise ValueError("Resume config differs or no safe run state exists; choose a new output directory")
    else:
        output.mkdir(parents=True)
    _atomic_json(output / "run_state.json", {
        "schema_version": "football.video-ingestion-state.0.6",
        "status": "RUNNING",
        "config_sha256": config_sha,
        "detector_weights_sha256": weights_sha,
        "resume_policy": "Reuse only atomically completed, hash-verified chunks with exact source/config fingerprints.",
    })

    from ultralytics import YOLO, __version__ as ultralytics_version

    cv2.setNumThreads(4)
    if workload["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable; no CPU substitution is permitted")
    model = YOLO(str(weights_path))
    started = time.monotonic()
    max_runtime_s = float(workload["max_runtime_minutes"]) * 60
    totals = {
        "new_inference_frames": 0,
        "reused_inference_frames": 0,
        "reused_chunks": 0,
        "detections": 0,
        "fitting_frames": 0,
        "fitting_unique_source_frames": 0,
        "development_frames": 0,
        "development_unique_source_frames": 0,
    }
    clip_manifests = []
    try:
        for split, clip in clips:
            clip_manifests.append(_run_clip(
                split, clip, config=config, config_sha=config_sha, root=root,
                output=output, model=model, weights_sha=weights_sha, started=started,
                max_runtime_s=max_runtime_s, totals=totals, resume=resume,
            ))
    except TimeoutError:
        _atomic_json(output / "run_state.json", {
            "schema_version": "football.video-ingestion-state.0.6",
            "status": "PAUSED_SAFE_BOUNDARY",
            "config_sha256": config_sha,
            "detector_weights_sha256": weights_sha,
            "totals": totals,
        })
        write_json(output / "progress.json", {
            "status": "PAUSED_SAFE_BOUNDARY",
            "totals": totals,
            "resume_command": (
                "python -m src.backbone.video_ingest_v06 --config "
                f"{config_path} --output {output} --resume"
            ),
        })
        return {"status": "PAUSED_SAFE_BOUNDARY", "totals": totals}

    expected_fit, expected_dev = 36000, 6000
    coverage_complete = (
        totals["fitting_frames"] == expected_fit
        and totals["development_frames"] == expected_dev
        and totals["fitting_unique_source_frames"] == expected_fit
        and totals["development_unique_source_frames"] == expected_dev
        and all(row["coverage_shortfall_s"] == 0 for row in clip_manifests)
    )
    manifest = {
        "schema_version": "football.video-ingestion.0.6",
        "status": "COMPLETE" if coverage_complete else "COMPLETE_WITH_COVERAGE_SHORTFALL",
        "target_frames": {"fitting": expected_fit, "development": expected_dev},
        "config_sha256": config_sha,
        "execution_seconds": time.monotonic() - started,
        "detector": {
            "implementation": "ultralytics",
            "version": ultralytics_version,
            "opencv_version": cv2.__version__,
            "weights": str(weights_path.relative_to(root)),
            "weights_sha256": weights_sha,
            "device": workload["device"],
            "imgsz": 1280,
            **workload["detector"],
            "new_downloads": False,
        },
        "totals": totals,
        "clips": clip_manifests,
        "fit_development_separation": "Fitting and development artifacts are stored in separate directories; development never pooled into fitting.",
        "limitations": [
            "Pixels and image-space candidate tracklets only; no verified player identity.",
            "Pitch coordinates and calibrated physical accuracy remain unsupported unless separately validated.",
            "Team affiliation and goalkeeper roles remain UNKNOWN.",
            "No RGB frames are persisted; only selected overlay diagnostics and structured detections are retained.",
        ],
    }
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(output / "run_state.json", {
        "schema_version": "football.video-ingestion-state.0.6",
        "status": manifest["status"],
        "config_sha256": config_sha,
        "detector_weights_sha256": weights_sha,
        "totals": totals,
    })
    write_json(output / "progress.json", {"status": manifest["status"], "totals": totals})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    output = args.output if args.output.is_absolute() else args.root / args.output
    result = run(config_path.resolve(), output.resolve(), root=args.root, resume=args.resume)
    print(json.dumps({"status": result["status"], "totals": result.get("totals"),
                      "output": str(output.resolve())}, indent=2))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
