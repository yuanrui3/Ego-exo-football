"""Independently verify the v0.6 streaming video corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.backbone.video_ingest_v06 import (
    _chunk_rows,
    _fingerprint,
    _sampled_frame_indices,
    _validate_config,
)
from src.trajectory_dataset import file_digest


def validate(config_path: Path, output: Path, *, root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha = file_digest(config_path)
    clips = _validate_config(config)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    previous_manifest_path = root / "outputs/backbone_v0_5_20261020/video/manifest.json"
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    for split, clip in clips:
        start, end = float(clip["start_s"]), float(clip["start_s"]) + int(clip["duration_s"])
        for previous in previous_manifest["clips"]:
            if (str(previous["match"]) == str(clip["match"])
                    and previous["half"] == clip["half"]):
                old_start = float(previous["start_s"])
                old_end = old_start + float(previous["duration_s"])
                if max(start, old_start) < min(end, old_end):
                    errors.append(
                        f"New {split} excerpt overlaps historical pilot: "
                        f"{clip['match']} {clip['half']}"
                    )
    if manifest.get("status") != "COMPLETE" or state.get("status") != "COMPLETE":
        errors.append("Ingestion run is not marked COMPLETE")
    if manifest.get("config_sha256") != config_sha or state.get("config_sha256") != config_sha:
        errors.append("Run configuration hash mismatch")
    manifest_by_id = {
        f"{row['split']}:{row['match']}:{row['half']}:{row['start_s']}": row
        for row in manifest.get("clips", [])
    }
    if len(manifest_by_id) != 8:
        errors.append(f"Expected eight unique clip manifests, found {len(manifest_by_id)}")

    totals = {
        "fitting_frames": 0,
        "fitting_unique_source_frames": 0,
        "development_frames": 0,
        "development_unique_source_frames": 0,
        "chunks": 0,
        "source_hashes_rechecked": 0,
    }
    expected_weights_sha = manifest.get("detector", {}).get("weights_sha256")
    for split, clip in clips:
        match, half = str(clip["match"]), str(clip["half"])
        duration = int(clip["duration_s"])
        start_s = float(clip["start_s"])
        key = f"{split}:{match}:{half}:{start_s}"
        record = manifest_by_id.get(key)
        if record is None:
            errors.append(f"Missing clip manifest: {key}")
            continue
        clip_folder = output / split / f"{match}_{half}_{int(start_s):05d}s_{duration:04d}s"
        clip_manifest_path = clip_folder / "clip_manifest.json"
        if not clip_manifest_path.is_file():
            errors.append(f"Missing clip manifest artifact: {clip_manifest_path}")
            continue
        clip_manifest = json.loads(clip_manifest_path.read_text(encoding="utf-8"))
        if clip_manifest != record:
            errors.append(f"Global/clip manifest mismatch: {key}")
        source = root / config["source_paths"]["video_root"] / match / f"{match}_panorama_{half}_half.mp4"
        if not source.is_file() or file_digest(source) != record.get("source_video_sha256"):
            errors.append(f"Source video missing or changed: {source}")
        totals["source_hashes_rechecked"] += 1
        indices = _sampled_frame_indices(
            start_s, duration, float(record["source_fps"]), int(config["workload"]["target_fps"]),
        )
        if record.get("inference_frames") != len(indices):
            errors.append(f"Wrong inference frame count: {key}")
        if record.get("unique_source_frames") != len(set(indices)):
            errors.append(f"Duplicate or missing native source frames: {key}")
        if record.get("source_frame_count") <= indices[-1]:
            errors.append(f"Excerpt crosses the native video boundary: {key}")
        rows = []
        chunk_size = int(config["workload"]["chunk_samples"])
        for chunk_index, begin in enumerate(range(0, len(indices), chunk_size)):
            expected = indices[begin:begin + chunk_size]
            chunk_fingerprint = _fingerprint({
                "schema": "football.video-chunk.0.6",
                "config_sha256": config_sha,
                "source_sha256": record["source_video_sha256"],
                "weights_sha256": expected_weights_sha,
                "clip_id": clip_folder.name,
                "chunk_index": chunk_index,
                "frame_indices": expected,
                "preprocessing": config["workload"]["detector"],
            })
            try:
                chunk_rows, _ = _chunk_rows(clip_folder, chunk_index, chunk_fingerprint, expected)
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"Invalid chunk {chunk_index} in {key}: {error}")
                continue
            if chunk_rows is None:
                errors.append(f"Missing completed chunk {chunk_index} in {key}")
                continue
            rows.extend(chunk_rows)
            totals["chunks"] += 1
        if [row["frame_index"] for row in rows] != indices:
            errors.append(f"Chunk rows do not reproduce the selected timeline: {key}")
        for row in rows:
            if row.get("pitch_positions") is not None or row.get("team_assignments") != "UNKNOWN":
                errors.append(f"Unsupported metadata was populated in {key} frame {row['frame_index']}")
        observations = clip_folder / "pixel_tracklets.jsonl"
        if not observations.is_file() or file_digest(observations) != record.get("observations_sha256"):
            errors.append(f"Tracklet observations missing or changed: {key}")
        else:
            with observations.open(encoding="utf-8") as stream:
                observed_count = sum(bool(line.strip()) for line in stream)
            if observed_count != len(indices):
                errors.append(f"Tracklet observation count mismatch: {key}")
        if len(record.get("overlays", [])) < 3:
            errors.append(f"Expected first/middle/last overlays: {key}")
        if record.get("tracklet_status") != "UNVERIFIED_IMAGE_SPACE_CANDIDATES":
            errors.append(f"Tracklet candidate status overclaims identity: {key}")
        if record.get("pitch_projection") != "UNSUPPORTED":
            errors.append(f"Unsupported pitch projection status changed: {key}")
        totals[f"{split}_frames"] += len(indices)
        totals[f"{split}_unique_source_frames"] += len(set(indices))

    expected_totals = {
        "fitting_frames": 36000,
        "fitting_unique_source_frames": 36000,
        "development_frames": 6000,
        "development_unique_source_frames": 6000,
    }
    checks = {
        "workload_counts": all(totals.get(key) == value for key, value in expected_totals.items()),
        "clip_count_and_role_split": len(clips) == 8
        and sum(split == "fitting" for split, _ in clips) == 6
        and sum(split == "development" for split, _ in clips) == 2,
        "all_source_and_output_hashes": not any(
            "Source video missing or changed" in error
            or "observations missing or changed" in error
            or "Invalid chunk" in error
            for error in errors
        ),
        "no_unverified_physical_or_identity_claims": not any(
            "Unsupported metadata" in error
            or "overclaims identity" in error
            or "pitch projection status changed" in error
            for error in errors
        ),
    }
    return {
        "schema_version": "football.backbone-v0.6-video-validation.1",
        "status": "PASS" if all(checks.values()) and not errors else "FAIL",
        "config_path": str(config_path.resolve()),
        "output": str(output.resolve()),
        "totals": totals,
        "checks": checks,
        "errors": errors,
        "unique_source_minutes": {
            "fitting": totals["fitting_unique_source_frames"] / 300,
            "development": totals["development_unique_source_frames"] / 300,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/backbone_v0_6_20261021/stage_b/video_validation.json"),
    )
    args = parser.parse_args()
    report = validate(args.config.resolve(), args.output.resolve(), root=args.root.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
