"""One frozen narrow, annotation-assisted interpolation diagnostic, never forced deployment."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "outputs" / "backbone_v0_6_20261021" / "stage_b"
POLICY = {
    "region": "Released raw-image landmarks with x <= 1510 px, left penalty-area support only",
    "mapping": "Raw-image thin-plate spline, degree1, zero smoothing, no lens-parameter reuse",
    "cross_validation": "Three contiguous spatial pitch-width bands; no held-out landmark in plane fit",
    "acceptance": {"median_m_max": 2.0, "p90_m_max": 5.0, "minimum_inside_hull_withheld": 12,
                   "minimum_measured_blocks": 3},
    "original_full_panorama_gate": "NOT SATISFIED by this narrow protocol; original minimum50/3 bands preserved",
    "supported_domain": "Intersection of selected image support hull and x<=1510; never extrapolate",
    "continuous_segment": "Existing 117092 first-half detections, video seconds [240,300), 300 samples at5Hz",
    "fallback": "If rejected, no pitch observations or accepted mapping are emitted",
}


def evaluate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Preserve frozen regional experiment; inspect or validate its result")
    write_json(output / "policy.json", POLICY)
    previous = json.loads((PRIOR / "regional_mapping_117092.json").read_text(encoding="utf-8"))
    path = Path(previous["keypoints_path"])
    if file_digest(path) != previous["keypoints_sha256"]:
        raise ValueError("Released calibration points changed")
    raw = json.loads(path.read_text(encoding="utf-8"))
    image = np.asarray(list(raw.values()), np.float64)
    pitch = np.asarray([ast.literal_eval(key) for key in raw], np.float64)
    selected = np.flatnonzero(image[:, 0] <= 1510)
    blocks = np.array_split(selected[np.argsort(pitch[selected, 1], kind="stable")], 3)
    errors, rows = [], []
    for number, testing in enumerate(blocks):
        fitting = np.setdiff1d(selected, testing)
        interpolator = RBFInterpolator(image[fitting], pitch[fitting], kernel="thin_plate_spline", degree=1)
        predictions = interpolator(image[testing])
        inside = Delaunay(image[fitting]).find_simplex(image[testing]) >= 0
        residuals = np.linalg.norm(predictions - pitch[testing], axis=1)
        if not np.isfinite(residuals).all():
            raise FloatingPointError("Regional map generated nonfinite validation residual")
        errors.extend(residuals[inside].tolist())
        rows.append({"block": number, "fitting_indices": fitting.tolist(), "withheld_indices": testing.tolist(),
                     "inside_fit_hull": inside.tolist(), "pitch_errors_m": residuals.tolist(),
                     "within_hull_count": int(inside.sum()),
                     "within_hull_median_m": float(np.median(residuals[inside])) if inside.any() else None,
                     "within_hull_p90_m": float(np.quantile(residuals[inside], .9)) if inside.any() else None,
                     "all_withheld_median_m": float(np.median(residuals)),
                     "all_withheld_p90_m": float(np.quantile(residuals, .9))})
    median = float(np.median(errors)) if errors else None
    p90 = float(np.quantile(errors, .9)) if errors else None
    enough = len(errors) >= 12 and all(row["within_hull_count"] > 0 for row in rows)
    accepted = enough and median <= 2 and p90 <= 5
    result = {
        "status": "ACCEPTED_NARROW_INTERNAL_DIAGNOSTIC" if accepted else "BLOCKED_REGIONAL_VALIDATION",
        "keypoints_sha256": file_digest(path), "correspondences_in_selected_region": len(selected),
        "withheld_inside_fitting_hull": len(errors), "median_inside_hull_m": median, "p90_inside_hull_m": p90,
        "blocks": rows, "original_panorama_gate_status": "UNCHANGED_REJECTED",
        "independence": "Only this raw-image mapping excludes held-out points; no lens calibration reused. Released field points remain annotation-assisted, not new physical ground truth.",
        "unresolved": ["Sparse 2D support", "Footpoint/detection error not independently validated",
                       "Teams and goalkeeper roles unknown", "Pixel associations unverified",
                       "Calibration-valid landmark residual is not validated player tracking accuracy"],
        "projection_deployed": False,
    }
    # This program deliberately stops at the calibration gate; a passing result still
    # requires the separate detector-footpoint/identity acceptance before deployment.
    clip = PRIOR / "video_corpus" / "fitting" / "117092_1st_00240s_1200s"
    clip_manifest = json.loads((clip / "clip_manifest.json").read_text(encoding="utf-8"))
    stream = clip / "pixel_tracklets.jsonl"
    detections, samples, tracks, roi_detections = 0, 0, Counter(), 0
    review = []
    hull = Delaunay(image[selected])
    with stream.open(encoding="utf-8") as reader:
        for line in reader:
            row = json.loads(line)
            timestamp = row["time_from_video_start_s"]
            if timestamp >= 300:
                break
            if timestamp < 240:
                continue
            samples += 1
            for detection in row["detections"]:
                x1, _, x2, y2 = detection["bbox_xyxy_px"]
                point = np.asarray([[(x1 + x2) / 2, y2]], np.float64)
                detections += 1
                if point[0, 0] <= 1510 and hull.find_simplex(point)[0] >= 0:
                    roi_detections += 1
                    tracks[detection["tracklet_id"]] += 1
            if samples in (1, 150, 300):
                review.append({"timestamp_s": timestamp, "native_frame": row["frame_index"],
                               "detections": row["detections"], "pitch_positions": None,
                               "reason": result["status"] + "; detector positions not accepted"})
    if samples != 300:
        raise ValueError("Chosen continuous video segment is not 300 contiguous five-Hz records")
    result["continuous_segment"] = {
        "start_s": 240, "end_s_exclusive": 300, "samples": samples, "detections": detections,
        "inside_region_detections": roi_detections, "unique_unverified_roi_tracklets": len(tracks),
        "largest_roi_tracklet_samples": max(tracks.values(), default=0),
        "source_clip_manifest_sha256": file_digest(clip / "clip_manifest.json"),
        "pixel_tracklets_sha256": file_digest(stream), "video_sha256": clip_manifest["source_video_sha256"],
        "accepted_pitch_observations": 0,
    }
    write_json(output / "review_only.json", {"purpose": "Rejected calibration/identity review; NOT training labels",
                                           "records": review})
    write_json(output / "result.json", result)
    return result


def validate(output: Path) -> dict:
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    clip = PRIOR / "video_corpus" / "fitting" / "117092_1st_00240s_1200s"
    clip_manifest = json.loads((clip / "clip_manifest.json").read_text(encoding="utf-8"))
    segment = result["continuous_segment"]
    for path, digest in ((clip / "clip_manifest.json", segment["source_clip_manifest_sha256"]),
                         (clip / "pixel_tracklets.jsonl", segment["pixel_tracklets_sha256"]),
                         (ROOT / clip_manifest["source_video"], segment["video_sha256"])):
        if file_digest(path) != digest:
            raise ValueError(f"Regional diagnostic provenance changed: {path}")
    rows = []
    with (clip / "pixel_tracklets.jsonl").open(encoding="utf-8") as reader:
        for line in reader:
            row = json.loads(line)
            if row["time_from_video_start_s"] >= 300:
                break
            if row["time_from_video_start_s"] >= 240:
                rows.append(row)
    if (len(rows) != 300
            or not np.allclose([row["time_from_video_start_s"] for row in rows], 240 + np.arange(300) / 5)
            or [row["frame_index"] for row in rows] != list(range(6000, 7500, 5))):
        raise ValueError("Regional evidence is not a continuous minute at five Hz")
    measured = [error for block in result["blocks"]
                for error, inside in zip(block["pitch_errors_m"], block["inside_fit_hull"]) if inside]
    if (len(measured) != result["withheld_inside_fitting_hull"]
            or not np.isclose(np.median(measured), result["median_inside_hull_m"])
            or not np.isclose(np.quantile(measured, .9), result["p90_inside_hull_m"])):
        raise ValueError("Regional mapping metrics do not recompute")
    if result["projection_deployed"] or segment["accepted_pitch_observations"] != 0:
        raise ValueError("Rejected/undiagnosed footpoints were deployed")
    validation = {"status": "PASS", "capability_status": result["status"], "continuous_source_frames": 300,
                  "provenance_hashes_verified": True, "accepted_pitch_observations": 0,
                  "interpretation": "Blocked capability is explicitly preserved, not promoted to a passed bridge"}
    write_json(output / "validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("evaluate", "validate"), default="evaluate")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output) if args.phase == "evaluate" else validate(args.output), indent=2))


if __name__ == "__main__":
    main()
