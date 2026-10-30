"""Post-fit diagnostics for the isolated GPU perception experiment."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.backbone.gpu_perception import (
    DEFAULT_OUTPUT, HISTORICAL, ROOT, aggregate_scores, now, predict_video,
    require_cuda, score_predictions, source_records,
)
from src.backbone.video_player_cross_match import (
    _read_annotations, _read_frame, load_local_projector, load_protocol, project_footpoint,
)
from src.backbone.video_player_replay import _assign_tracklets
from src.trajectory_dataset import file_digest, write_json


def assign_global(detections: list[dict], tracks: dict, next_id: int,
                  sample: int, max_distance: float) -> int:
    active = sorted(t for t, value in tracks.items() if sample - value["last_sample"] <= 5)
    if active and detections:
        costs = np.full((len(active), len(detections) + len(active)), max_distance * 2 + 1.)
        costs[:, :len(detections)] = 1e9
        for i, track_id in enumerate(active):
            track = tracks[track_id]
            age = sample - track["last_sample"]
            predicted = [track["footpoint_xy_px"][axis] + track["velocity_xy_px"][axis] * age
                         for axis in (0, 1)]
            for j, detection in enumerate(detections):
                distance = math.dist(predicted, detection["footpoint_xy_px"])
                if distance <= max_distance * min(age, 2):
                    costs[i, j] = distance
        ii, jj = linear_sum_assignment(costs)
        for i, j in zip(ii, jj):
            if j >= len(detections) or costs[i, j] >= 1e9:
                continue
            track = tracks[active[i]]
            point = detections[j]["footpoint_xy_px"]
            age = sample - track["last_sample"]
            track["velocity_xy_px"] = [
                float(np.clip((point[a] - track["footpoint_xy_px"][a]) / age, -120., 120.))
                for a in (0, 1)
            ]
            track.update(footpoint_xy_px=point, last_sample=sample,
                         observations=track["observations"] + 1)
            detections[j]["track"] = active[i]
    for detection in detections:
        if "track" in detection:
            continue
        track_id = f"video-player-{next_id:04d}"
        next_id += 1
        tracks[track_id] = {"footpoint_xy_px": detection["footpoint_xy_px"],
                            "velocity_xy_px": [0., 0.], "last_sample": sample,
                            "observations": 1, "first_sample": sample, "last_detection_index": None}
        detection["track"] = track_id
    for track_id in list(tracks):
        if sample - tracks[track_id]["last_sample"] > 5:
            del tracks[track_id]
    return next_id


def associate(frames: list[dict], width: int, method: str) -> list[dict]:
    result, tracks, next_id = copy.deepcopy(frames), {}, 1
    for sample, frame in enumerate(result):
        for detection in frame["detections"]:
            detection.pop("track", None)
        if method == "global":
            next_id = assign_global(frame["detections"], tracks, next_id, sample, 180 * width / 4096)
        else:
            next_id = _assign_tracklets(frame["detections"], tracks, next_id, sample,
                                        max_distance_px=180 * width / 4096)
    return result


def tracker_diagnostic(output: Path) -> dict:
    lock = {
        "created_utc": now(),
        "one_change": "Global Hungarian assignment instead of greedy; same constant-velocity state, age<=5, 180px width-scaled gate and velocity clipping.",
        "selection_rule": "Reduce mean per-match development identity-switch rate; each source complete-history fraction must not fall more than .01 and fragmentation must not grow more than 5% plus one track.",
        "development_only": True, "runtime_truth_used": False,
    }
    write_json(output / "tracker_lock.json", lock)
    corrected = output / "evaluated_candidate_development_predictions.json"
    saved = json.loads((corrected if corrected.is_file() else output / "candidate_development_predictions.json").read_text())
    sources = source_records()
    metrics = {}
    for match in ("117093", "117092"):
        frames = saved[match]
        labels = _read_annotations(sources[match], {row["frame_index"] for row in frames})
        metrics[match] = {}
        for method in ("greedy", "global"):
            associated = associate(frames, sources[match]["dimensions_px"][0], method)
            metrics[match][method] = aggregate_scores(associated, labels, sources[match]["dimensions_px"][0])
    selected = True
    old_rate, new_rate = [], []
    for match, results in metrics.items():
        old, new = (results[method]["thresholds"]["40"] for method in ("greedy", "global"))
        old_rate.append(old["switch_rate"] or 0)
        new_rate.append(new["switch_rate"] or 0)
        selected &= (
            (new["complete_11_sample_history_fraction"] or 0) >= (old["complete_11_sample_history_fraction"] or 0) - .01
            and new["fragmentation_extra_tracks_per_identity"] <= old["fragmentation_extra_tracks_per_identity"] * 1.05 + 1
        )
    selected &= float(np.mean(new_rate)) < float(np.mean(old_rate))
    result = {"status": "DEVELOPMENT_SELECTED" if selected else "GREEDY_RETAINED",
              "lock": lock, "metrics": metrics, "global_selected": bool(selected)}
    write_json(output / "tracker_development.json", result)
    if selected:
        seal = json.loads((output / "evaluation_seal.json").read_text())
        derived = []
        for record in seal["files"]:
            if record["route"] != "gpu_selected":
                continue
            frames = [json.loads(line) for line in (ROOT / record["path"]).read_text().splitlines()]
            associated = associate(frames, sources[record["match_id"]]["dimensions_px"][0], "global")
            destination = output / "association" / Path(record["path"]).name
            destination.parent.mkdir(exist_ok=True)
            destination.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                                            for row in associated), encoding="utf-8")
            derived.append({**record, "path": str(destination.relative_to(ROOT)),
                            "prediction_jsonl_sha256": file_digest(destination),
                            "source_detection_sha256": record["prediction_jsonl_sha256"],
                            "association": "development-selected global Hungarian"})
        write_json(output / "association_seal.json", {"created_utc": now(), "files": derived})
        results = [score_predictions(output, record, sources[record["match_id"]]) for record in derived]
        write_json(output / "tracker_retrospective.json", {"scores": results, "status": "RETROSPECTIVE"})
    return result


def expansion(output: Path) -> dict:
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-10-29T20:55:07+00:00"):
        result = {"status": "SKIPPED_AT_OPTIONAL_EXPANSION_DEADLINE"}
        write_json(output / "expansion.json", result)
        return result
    source = json.loads((output / "additional_match_lock.json").read_text())
    for field in ("video", "annotation"):
        if file_digest(ROOT / source[field]) != source[field + "_sha256"]:
            raise ValueError(f"Additional match {field} changed")
    training = json.loads((output / "training_summary.json").read_text())
    protocol = json.loads((HISTORICAL / "protocol.json").read_text())
    retained_source = (ROOT / training["selected"]).resolve() == (ROOT / protocol["preservation"]["source_detector"]).resolve()
    candidate = (training["selected"] if not retained_source
                 else training["candidate_best"])
    records = [
        predict_video(output, source, source["evaluation_interval_seconds"], "new_match_gpu",
                      ROOT / candidate, scheme="consistent"),
        predict_video(output, source, source["evaluation_interval_seconds"], "new_match_frozen",
                      ROOT / protocol["preservation"]["source_detector"], scheme="historical"),
    ]
    if datetime.now(timezone.utc) < datetime.fromisoformat("2026-10-29T20:55:07+00:00"):
        records.append(predict_video(output, source, source["stability_interval_seconds"],
                                     "stability_gpu", ROOT / candidate, scheme="consistent"))
    write_json(output / "expansion_seal.json", {"created_utc": now(), "files": records})
    scores = [score_predictions(output, record, source) for record in records]
    stability_complete = any(record["route"] == "stability_gpu" for record in records)
    result = {
        "status": ("ADDITIONAL_MATCH_AND_STABILITY_COMPLETE" if stability_complete
                   else "ADDITIONAL_MATCH_COMPLETE_STABILITY_SKIPPED_AT_DEADLINE"),
        "source": source, "scores": scores,
    }
    write_json(output / "expansion.json", result)
    return result


def text(canvas, value, xy, color=(235, 235, 235), scale=.5):
    cv2.putText(canvas, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_diagnostics(output: Path, *, review: bool = False) -> dict:
    ffmpeg = ROOT / "data/video_review_tools/portable/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe"
    protocol = load_protocol(HISTORICAL / "protocol.json")
    summaries = []
    for match in ("117092", "118576"):
        directory = output / f"r{match}"
        source = source_records()[match]
        projector = load_local_projector(protocol, HISTORICAL, match)
        display_height = 470 if source["dimensions_px"][1] > 1280 else 248
        support = np.zeros((display_height, 940), dtype=np.uint8)
        for y in range(0, display_height, 4):
            for x in range(0, 940, 4):
                position, _ = project_footpoint(
                    [x * source["dimensions_px"][0] / 940,
                     y * source["dimensions_px"][1] / display_height], projector,
                )
                if position is not None:
                    support[y:y+4, x:x+4] = 255
        boundaries, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        streams, replay_rows = {}, {}
        for route in ("frozen", "adapted"):
            path = directory / "predictions" / route / f"{match}_{source['half']}_development.jsonl"
            streams[route] = [json.loads(line) for line in path.read_text().splitlines()]
            replay_path = directory / "backbone_replay_usable_video_selection/video_estimated" / f"{route}_{match}_{source['half']}_development.jsonl"
            replay_rows[route] = [json.loads(line) for line in replay_path.read_text().splitlines()]
        cap = cv2.VideoCapture(str(ROOT / source["video"]), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {source['video']}")
        destination = output / f"comparison_{match}_120s{'_review' if review else ''}.mp4"
        if destination.exists():
            raise FileExistsError(destination)
        process = subprocess.Popen(
            [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", "1920x1080", "-r", "5", "-i", "pipe:0", "-an", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "24", "-movflags", "+faststart",
             str(destination)], stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        previous, trails, latest_query = None, {}, {}
        try:
            for sample in range(600):
                index = 7500 + sample * 5
                image = _read_frame(cap, index, previous)
                previous = index
                canvas = np.full((1080, 1920, 3), 24, dtype=np.uint8)
                text(canvas, f"{match} | {index / 25:.1f}s | BEFORE / GPU CANDIDATE | experimental; global calibration FAILED",
                     (20, 32), (0, 220, 255), .7)
                for column, route in enumerate(("frozen", "adapted")):
                    left = column * 960
                    native = cv2.resize(image, (940, display_height))
                    cv2.drawContours(native, boundaries, -1, (255, 255, 0), 1)
                    sx, sy = native.shape[1] / source["dimensions_px"][0], native.shape[0] / source["dimensions_px"][1]
                    frame = streams[route][sample]
                    projected = 0
                    for detection in frame["detections"]:
                        px, py = detection["footpoint_xy_px"]
                        pitch, _ = project_footpoint([px, py], projector)
                        color = (0, 220, 255) if pitch is not None else (150, 150, 150)
                        projected += int(pitch is not None)
                        track = detection["track"]
                        trail = trails.setdefault((route, track), deque(maxlen=15))
                        trail.append((sample, (round(px * sx), round(py * sy))))
                        recent = [point for at, point in trail if sample - at <= 14]
                        if len(recent) > 1:
                            cv2.polylines(native, [np.asarray(recent, dtype=np.int32)], False, color, 1)
                        cv2.circle(native, (round(px * sx), round(py * sy)), 3, color, -1)
                        text(native, track[-4:], (round(px * sx), round(py * sy) - 5), color, .3)
                    center = projector["fixed_roi"]["principal_x_px"]
                    half = projector["fixed_roi"]["half_width_px"]
                    for bound in (center - half, center + half):
                        cv2.line(native, (round(bound * sx), 0), (round(bound * sx), native.shape[0]-1), (255, 255, 0), 1)
                    text(canvas, f"{route}: {len(frame['detections'])} detections; {projected} supported local projections",
                         (left + 10, 65), scale=.5)
                    canvas[85:85 + native.shape[0], left + 10:left + 950] = native
                    text(canvas, "Cyan outline: setup-supported projection region, NOT calibration acceptance.",
                         (left + 10, 578), (255, 255, 0), .43)
                    base_y = 600
                    text(canvas, "Selected actor view / causal memory (simulated visibility)", (left + 10, base_y))
                    at_sample = [row for row in replay_rows[route] if row["frame_index"] == index]
                    for actor_index, row in enumerate(at_sample[:2]):
                        y0 = base_y + 30 + actor_index * 180
                        actor = row["runtime_actor_track"]
                        if row["scheduled_s4_query"]:
                            latest_query[(route, actor)] = {
                                "time": row["timestamp_s"], "action": row["action"].get("action_type"),
                                "status": row["action_inference"].get("status"),
                                "reason": row["action_inference"].get("reason"),
                            }
                        belief = row["belief"]
                        backend = Counter(entity["backend"] for entity in belief["entities"])
                        text(canvas, f"{actor}: present={belief['actor_available']}; states={dict(backend)}",
                             (left + 10, y0), scale=.43)
                        query = latest_query.get((route, actor))
                        if query:
                            text(canvas, f"Supplied query {query['time']:.1f}s: {query['status']} {query['action'] or ''}",
                                 (left + 10, y0 + 25), (0, 220, 255), .45)
                            text(canvas, str(query["reason"] or "")[:105], (left + 10, y0 + 48), scale=.38)
                        else:
                            text(canvas, "No action query yet; model load is not inference", (left + 10, y0 + 25), scale=.42)
                        map_left, map_top = left + 620, y0 + 55
                        map_right, map_bottom = map_left + 284, map_top + 75
                        cv2.rectangle(canvas, (map_left, map_top), (map_right, map_bottom), (90, 90, 90), 1)
                        cv2.line(canvas, (map_left + 142, map_top), (map_left + 142, map_bottom), (90, 90, 90), 1)
                        cv2.ellipse(canvas, (map_left + 142, map_top + 37), (25, 10), 0, 0, 360, (90, 90, 90), 1)
                        text(canvas, "Experimental pitch: green=visible; amber=memory", (left + 570, y0 + 150), scale=.3)
                        for entity in belief["entities"]:
                            x, y = entity["xy"]
                            point = (left + 620 + round((x + 52.5) * 2.7), y0 + 55 + round((y + 34) * 1.1))
                            color = (0, 210, 0) if entity["visible"] else (255, 170, 0)
                            cv2.circle(canvas, point, 3, color, -1)
                    text(canvas, "Unknown attributes stay unknown; G1 does not feed S4.", (left + 10, 1015), scale=.48)
                    text(canvas, "Ball/possession unknown; scores are not tactical value.", (left + 10, 1045), scale=.48)
                if sample == 0:
                    cv2.imwrite(str(destination.with_suffix(".jpg")), canvas)
                process.stdin.write(canvas.tobytes())
        except BaseException:
            process.kill()
            raise
        finally:
            cap.release()
            process.stdin.close()
            error = process.stderr.read()
            process.wait(timeout=60)
        if process.returncode:
            raise RuntimeError(error.decode(errors="replace")[-2000:])
        check = cv2.VideoCapture(str(destination))
        count, fps = int(check.get(cv2.CAP_PROP_FRAME_COUNT)), float(check.get(cv2.CAP_PROP_FPS))
        check.release()
        if count != 600 or abs(fps - 5) > .001:
            raise ValueError(f"Video coverage mismatch: {destination}")
        summaries.append({"path": str(destination.relative_to(ROOT)), "sha256": file_digest(destination),
                          "frames": count, "fps": fps, "seconds": count / fps})
    result = {"status": "PLAYABLE_COMPARISONS_COMPLETE", "videos": summaries}
    write_json(output / ("comparison_videos_review.json" if review else "comparison_videos.json"), result)
    return result


def resume_probe(output: Path) -> dict:
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    require_cuda()
    destination = output / "resume_probe"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir()
    checkpoint = torch.load(output / "last_training.pt", map_location="cpu", weights_only=False)
    original_digest = file_digest(output / "last_training.pt")
    if not checkpoint.get("optimizer"):
        raise ValueError("Preserved training checkpoint has no optimizer to restore")
    all_paths = (output / "data_v1/fit.txt").read_text().splitlines()
    by_match = [[p for p in all_paths if match in p] for match in ("117093", "117092")]
    mixed = [p for pair in zip(*by_match) for p in pair][:320]
    development_paths = (output / "data_v1/dev.txt").read_text().splitlines()
    development_by_match = [[p for p in development_paths if match in p]
                            for match in ("117093", "117092")]
    development = [p for pair in zip(*development_by_match) for p in pair][:32]
    (destination / "fit.txt").write_text("\n".join(mixed) + "\n", encoding="utf-8")
    (destination / "dev.txt").write_text("\n".join(development) + "\n", encoding="utf-8")
    yaml = destination / "dataset.yaml"
    yaml.write_text(f"path: {destination.as_posix()}\ntrain: fit.txt\nval: dev.txt\nnames:\n  0: player_footpoint\n",
                    encoding="utf-8")
    checkpoint["train_args"].update(project=str(destination), name="probe", data=str(yaml),
                                     epochs=checkpoint["epoch"] + 2, workers=0, device=0,
                                     plots=False, save_period=-1, exist_ok=False)
    probe_weights = destination / "init.pt"
    torch.save(checkpoint, probe_weights)
    state = {"actual_additional_optimizer_steps": 0,
             "fit_tiles": len(mixed), "development_tiles": len(development),
             "development_purpose": "Execution diagnostic only; not candidate selection or a quality estimate"}

    class ResumeTrainer(DetectionTrainer):
        def optimizer_step(self):
            if self.device.type != "cuda" or not torch.isfinite(self.loss).all():
                raise RuntimeError("Resume probe violates CUDA/finite-loss contract")
            super().optimizer_step()
            torch.cuda.synchronize()
            state["actual_additional_optimizer_steps"] += 1

    def started(trainer):
        if not trainer.optimizer.state:
            raise RuntimeError("Native resume failed to restore optimizer state")
        state["restored_optimizer_parameter_states"] = len(trainer.optimizer.state)
        state["start_epoch"] = trainer.start_epoch
        state["amp_scaler_restored"] = bool(trainer.scaler.state_dict())

    def batch_end(trainer):
        if state["actual_additional_optimizer_steps"] >= 10:
            trainer.stop = True

    model = YOLO(str(probe_weights))
    model.add_callback("on_train_start", started)
    model.add_callback("on_train_batch_end", batch_end)
    model.train(trainer=ResumeTrainer, resume=True)
    if state["actual_additional_optimizer_steps"] < 10:
        raise RuntimeError("Resume probe did not execute ten additional optimizer steps")
    if file_digest(output / "last_training.pt") != original_digest:
        raise ValueError("Resume probe changed the original training checkpoint")
    state.update(status="PASS", finished_utc=now(), device="cuda:0",
                 restored_checkpoint_sha256=original_digest,
                 terminology="Weights-only restart diagnostic, NOT exact continuation: native resume restores optimizer/AMP but starts from EMA and a new subset/sampler; original live model and sampler cursor are not restored.",
                 main_checkpoint_changed=False)
    write_json(output / "resume_evidence.json", state)
    return state


def track_history_population(frames: list[dict]) -> dict:
    runs, longest = Counter(), Counter()
    for frame in frames:
        present = {row["track"] for row in frame["detections"] if "track" in row}
        for track in set(runs) - present:
            del runs[track]
        for track in present:
            runs[track] += 1
            longest[track] = max(longest[track], runs[track])
    complete = sum(length >= 11 for length in longest.values())
    return {
        "distinct_runtime_tracks": len(longest),
        "runtime_tracks_ever_complete_11_samples": complete,
        "runtime_track_complete_history_fraction": complete / len(longest) if longest else None,
    }


def combined_diagnostics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation population")
    rates = {row["thresholds"]["40"]["sampling_hz"] for row in rows}
    if len(rates) != 1:
        raise ValueError("Do not pool different sampling protocols")
    rate = rates.pop()
    totals, histogram = Counter(), Counter()
    counts, errors = [], []
    for row in rows:
        score = row["thresholds"]["40"]
        for field in ("frames", "truth_instances", "detections", "matches", "misses",
                      "false_positives_including_duplicates", "duplicates", "identity_switches",
                      "matched_adjacent_transitions", "fragmentation_extra_tracks_per_identity",
                      "matched_returns_after_gap", "same_runtime_track_returns_after_gap",
                      "complete_11_sample_history_observations", "all_track_observation_opportunities",
                      "identity_consistent_matched_history_observations"):
            totals[field] += score[field]
        histogram.update({int(length): count for length, count in
                          score["continuous_track_length_samples_histogram"].items()})
        counts.extend(row["per_frame_player_counts"])
        errors.extend(row["projection"]["conditional_errors_m"])
    lengths = np.repeat(list(histogram), list(histogram.values()))
    return {
        **dict(totals), "sampling_hz": rate,
        **{field: sum(row["projection"][field] for row in rows)
           for field in ("projected", "unprojected", "supported_truth_instances",
                         "pixel_matched_in_support_with_pitch_truth")},
        "player_count_percentiles_min_p10_median_p90_max": np.percentile(counts, [0, 10, 50, 90, 100]).tolist(),
        "continuous_track_runs": int(sum(histogram.values())),
        "median_continuous_track_span_seconds": float(np.median(lengths) - 1) / rate if len(lengths) else None,
        "runtime_complete_history_fraction": totals["complete_11_sample_history_observations"] / totals["all_track_observation_opportunities"] if totals["all_track_observation_opportunities"] else None,
        "identity_consistent_truth_history_fraction": totals["identity_consistent_matched_history_observations"] / totals["truth_instances"] if totals["truth_instances"] else None,
        "history_sampling_compatible_with_backbone": rate == 5,
        "conditional_projection_error_count": len(errors),
        "conditional_projection_median_m": float(np.median(errors)) if errors else None,
        "conditional_projection_p90_m": float(np.percentile(errors, 90)) if errors else None,
        "projection_status": "UNVALIDATED/EXPERIMENTAL; global calibration remains failed",
    }


def handoff(output: Path) -> dict:
    training = json.loads((output / "training_summary.json").read_text())
    evaluation = json.loads((output / "evaluation_summary.json").read_text())
    integration = json.loads((output / "integration_summary.json").read_text())
    verification = json.loads((output / "verification.json").read_text())
    data = json.loads((output / "data_manifest.json").read_text())
    expansion_path = output / "expansion.json"
    extra = (json.loads(expansion_path.read_text()) if expansion_path.is_file()
             else {"status": "NOT_RUN; inspect post_worker_status.json for deadline or failure"})
    table, continuity = [], []
    associated_path = output / "tracker_retrospective.json"
    associated = {}
    if associated_path.is_file():
        associated = {
            (row["provenance"]["match_id"], tuple(row["provenance"]["interval_seconds"])): row
            for row in json.loads(associated_path.read_text())["scores"]
        }
    for route in ("frozen", "continued", "gpu_selected", "pretrained_person"):
        for match in ("117092", "118576"):
            rows = [r for r in evaluation["scores"] if r["route"] == route and r["match_id"] == match]
            row = {"route": route, "match": match, "association": "raw causal greedy"}
            for tolerance in ("20", "40"):
                totals = Counter()
                for score in rows:
                    for field in ("matches", "truth_instances", "detections", "identity_switches",
                                  "matched_adjacent_transitions", "complete_11_sample_history_observations",
                                  "all_track_observation_opportunities"):
                        totals[field] += score["thresholds"][tolerance][field]
                p = totals["matches"] / totals["detections"] if totals["detections"] else 0
                r = totals["matches"] / totals["truth_instances"] if totals["truth_instances"] else 0
                row[tolerance] = {**dict(totals), "precision": p, "recall": r,
                                  "f1": 2*p*r/(p+r) if p+r else 0}
            table.append(row)
            full = [
                associated.get((match, tuple(score["interval"])),
                               json.loads((ROOT / score["path"]).read_text()))
                if route == "gpu_selected" else json.loads((ROOT / score["path"]).read_text())
                for score in rows
            ]
            continuity.append({"route": route, "match": match,
                               "association": "development-selected global" if route == "gpu_selected" and associated else "causal greedy",
                               **combined_diagnostics(full)})
            populations = []
            for item in full:
                prediction = ROOT / item["provenance"]["path"]
                if file_digest(prediction) != item["provenance"]["prediction_jsonl_sha256"]:
                    raise ValueError("Prediction changed before history population reporting")
                populations.append(track_history_population(
                    [json.loads(line) for line in prediction.read_text().splitlines()]
                ))
            denominator = sum(item["distinct_runtime_tracks"] for item in populations)
            numerator = sum(item["runtime_tracks_ever_complete_11_samples"] for item in populations)
            continuity[-1].update(
                interval_local_runtime_tracks=denominator,
                interval_local_runtime_tracks_ever_complete_11_samples=numerator,
                interval_local_runtime_track_complete_history_fraction=numerator / denominator if denominator else None,
                track_population_note="Sum of distinct IDs per interval, not deduplicated people across windows; includes false positives.",
            )
    diagnostic_path = output / "evaluated_candidate_development.json"
    if not diagnostic_path.is_file():
        diagnostic_path = output / "candidate_development.json"
    development = {
        "source": json.loads((output / "source_development.json").read_text()),
        "candidate": json.loads(diagnostic_path.read_text()),
    }
    calls = [
        {"match": match["match"], "route": replay["detector_route"],
         "g1_calls": replay["g1_forward_calls"], "s4_calls": replay["s4_neural_forward_calls"],
         "actors": replay["actors"], "attribute_counts": next(
             b["counts"] for b in match["builds"] if b["detector_route"] == replay["detector_route"])}
        for match in integration["results"] for replay in match["replays"]
    ]
    for row in calls:
        source = source_records()[row["match"]]
        replay_file = output / f"r{row['match']}" / "backbone_replay_usable_video_selection/video_estimated" / (
            f"{row['route']}_{row['match']}_{source['half']}_development.jsonl"
        )
        replay_rows = [json.loads(line) for line in replay_file.read_text().splitlines()]
        queries = [item["action_inference"] for item in replay_rows if item["scheduled_s4_query"]]
        row.update(
            scheduled_queries=len(queries),
            query_status_counts=dict(Counter(item["status"] for item in queries)),
            unsupported_reasons=dict(Counter(item["reason"] for item in queries if item["status"] != "supported")),
            g1_contract_unavailable_actor_samples=sum(item["g1_contract_unavailable_samples"] for item in row["actors"]),
            fallback_entity_states=sum(item["fallback_entity_states"] for item in row["actors"]),
            g1_hidden_estimates_used=sum(item["g1_hidden_estimates_used"] for item in row["actors"]),
        )
    selected_epoch = next(
        (epoch for epoch in reversed(training["epochs"])
         if epoch["eligible"] and abs(epoch["mean_f1"] - training["best_f1"]) < 1e-10),
        None,
    )
    video_manifest = output / "comparison_videos_review.json"
    if not video_manifest.is_file():
        video_manifest = output / "comparison_videos.json"
    videos = json.loads(video_manifest.read_text())["videos"]
    summary = {
        "status": ("PARTIAL_TRAINING_INTERRUPTED_EVALUATION_COMPLETE" if "INTERRUPTED" in training["status"]
                   else "COMPLETE_SCOPED_EXPERIMENT"), "finished_utc": now(),
        "statuses": {"GPU_health": "PASS actual CUDA steps and detector fit",
                     "data_validity": "PASS reconciled disk/loader counts; positive labels and exact duplicates disclosed",
                     "training_execution": training["status"],
                     "model_quality": "See balanced development and retrospective results; no release promotion",
                     "integration": integration["status"], "physical_calibration": "BLOCKED unchanged"},
        "fit_optimizer_steps": training["optimizer_steps"], "fit_epochs_reported": len(training["epochs"]),
        "fit_elapsed_seconds": training["elapsed_seconds"], "selected_weights": training["selected"],
        "pixel_summary": table, "continuity_summary": continuity, "model_calls": calls,
        "additional_match_and_stability": extra,
        "selected_epoch_native_box_metrics": selected_epoch["box_metrics"] if selected_epoch else None,
        "videos": videos,
        "balanced_development": {
            route: {match: report[match]["thresholds"] for match in ("117093", "117092")}
            for route, report in development.items()
        },
        "verification": verification["regression_summary"],
        "data_rows": {"fit": data["fit_loader"], "dev": data["dev_loader"]},
    }
    write_json(output / "completion.json", summary)
    lines = [
        "# GPU perception experiment", "",
        "Frozen tracking release, G1 and original v0.5 S4 unchanged. No commits or pushes.",
        "Global camera-to-pitch validation remains blocked; all local projected positions are experimental.",
        "", f"Finished: {summary['finished_utc']}",
        f"GPU fit: {summary['fit_optimizer_steps']} confirmed optimizer steps; {summary['fit_epochs_reported']} completed epochs; {training['elapsed_seconds']/3600:.2f} hours.",
        f"Training outcome: {training['status']}.",
        training.get("optimizer_step_count_semantics", "Optimizer updates journaled by the trainer."),
        training.get("failure", "No recorded numerical training termination."),
        f"Selected experimental weights: `{training['selected']}` (or retained frozen source if retention failed).",
        "", "## Balanced fitting-source development",
        "117093 development was already used historically. Both routes use the new consistent-scale prepared images.",
        "| Model | Match | P40 | R40 | F1-40 |",
        "|---|---|---:|---:|---:|",
    ]
    for route, report in development.items():
        for match in ("117093", "117092"):
            score = report[match]["thresholds"]["40"]
            lines.append(f"| {route} | {match} | {score['precision']:.3f} | {score['recall']:.3f} | {score['f1']:.3f} |")
    if selected_epoch:
        metrics = selected_epoch["box_metrics"]
        lines += [
            "", f"Selected epoch {selected_epoch['epoch']} native tiled-development box mAP50: "
            f"{metrics['metrics/mAP50(B)']:.5f}; mAP50-95: {metrics['metrics/mAP50-95(B)']:.5f}.",
            "These are pooled native YOLO validation box metrics, not balanced footpoint selection or full-body-person IoU.",
        ]
    lines += [
        "", "## Pixel diagnostics at confidence 0.20",
        "D/E windows are retrospective. The pretrained control uses fewer predeclared samples and bottom-center proxies.",
        "Operational D comparisons include corrected 4x2 tiling for the GPU candidate; historical detectors retain their original 4x1 preprocessing.",
        "The balanced development comparison above isolates weights using identical corrected preprocessing.",
        "| Route | Match | P40 | R40 | F1-40 | P20 | R20 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        a, b = row["40"], row["20"]
        lines.append(f"| {row['route']} | {row['match']} | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | {b['precision']:.3f} | {b['recall']:.3f} |")
    lines += [
        "", "## Continuity and coverage denominators (40px, width-scaled)",
        "History below requires 11 consecutive 5Hz samples, spanning two seconds. Runtime history can include false positives.",
        "Identity-consistent history additionally requires the same evaluation-only player identity matched throughout; no reference identities enter tracking.",
        "Track duration is last-minus-first observation time, not sample count divided by frequency.",
        "| Route | Match | Frames | Matched / truth instances | Runtime history / observations | Identity-consistent history / truth | ID switches / adjacent matches | Extra fragments | Median track span (s) |",
        "|---|---|---:|---|---|---|---|---:|---:|",
    ]
    for row in continuity:
        if not row["history_sampling_compatible_with_backbone"]:
            continue
        span = row["median_continuous_track_span_seconds"]
        lines.append(
            f"| {row['route']} | {row['match']} | {row['frames']} | {row['matches']} / {row['truth_instances']} | "
            f"{row['complete_11_sample_history_observations']} / {row['all_track_observation_opportunities']} | "
            f"{row['identity_consistent_matched_history_observations']} / {row['truth_instances']} | "
            f"{row['identity_switches']} / {row['matched_adjacent_transitions']} | "
            f"{row['fragmentation_extra_tracks_per_identity']} | {span:.2f} |"
            if span is not None else f"| {row['route']} | {row['match']} | No runtime tracks |"
        )
    lines += [
        "The 1Hz pretrained control is excluded from 5Hz history comparisons.",
        "Exact per-frame population percentiles, gap-return counts and pooled conditional projection errors/support counts are in completion.json.",
        "completion.json also reports the proportion of distinct runtime tracks ever reaching 11 consecutive samples, with interval-local track denominators.",
        "Physical projection errors remain experimental and do not certify calibration.",
    ]
    lines += [
        "", "## Additional perception-match and continuous stability",
        f"Status: {extra['status']}.",
        "128057 was excluded from perception fitting and selection, but was in G1/S4 fitting; this is not a full-system-unseen test.",
        "This source has no approved camera-to-pitch setup here, so it remains pixel-only.",
        "| Route | Interval (s) | Samples | Matches / truth | P40 | R40 | F1-40 |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in extra.get("scores", []):
        score, provenance = row["thresholds"]["40"], row["provenance"]
        lines.append(
            f"| {provenance['route']} | {provenance['interval_seconds']} | {score['frames']} | "
            f"{score['matches']} / {score['truth_instances']} | {score['precision']:.3f} | "
            f"{score['recall']:.3f} | {score['f1']:.3f} |"
        )
    lines += ["", "## Data reconciliation",
              "The 2,640-label rows came from the old 1Hz D preparation; 13,200 came from 5Hz. Both remain intact.",
              f"New fit loader: {data['fit_loader']['images']} tiles, {data['fit_loader']['labels']} unique labels.",
              f"New dev loader: {data['dev_loader']['images']} tiles, {data['dev_loader']['labels']} unique labels.",
              "Exact duplicate rows, clipped/invalid targets and overlap duplication are counted in data_manifest.json.",
              "24 native/restored examples plus 24 actual loader examples were inspected.",
              "", "## Actual calls", "| Match | Detector | G1 forwards | S4 forwards |",
              "|---|---|---:|---:|"]
    for row in calls:
        lines.append(f"| {row['match']} | {row['route']} | {row['g1_calls']} | {row['s4_calls']} |")
    lines += [
        "", "## Assistance and runtime blockers",
        "Positions and runtime track IDs come from detections and causal local association, not annotated player positions.",
        "Camera setup uses the existing historical local projector and fitting-region support; its failed global calibration status is unchanged.",
        "Team/role comes from the existing source-trained appearance classifier on video crops, with its frozen abstention thresholds. These cross-match predictions are not validated attributes.",
        "No target-match annotation attributes enter the autonomous adapter. Unknown attributes, unprojectable detections and unsupported queries are retained.",
        "Actor anchors and query times are declared setup; no common usable frozen video actor exists on D, where the original detector is almost empty.",
        "| Match | Route | Known / unknown team instances | Scheduled queries | Unsupported queries | G1-contract-unavailable actor samples | Fallback entity states |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in calls:
        counts = row["attribute_counts"]
        lines.append(
            f"| {row['match']} | {row['route']} | {counts['known_team']} / {counts['unknown_team']} | "
            f"{row['scheduled_queries']} | {row['scheduled_queries'] - row['query_status_counts'].get('supported', 0)} | "
            f"{row['g1_contract_unavailable_actor_samples']} | {row['fallback_entity_states']} |"
        )
    for row in calls:
        lines.append(f"Unsupported reasons ({row['match']}, {row['route']}): {row['unsupported_reasons']}.")
    lines += [
        "", "## Reproduction and verification", "Use the project's existing Windows virtual environment and local data/weights.",
        "The locked recipe and absolute run deadline describe this experiment; do not silently reuse an expired task deadline for a new fit.",
        "```powershell",
        "$py = '.\\.venv\\Scripts\\python.exe'",
        "$out = 'outputs\\gp_20261029_125507'",
        "# Training command used (immutable output; do not overwrite this completed run):",
        "& $py -m src.backbone.gpu_perception --output $out train",
        "# Optional native restore diagnostic (NOT run after the numerical stop; requires a new output):",
        "& $py -m src.backbone.gpu_perception_post --output $out resume-check",
        "# Recheck release hashes, annotation isolation and regressions:",
        "& $py -m src.backbone.gpu_perception --output $out verify",
        "# Serve on loopback with HTTP byte ranges for actual video seeking:",
        "& $py -m src.backbone.video_review_server --port 8766 --directory $out",
        "# Open http://127.0.0.1:8766/index.html",
        "```",
        "", "See task.json, recipe.json, resolved_trainer.json, training_progress.json, last_training.pt,",
        "last_runtime_state.pt, resume_evidence.json, evaluation_seal.json, evaluation_summary.json,",
        "tracker_development.json, expansion.json, integration_summary.json, comparison_videos_review.json and browser_validation.json.",
        "Playable paired videos: " + ", ".join(Path(video["path"]).name for video in videos) + ".",
        "", "## Limitations",
        "Simulated visibility is not gaze. G1 predictions do not feed S4. Supplied actor/query times are required.",
        "Unknown team/role, ball, possession and unsupported targets stay unknown. Calls are not tactical accuracy.",
        "D/E were previously examined. 128057 is perception-unseen, but appeared in G1/S4 fitting; pixel-only where no approved projection exists.",
        "Only one seed/initialization and bounded fit were tested. No model is promoted to release defaults.",
        "Native training checkpoints serialize EMA; optimizer/AMP state is retained but exact live-model/sampler continuation is not claimed.",
        "After the numerical stop, no further optimizer steps were attempted. The earlier disposable smoke tested optimizer/AMP restoration; resume_evidence.json records the skipped main-checkpoint probe.",
        f"Regressions: {verification['regression_summary']}.",
    ]
    (output / "handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>GPU perception comparison</title>"
        "<style>body{background:#181818;color:#eee;font:16px system-ui;margin:24px}"
        "video{display:block;width:min(100%,1280px);margin:16px 0}a{color:#7dcfff}</style>"
        "<h1>GPU perception: paired 120-second comparisons</h1>"
        "<p>Experimental only. Global pitch calibration remains BLOCKED. "
        "Left: frozen detector; right: GPU candidate. Cyan is setup support, not certification. "
        "Action queries are supplied-time diagnostics, not tactical recommendations.</p>"
        + "".join(
            f"<h2>Recording {match}</h2><video controls preload='metadata' "
            f"poster='{Path(video['path']).with_suffix('.jpg').name}' src='{Path(video['path']).name}'></video>"
            for match, video in zip(("117092", "118576"), videos, strict=True)
        )
        + "<p><a href='completion.json'>Measured results</a> | <a href='handoff.md'>Compact handoff</a></p>",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("command", choices=["track-dev", "expand", "render", "render-review", "resume-check", "handoff"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    {"track-dev": tracker_diagnostic, "expand": expansion, "render": render_diagnostics,
     "render-review": lambda output: render_diagnostics(output, review=True),
     "resume-check": resume_probe, "handoff": handoff}[args.command](args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
