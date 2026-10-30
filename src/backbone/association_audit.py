"""Frozen-input association audit; no fitting and no labels in assignment."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from src.backbone.actor_continuity import identity_audit
from src.backbone.gpu_perception import aggregate_scores
from src.backbone.identity_program import (
    LearnedAssociation, apply_association, eligible_identity, frames, json_hash, read,
)
from src.backbone.perception_continuation import same_detections
from src.backbone.video_player_cross_match import _maximum_matches, _read_annotations
from src.trajectory_dataset import file_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "outputs" / "ip_20261029_202245"


def valid_feature(value) -> bool:
    if value is None:
        return False
    array = np.asarray(value, float)
    return bool(array.ndim == 1 and array.size and np.isfinite(array).all()
                and np.linalg.norm(array) > 1e-8)


class SharedAssociation:
    """Same assignment/lifecycle for each representation, with finite feature age."""

    def __init__(self, width: int, representation: str, threshold: float = .35,
                 *, retain: bool = True, lifetime: float = .8, original_scale: bool = False):
        if representation not in ("motion", "hsv", "R1", "R2"):
            raise ValueError("Unknown frozen feature representation")
        if threshold <= 0 or not 0 < lifetime <= 1:
            raise ValueError("Positive threshold and finite feature lifecycle required")
        self.scale = width / 4096
        self.representation, self.threshold = representation, threshold
        self.retain, self.lifetime, self.original_scale = retain, lifetime, original_scale
        self.tracks, self.next_id, self.previous = {}, 1, None
        self.counts = Counter()
        self.feature_ages = []

    def distance(self, a, b) -> float:
        if self.representation == "hsv":
            return float(np.abs(np.asarray(a) - b).sum()) / 2
        return max(0., 1 - float(np.dot(a, b)))

    def step(self, frame: dict, features: list) -> dict:
        now = frame.get("timestamp_s", frame["frame_index"] / 25)
        if not math.isfinite(now) or self.previous is not None and now <= self.previous:
            raise ValueError("Strict chronological timestamps required")
        if len(features) != len(frame["detections"]):
            raise ValueError("Cached feature/detection population differs")
        self.previous = now
        if frame.get("camera_cut") or frame.get("identity_reset"):
            self.counts["cut_retirements"] += len(self.tracks)
            self.tracks.clear()
        expired = [key for key, value in self.tracks.items() if now - value["time"] > 1 + 1e-6]
        self.counts["retirements"] += len(expired)
        for key in expired:
            del self.tracks[key]
        result = copy.deepcopy(frame)
        keys = sorted(self.tracks)
        cost = np.full((len(keys), len(features)), np.inf)
        details = {}
        self.counts["detections"] += len(features)
        self.counts["missing_features"] += sum(not valid_feature(f) for f in features)
        for i, key in enumerate(keys):
            track = self.tracks[key]
            dt = now - track["time"]
            age = now - track["feature_time"] if track["feature_time"] is not None else None
            if age is not None:
                self.feature_ages.append(age)
            usable_old = valid_feature(track["feature"]) and (not self.retain or age <= self.lifetime + 1e-6)
            for j, detection in enumerate(result["detections"]):
                self.counts["candidate_pairs"] += 1
                distance = float(np.linalg.norm(np.asarray(detection["footpoint_xy_px"])
                                               - track["point"] - track["velocity"] * dt)) / self.scale
                if distance > 45 * min(dt / .2, 2):
                    self.counts["motion_distance_rejected"] += 1
                    continue
                appearance = None
                if self.representation != "motion" and valid_feature(features[j]) and usable_old:
                    appearance = self.distance(features[j], track["feature"])
                    if appearance > self.threshold:
                        self.counts["appearance_rejected"] += 1
                        continue
                gap = dt > .2001
                gap_limit = min(.15, self.threshold) if self.original_scale else self.threshold * 3 / 7
                if gap and (distance > 20 or track["observations"] < 5
                            or self.representation != "motion" and (appearance is None or appearance > gap_limit)):
                    self.counts["gap_rejected"] += 1
                    self.counts["gap_missing_current"] += int(not valid_feature(features[j]))
                    self.counts["gap_missing_or_stale_previous"] += int(not usable_old)
                    continue
                term = appearance if self.original_scale else (appearance / self.threshold if appearance is not None else 0)
                cost[i, j] = distance + 30 * (term or 0) + 8 * (dt / .2 - 1)
                details[i, j] = {"gap": gap, "elapsed_s": dt, "feature_distance": appearance}
        accepted, ambiguous = {}, set()
        if cost.size:
            augmented = np.concatenate([np.where(np.isfinite(cost), cost, 1e9),
                                        np.full((len(keys), len(keys)), 100.)], axis=1)
            rows, cols = linear_sum_assignment(augmented)
            for i, j in zip(rows, cols, strict=True):
                if j >= len(features) or not np.isfinite(cost[i, j]):
                    continue
                margin = 8 if details[i, j]["gap"] else 2
                row, col = np.delete(cost[i], j), np.delete(cost[:, j], i)
                if row.size and row.min() - cost[i, j] < margin or col.size and col.min() - cost[i, j] < margin:
                    ambiguous.add(int(j))
                    self.counts["ambiguity_rejected"] += 1
                else:
                    accepted[int(j)] = (keys[int(i)], details[i, j])
        for j, detection in enumerate(result["detections"]):
            point = np.asarray(detection["footpoint_xy_px"], float)
            feature = features[j] if valid_feature(features[j]) else None
            if j in accepted:
                key, detail = accepted[j]
                track = self.tracks[key]
                velocity = (.35 * (point - track["point"]) / detail["elapsed_s"]
                            + .65 * track["velocity"]).clip(-200 * self.scale, 200 * self.scale)
                observations = track["observations"] + 1
                self.counts["accepted"] += 1
                self.counts["gap_accepted"] += int(detail["gap"])
                if feature is None and valid_feature(track["feature"]):
                    self.counts["valid_feature_would_be_erased"] += 1
                feature_time = now if feature is not None else None
                if self.retain and feature is None and track["feature_time"] is not None:
                    if now - track["feature_time"] <= self.lifetime + 1e-6:
                        feature, feature_time = track["feature"], track["feature_time"]
                        self.counts["last_valid_retained"] += 1
                quality = "APPEARANCE_GAP_RELINK" if detail["gap"] else "APPEARANCE_ASSOCIATED"
            else:
                key = f"appearance-session-{self.next_id:06d}"
                self.next_id += 1
                velocity, observations = np.zeros(2), 1
                feature_time = now if feature is not None else None
                quality = "AMBIGUOUS_ABSTAIN" if j in ambiguous else "NEW_OBSERVED_SESSION"
                self.counts["births"] += 1
            detection.update(track=key, session_id=key, association_quality={"status": quality})
            self.tracks[key] = {"point": point, "velocity": velocity, "time": now,
                                "feature": feature, "feature_time": feature_time, "observations": observations}
        return result


def synthetic_reproduction() -> dict:
    results = {}
    for missing in (False, True):
        for method in ("original", "retained"):
            tracker = (LearnedAssociation(4096, .35) if method == "original" else
                       SharedAssociation(4096, "R1", .35, original_scale=True))
            ids = []
            for timestamp in (0., .2, .4, .6, .8, 1., 1.4):
                row = {"frame_index": round(timestamp * 25), "timestamp_s": timestamp,
                       "detections": [{"track": "input", "footpoint_xy_px": [100., 100.],
                                       "bbox_xyxy_px": [90., 50., 110., 100.], "detector_confidence": .9}]}
                feature = None if missing and timestamp == 1 else [1., 0.]
                ids.append(tracker.step(row, [feature])["detections"][0]["track"])
            results[f"{method}_missing_{missing}"] = {"ids": ids, "survives_gap": ids[0] == ids[-1]}
    if results["original_missing_True"]["survives_gap"] or not results["retained_missing_True"]["survives_gap"]:
        raise AssertionError("Specified feature lifecycle reproduction failed")
    return {"status": "REPRODUCED_AND_REPAIRED_SYNTHETIC", "results": results,
            "warning": "Synthetic correctness alone does not establish real-video improvement"}


def compact_score(stream, labels, width) -> dict:
    score = aggregate_scores(stream, labels, width)["thresholds"]["40"]
    score.pop("continuous_track_length_samples_histogram", None)
    audit = identity_audit(stream, labels, width)
    counts, previous = Counter(), {}
    for frame in stream:
        truth = labels.get(frame["frame_index"], [])
        pairs = _maximum_matches(frame["detections"], truth, 40 * width / 4096)
        current = {frame["detections"][i]["track"]: truth[j]["track_id"] for i, j, _ in pairs}
        for track, identity in current.items():
            if track in previous:
                counts["reference_scorable_adjacent_runtime_associations"] += 1
                counts["correct_adjacent_runtime_associations"] += int(previous[track] == identity)
                counts["incorrect_adjacent_runtime_associations"] += int(previous[track] != identity)
        previous = current
    return {"score": score, "relinks": audit["relinks"], "accepted_association_reference_audit": dict(counts)}


def distributions(saved, features, labels, width) -> dict:
    values = {name: defaultdict(list) for name in ("hsv", "R1", "R2")}
    previous = None
    for i, frame in enumerate(saved):
        truth = labels.get(frame["frame_index"], [])
        pairs = _maximum_matches(frame["detections"], truth, 40 * width / 4096)
        current = {di: truth[li] for di, li, _ in pairs}
        if previous is not None:
            old_frame, old = previous
            for di, a in old.items():
                for dj, b in current.items():
                    if math.dist(old_frame["detections"][di]["footpoint_xy_px"],
                                 frame["detections"][dj]["footpoint_xy_px"]) > 90 * width / 4096:
                        continue
                    kind = "positive" if a["track_id"] == b["track_id"] else (
                        "same_kit_negative" if a.get("team") == b.get("team") else "opponent_negative")
                    for name in values:
                        source = features["hsv"] if name == "hsv" else features["embeddings"][name]
                        x, y = source[i - 1][di], source[i][dj]
                        if valid_feature(x) and valid_feature(y):
                            distance = (float(np.abs(np.asarray(x) - y).sum()) / 2 if name == "hsv"
                                        else max(0., 1 - float(np.dot(x, y))))
                            values[name][kind].append(distance)
        previous = frame, current
    return values


def cache_inputs(source_key: str, development: bool = False) -> tuple:
    manifest = read(PRIOR / "source_manifest.json")
    source = next(s for s in manifest["sources"] if s["key"] == source_key)
    if development:
        match = source["id"]
        path = ROOT / "outputs" / "pc_20261029_144726" / "selected_final_development_predictions.json"
        saved = read(path)[match]
        feature_dir = PRIOR / "features" / f"development_{match}"
        contract = read(feature_dir / "contract.json")
        if contract["predictions"] != json_hash([[r["frame_index"], r["detections"]] for r in saved]):
            raise ValueError("Development cached input hash mismatch")
    else:
        seal = read(PRIOR / f"detections_{source_key}.json")
        chunk = next(c for c in seal["chunks"] if c["start_frame"] == 15000)
        path = ROOT / chunk["path"]
        if file_digest(path) != chunk["sha256"]:
            raise ValueError("Sealed retrospective detections changed")
        saved = frames(path)
        feature_dir = PRIOR / "features" / f"{source_key}_0015000"
        contract = read(feature_dir / "contract.json")
        if contract["predictions"] != json_hash([[r["frame_index"], r["detections"]] for r in saved]):
            raise ValueError("Retrospective cached input hash mismatch")
    feature_path = feature_dir / "embeddings.pt"
    features = torch.load(feature_path, map_location="cpu", weights_only=False)
    evidence = {"predictions": str(path.relative_to(ROOT)), "prediction_file_sha256": file_digest(path),
                "features": str(feature_path.relative_to(ROOT)), "feature_sha256": file_digest(feature_path),
                "contract": contract, "samples": len(saved)}
    return source, saved, features, evidence


def run_association(output: Path, check) -> dict:
    write_json(output / "missing_feature_reproduction.json", synthetic_reproduction())
    development, distances, inputs = {}, {n: defaultdict(list) for n in ("hsv", "R1", "R2")}, {}
    for key in ("117092_1st", "117093_1st"):
        check(association=True)
        source, saved, features, evidence = cache_inputs(key, True)
        labels = _read_annotations(source, {r["frame_index"] for r in saved})
        development[key] = source, saved, features, labels
        inputs[key] = evidence
        dist = distributions(saved, features, labels, source["dimensions_px"][0])
        for name, groups in dist.items():
            for group, values in groups.items():
                distances[name][group].extend(values)
    distance_report = {}
    grid = [{"name": "motion", "representation": "motion", "threshold": 1.}]
    for name, groups in distances.items():
        if not groups.get("positive"):
            raise ValueError(f"No development positive feature pairs for {name}")
        distance_report[name] = {group: {"n": len(values), "quantiles": np.quantile(values, [0, .1, .5, .9, .97, 1]).tolist()}
                                 for group, values in groups.items()}
        for quantile in (.90, .97):
            grid.append({"name": f"{name}_q{quantile}", "representation": name,
                         "threshold": float(np.clip(np.quantile(groups["positive"], quantile), .02, 1.8))})
    write_json(output / "association_lock.json", {"grid": grid, "distance_distributions": distance_report,
               "inputs": inputs, "locked_before_retrospective": True, "feature_lifetime_s": .8})
    metrics, counters, repair = {r["name"]: {} for r in grid}, {}, {}
    metrics["production_baseline"] = {}
    for key, (source, saved, features, labels) in development.items():
        check(association=True)
        width = source["dimensions_px"][0]
        clean = copy.deepcopy(saved)
        for row, cut in zip(clean, features["cuts"], strict=True):
            row["camera_cut"] = cut
        original = apply_association(saved, features, width, {"encoder": "baseline"})
        metrics["production_baseline"][key] = compact_score(original, labels, width)
        for name in ("R1", "R2"):
            vector = features["embeddings"][name]
            original_tracker = LearnedAssociation(width, .35)
            old = [original_tracker.step(f, v) for f, v in zip(clean, vector, strict=True)]
            instrumented = SharedAssociation(width, name, .35, original_scale=True, retain=False)
            unmodified = [instrumented.step(f, v) for f, v in zip(clean, vector, strict=True)]
            if [[d["track"] for d in f["detections"]] for f in old] != [
                    [d["track"] for d in f["detections"]] for f in unmodified]:
                raise AssertionError("Instrumentation changed original assignment")
            fixed_tracker = SharedAssociation(width, name, .35, original_scale=True)
            fixed = [fixed_tracker.step(f, v) for f, v in zip(clean, vector, strict=True)]
            same_detections(saved, fixed)
            repair[f"{key}_{name}"] = {"original": compact_score(old, labels, width),
                                      "retained": compact_score(fixed, labels, width),
                                      "original_counters": dict(instrumented.counts),
                                      "counters": dict(fixed_tracker.counts),
                                      "feature_age_quantiles_s": np.quantile(fixed_tracker.feature_ages, [0, .5, .9, 1]).tolist()}
        for route in grid:
            check(association=True)
            rep = route["representation"]
            vector = ([[] for _ in clean] if rep == "motion" else features["hsv"] if rep == "hsv"
                      else features["embeddings"][rep])
            if rep == "motion":
                vector = [[None] * len(f["detections"]) for f in clean]
            tracker = SharedAssociation(width, rep, route["threshold"])
            stream = [tracker.step(f, v) for f, v in zip(clean, vector, strict=True)]
            same_detections(saved, stream)
            metrics[route["name"]][key] = compact_score(stream, labels, width)
            counters[f"{key}_{route['name']}"] = dict(tracker.counts)
    baseline = metrics["production_baseline"]
    selected = "production_baseline"
    best = np.mean([v["score"]["switch_rate"] for v in baseline.values()])
    for route in grid:
        value = metrics[route["name"]]
        rate = np.mean([v["score"]["switch_rate"] for v in value.values()])
        if rate < best and eligible_identity(baseline, value):
            best, selected = rate, route["name"]
    locked = {"selected": selected, "development_metrics": metrics, "repair": repair, "counters": counters,
              "no_release_change": True, "negative_result": selected == "production_baseline"}
    write_json(output / "association_development.json", locked)
    retrospective = {}
    # One development-best setting per representation; no retrospective tuning.
    comparison = [grid[0]]
    for rep in ("hsv", "R1", "R2"):
        choices = [r for r in grid if r["representation"] == rep]
        comparison.append(min(choices, key=lambda r: np.mean([v["score"]["switch_rate"] for v in metrics[r["name"]].values()])))
    write_json(output / "association_retrospective_lock.json", {"routes": comparison,
               "intervals": ["118576_2nd:600-1200s", "128057_1st:600-1200s"], "selected": selected})
    for key in ("118576_2nd", "128057_1st"):
        check(association=True)
        source, saved, features, evidence = cache_inputs(key)
        width = source["dimensions_px"][0]
        frozen = {"production_baseline": apply_association(saved, features, width, {"encoder": "baseline"})}
        for route in comparison:
            rep = route["representation"]
            vector = ([[None] * len(f["detections"]) for f in saved] if rep == "motion"
                      else features["hsv"] if rep == "hsv" else features["embeddings"][rep])
            tracker = SharedAssociation(width, rep, route["threshold"])
            clean = copy.deepcopy(saved)
            for f, cut in zip(clean, features["cuts"], strict=True):
                f["camera_cut"] = cut
            frozen[route["name"]] = [tracker.step(f, v) for f, v in zip(clean, vector, strict=True)]
            same_detections(saved, frozen[route["name"]])
        labels = _read_annotations(source, {f["frame_index"] for f in saved})
        retrospective[key] = {"inputs": evidence,
                              "routes": {name: compact_score(stream, labels, width) for name, stream in frozen.items()}}
        write_json(output / f"association_{key}.json", retrospective[key])
    result = {"status": "COMPLETE", **locked, "retrospective": retrospective,
              "meaning": "Retrospective diagnostics; embedding effect isolated only within shared mechanics"}
    write_json(output / "association_summary.json", result)
    return {"status": "COMPLETE", "selected": selected, "summary": "association_summary.json"}
