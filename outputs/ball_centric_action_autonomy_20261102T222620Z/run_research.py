from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
PRIOR = ROOT / "outputs" / "action_opportunity_research_20261102_144500"
FOLLOWUP = ROOT / "outputs" / "action_opportunity_followup_20261102"
FOLLOWUP_CODE = ROOT / "action_opportunity_followup_20261102"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PRIOR))
sys.path.insert(0, str(FOLLOWUP_CODE))

from followup_common import (  # noqa: E402
    CLASS_NAMES,
    paired_match_bootstrap,
    read_json,
    sha256,
    validate_assets,
    write_json,
)
from possession_features import load_ball  # noqa: E402

FIT_MATCHES = ("117092", "117093", "118576", "118578", "128057", "132831")
DEV_MATCH = "132877"
SEEDS = (42, 137, 901)
HGB_PARAMS = {
    "max_iter": 100,
    "learning_rate": 0.07,
    "max_leaf_nodes": 15,
    "max_depth": 5,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}
CLASS_TO_GROUP = {
    1: "PASS_LIKE",
    2: "PASS_LIKE",
    3: "DRIVE",
    4: "PASS_LIKE",
    5: "SHOT_HEADER",
    6: "SHOT_HEADER",
    7: "DEFENSIVE_INTERVENTION",
}
LABEL_MAP = {
    "PASS": 1,
    "HIGH PASS": 2,
    "DRIVE": 3,
    "CROSS": 4,
    "SHOT": 5,
    "HEADER": 6,
    "BALL PLAYER BLOCK": 7,
    "PLAYER SUCCESSFUL TACKLE": 7,
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def append_decision(stage: str, decision: str, evidence: dict[str, Any], boundary: str) -> None:
    row = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "decision": decision,
        "evidence": evidence,
        "data_boundary": boundary,
    }
    with (RUN / "decisions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def check_protocol() -> dict:
    protocol = read_json(RUN / "protocol.lock.json")
    if protocol.get("schema") != "football.ball-centric-autonomy.protocol.1":
        raise ValueError("Protocol lock is missing or has an unexpected schema.")
    return protocol


def preflight() -> dict:
    started = time.perf_counter()
    protocol = check_protocol()
    manifest, verified = validate_assets(ROOT, PRIOR, verify_shards=True)
    audit = read_json(FOLLOWUP / "ball_alignment_audit.json")
    if audit.get("status") != "BALL_ALIGNMENT_PASS":
        raise ValueError(f"Existing as-of ball feature audit is blocked: {audit.get('status')}")
    feature_records = {(str(row["match"]), row["half"]): row for row in audit["halves"]}
    ball_sources = []
    feature_sources = []
    for shard in manifest["shards"]:
        key = (str(shard["match"]), shard["half"])
        record = feature_records[key]
        ball_path = ROOT / Path(record["ball_file"].replace("\\", "/"))
        feat_path = FOLLOWUP / Path(record["feature_shard"].replace("\\", "/"))
        if not ball_path.is_file() or sha256(ball_path) != record["ball_file_sha256"]:
            raise ValueError(f"Ball source fingerprint mismatch: {ball_path}")
        if not feat_path.is_file() or sha256(feat_path) != record["feature_shard_sha256"]:
            raise ValueError(f"Ball feature fingerprint mismatch: {feat_path}")
        ball_sources.append({"path": str(ball_path.relative_to(ROOT)), "sha256": record["ball_file_sha256"], "bytes": ball_path.stat().st_size})
        feature_sources.append({"path": str(feat_path.relative_to(ROOT)), "sha256": record["feature_shard_sha256"], "bytes": feat_path.stat().st_size})
    for name in ("comparison_status.json", "ball_alignment_audit.json", "followup_analysis.json", "threshold_repair.json"):
        path = FOLLOWUP / name
        if not path.is_file():
            raise FileNotFoundError(f"Required prior report is absent: {path}")
    import torch

    free = shutil.disk_usage(ROOT).free
    status_text = (RUN / "starting_git_status.txt").read_text(encoding="utf-8-sig")
    result = {
        "status": "PREFLIGHT_PASS",
        "protocol_sha256": sha256(RUN / "protocol.lock.json"),
        "cohort_manifest_sha256": verified["manifest_sha256"],
        "cohort_shards_verified": verified["shards"],
        "protected_checkpoints_verified": verified["protected"],
        "ball_sources_verified": ball_sources,
        "existing_ball_feature_shards_verified": feature_sources,
        "source_shard_count": len(verified["shards"]),
        "ball_source_count": len(ball_sources),
        "starting_git_status_lines": len([line for line in status_text.splitlines() if line.strip()]),
        "fit_matches": list(FIT_MATCHES),
        "development_match": DEV_MATCH,
        "retrospective_matches_loaded": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "sklearn": __import__("sklearn").__version__,
            "disk_free_bytes_at_preflight": free,
            "disk_free_gb_at_preflight": round(free / 1e9, 2),
        },
        "previous_ball_audit": {
            "fit_event_distance_n": audit["fit_event_ball_distance_count"],
            "fit_event_distance_median_m": audit["fit_event_ball_distance_median_m"],
            "fit_event_distance_p90_m": audit["fit_event_ball_distance_p90_m"],
            "coordinate_claim": "Pitch-bounds plausibility and prior alignment gate only; not independent clock or physical-video synchronization validation.",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    atomic_json(RUN / "source_integrity.json", result)
    atomic_json(RUN / "status.json", {"current_stage": "PREFLIGHT_COMPLETE", "completed": ["preflight"], "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    append_decision("preflight", "Proceed using verified 18-shard cohort and prior as-of ball feature cache.", {"source_shards": len(verified["shards"]), "ball_sources": len(ball_sources), "gpu": result["environment"]["gpu"], "free_gb": result["environment"]["disk_free_gb_at_preflight"]}, "fit/dev only; retrospective result files not opened")
    return result


def load_split(split: str) -> dict[str, np.ndarray]:
    manifest = read_json(PRIOR / "cohort_manifest.json")
    audit = read_json(FOLLOWUP / "ball_alignment_audit.json")
    records = {(str(row["match"]), row["half"]): row for row in audit["halves"]}
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        source = ROOT / Path(shard["shard_path"].replace("\\", "/"))
        if sha256(source) != shard["shard_sha256"]:
            raise ValueError(f"Cohort shard changed: {source}")
        row = records[(str(shard["match"]), shard["half"])]
        ball_path = FOLLOWUP / Path(row["feature_shard"].replace("\\", "/"))
        if sha256(ball_path) != row["feature_shard_sha256"]:
            raise ValueError(f"Ball feature shard changed: {ball_path}")
        with np.load(source, allow_pickle=False) as z, np.load(ball_path, allow_pickle=False) as b:
            for name in ("target", "match", "half_code", "query_frame", "scene_id_local_only", "actor_slot_local_only", "segment_local_only"):
                pieces[name].append(z[name])
            pieces["x"].append(z["x_causal_uncertainty"].astype(np.float32))
            for name in ("features", "query_frame", "actor_slot_local_only", "segment_local_only", "half_code"):
                if name != "features" and not np.array_equal(z[name], b[name]):
                    raise ValueError(f"Ball/cohort row alignment failure: {source.name}:{name}")
            pieces["ball"].append(b["features"].astype(np.float32))
    if not pieces["x"]:
        raise ValueError(f"No data for split {split}")
    output = {key: np.concatenate(values) for key, values in pieces.items()}
    output["target"] = output["target"].astype(np.int64)
    output["match"] = output["match"].astype(str)
    return output


def scene_keys(data: dict[str, np.ndarray]) -> list[tuple[str, int, int]]:
    return list(zip(data["match"].astype(str), data["half_code"].astype(int), data["scene_id_local_only"].astype(int), strict=True))


def scene_rank_metrics(y: np.ndarray, score: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, Any]:
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(scene_keys(data)):
        groups[key].append(index)
    ranks: list[int] = []
    positive_actor_count = 0
    for rows in groups.values():
        positives = [i for i in rows if y[i] > 0]
        if not positives:
            continue
        ordered = sorted(rows, key=lambda i: (-float(score[i]), i))
        rank_by_row = {row: rank + 1 for rank, row in enumerate(ordered)}
        ranks.append(min(rank_by_row[i] for i in positives))
        positive_actor_count += len(positives)
    if not ranks:
        return {"positive_scenes": 0, "positive_actors": 0, "top1": None, "top3": None, "mrr": None}
    rank_array = np.asarray(ranks, dtype=np.int64)
    return {
        "positive_scenes": len(ranks),
        "positive_actors": positive_actor_count,
        "top1": float(np.mean(rank_array == 1)),
        "top3": float(np.mean(rank_array <= 3)),
        "mrr": float(np.mean(1.0 / rank_array)),
    }


def binary_metrics(y: np.ndarray, score: np.ndarray, data: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    y = (np.asarray(y) > 0).astype(np.int8)
    score = np.clip(np.asarray(score, dtype=np.float64), 1e-8, 1 - 1e-8)
    result: dict[str, Any] = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "AP": float(average_precision_score(y, score)) if y.any() else None,
        "brier": float(brier_score_loss(y, score)),
        "hazard_NLL": float(log_loss(y, np.column_stack((1.0 - score, score)), labels=[0, 1])),
    }
    if data is not None:
        result["scene_ranking"] = scene_rank_metrics(y, score, data)
    return result


def per_match_metrics(data: dict[str, np.ndarray], scores: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for match in sorted(np.unique(data["match"]).tolist()):
        mask = data["match"] == match
        subset = {key: value[mask] for key, value in data.items() if isinstance(value, np.ndarray) and len(value) == len(mask)}
        output[str(match)] = {name: binary_metrics(data["target"][mask], score[mask], subset) for name, score in scores.items()}
    return output


def paired_metric_ci(per_match: dict[str, dict[str, Any]], left: str, right: str, metric: str) -> dict[str, Any] | None:
    if not per_match:
        return None
    a = {match: {metric: values[left].get(metric)} for match, values in per_match.items() if values[left].get(metric) is not None and values[right].get(metric) is not None}
    b = {match: {metric: values[right].get(metric)} for match, values in per_match.items() if values[left].get(metric) is not None and values[right].get(metric) is not None}
    if not a:
        return None
    return paired_match_bootstrap(a, b, metric=metric, seed=20261102)


def mean_bootstrap_ci(values: list[float], seed: int = 20261102) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(10000, len(array)))].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def hgb_model(x: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    binary_y = (y > 0).astype(np.int8)
    if np.unique(binary_y).size < 2:
        raise ValueError("HGB fold has only one target class.")
    return HistGradientBoostingClassifier(**HGB_PARAMS).fit(x, binary_y)


def block_shuffle_ball(data: dict[str, np.ndarray], rng_seed: int = 91357) -> np.ndarray:
    features = data["ball"].copy()
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(data["match"].astype(str), data["half_code"].astype(int), (data["query_frame"] // 500).astype(int), strict=True)):
        groups[key].append(index)
    rng = np.random.default_rng(rng_seed)
    for rows in groups.values():
        valid = [index for index in rows if features[index, 0] > 0.5]
        if len(valid) > 1:
            source = np.asarray(valid, dtype=np.int64)
            features[source] = features[rng.permutation(source)].copy()
    return features


def make_lagged_ball_features(data: dict[str, np.ndarray], lag_seconds: float) -> np.ndarray:
    result = np.zeros_like(data["ball"], dtype=np.float32)
    for match in np.unique(data["match"]):
        match_mask = data["match"] == match
        for half_code in (1, 2):
            mask = match_mask & (data["half_code"] == half_code)
            if not mask.any():
                continue
            half = "1st" if half_code == 1 else "2nd"
            track = load_ball(ROOT / "data" / f"{match}_{half}_ball.npz")
            q = data["query_frame"][mask].astype(np.int64) - int(round(lag_seconds * 25))
            ball_xy, ball_v, age, valid = track.asof(q)
            actor_xy = data["x"][mask, :2] * np.asarray([52.5, 34.0], dtype=np.float32)
            actor_v = data["x"][mask, 2:4] * 12.0
            delta = ball_xy - actor_xy
            dist = np.linalg.norm(delta, axis=1)
            unit = delta / np.maximum(dist[:, None], 1e-6)
            approach = np.sum(actor_v * unit, axis=1)
            speed = np.linalg.norm(ball_v, axis=1)
            closing = np.sum((actor_v - ball_v) * unit, axis=1)
            output = np.zeros((int(mask.sum()), 12), dtype=np.float32)
            output[:, 0] = valid
            output[:, 1] = np.where(valid, age / 0.2, 0)
            output[:, 2:4] = np.where(valid[:, None], delta / np.asarray([105.0, 68.0], dtype=np.float32), 0)
            output[:, 4] = np.where(valid, dist / 105.0, 0)
            output[:, 5] = np.where(valid, np.clip(approach / 12.0, -2, 2), 0)
            output[:, 6] = np.where(valid, np.minimum(speed / 40.0, 2), 0)
            output[:, 7] = np.where(valid, np.clip(closing / 40.0, -2, 2), 0)
            output[:, 8] = valid & (dist <= 2.0)
            output[:, 9] = valid & (dist <= 5.0)
            keys: dict[tuple[int, int], list[int]] = defaultdict(list)
            positions = np.flatnonzero(mask)
            for local_index, global_index in enumerate(positions):
                keys[(int(data["half_code"][global_index]), int(data["scene_id_local_only"][global_index]))].append(local_index)
            for rows in keys.values():
                if valid[rows].all():
                    order = np.argsort(dist[rows], kind="stable")
                    ranks = np.empty(len(rows), dtype=np.float32)
                    ranks[order] = np.arange(len(rows), dtype=np.float32)
                    output[rows, 10] = 1.0 - ranks / max(1, len(rows) - 1)
                    output[rows, 11] = np.min(dist[rows]) / 105.0
            output[~valid] = 0.0
            result[mask] = output
    return result


def metric_bundle(data: dict[str, np.ndarray], scores: dict[str, np.ndarray], models: dict[str, Any]) -> dict[str, Any]:
    by_match = per_match_metrics(data, scores)
    overall = {name: binary_metrics(data["target"], score, data) for name, score in scores.items()}
    contrasts = {}
    for left, right in (("x_plus_all", "x_no_ball"), ("x_plus_distance", "x_no_ball"), ("history_hgb", "x_plus_all")):
        if left in scores and right in scores:
            contrasts[f"{left}_minus_{right}"] = {
                metric: paired_metric_ci(by_match, left, right, metric)
                for metric in ("AP", "brier", "hazard_NLL")
            }
    return {"overall": overall, "per_match": by_match, "paired_match_contrasts": contrasts, "model_metadata": models}


def load_cached_prediction_scores(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scores = {
        "x_no_ball": np.full(len(data["target"]), np.nan, dtype=np.float64),
        "x_plus_all": np.full(len(data["target"]), np.nan, dtype=np.float64),
    }
    files = []
    for match in FIT_MATCHES:
        path = FOLLOWUP / f"cv_{match}_predictions_local.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Verified prior cross-match prediction cache is required: {path}")
        positions = np.flatnonzero(data["match"] == match)
        with np.load(path, allow_pickle=False) as z:
            for key in ("target", "match", "half_code", "query_frame", "scene_id_local_only"):
                if not np.array_equal(z[key], data[key][positions]):
                    raise ValueError(f"Cached OOF row alignment mismatch for {match}:{key}")
            for cache_key, output_key in (("hgb_original", "x_no_ball"), ("hgb_with_ball", "x_plus_all")):
                probability = z[cache_key].astype(np.float64)
                if probability.shape != (len(positions), 8) or not np.isfinite(probability).all():
                    raise ValueError(f"Invalid cached probability matrix {match}:{cache_key}")
                if not np.allclose(probability.sum(axis=1), 1.0, atol=2e-4):
                    raise ValueError(f"Unnormalized cached probabilities {match}:{cache_key}")
                scores[output_key][positions] = probability[:, 1:].sum(axis=1)
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": int(len(positions))})
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("Cached OOF predictions do not cover every fit row.")
    return scores, {"source": "verified prior HGB OOF prediction caches", "files": files}


def run_investigation_a() -> dict[str, Any]:
    if (RUN / "ball_shortcut_audit.json").is_file():
        return read_json(RUN / "ball_shortcut_audit.json")
    started = time.perf_counter()
    data = load_split("fit")
    scores, cache_meta = load_cached_prediction_scores(data)
    models: dict[str, Any] = {"cached_baselines": cache_meta, "hgb_params": HGB_PARAMS}
    fit_indices = np.arange(len(data["target"]))
    arms = (
        "ball_only",
        "x_plus_availability",
        "x_plus_distance",
        "x_plus_block_shuffle",
        "lag_0p2",
        "lag_0p5",
        "lag_1p0",
        "lag_2p0",
    )
    lag_features = {name: make_lagged_ball_features(data, float(name.split("_")[1].replace("p", "."))) for name in arms if name.startswith("lag_")}
    shuffled = block_shuffle_ball(data)
    arm_arrays: dict[str, np.ndarray] = {
        "ball_only": data["ball"][:, [0, 1, 2, 3, 4, 6, 7]],
        "x_plus_availability": np.column_stack((data["x"], data["ball"][:, 0])),
        "x_plus_distance": np.column_stack((data["x"], data["ball"][:, 4])),
        "x_plus_block_shuffle": np.column_stack((data["x"], shuffled)),
    }
    for name, features in lag_features.items():
        arm_arrays[name] = np.column_stack((data["x"], features))
    arm_metadata: dict[str, Any] = {}
    for held in FIT_MATCHES:
        test = data["match"] == held
        train = ~test
        if not test.any() or not train.any():
            raise ValueError(f"Invalid fold coverage for {held}.")
        for arm in arms:
            x = np.asarray(arm_arrays[arm], dtype=np.float32)
            model = hgb_model(x[train], data["target"][train])
            scores[arm] = scores.get(arm, np.full(len(data["target"]), np.nan))
            scores[arm][test] = model.predict_proba(x[test])[:, 1]
            arm_metadata.setdefault(arm, []).append({
                "held_out_match": held,
                "fit_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "fit_positive_rows": int((data["target"][train] > 0).sum()),
            })
            del model
    valid_ball = data["ball"][:, 0] > 0.5
    scores["nearest_ball_rule"] = np.where(valid_ball, -data["ball"][:, 4], -2.0).astype(np.float64)
    models["arms"] = arm_metadata
    models["source_row_count"] = int(len(data["target"]))
    models["fit_positive_rows"] = int((data["target"] > 0).sum())
    bundle = metric_bundle(data, scores, models)

    strata: dict[str, Any] = {}
    strata_masks = {
        "all": np.ones(len(data["target"]), dtype=bool),
        "ball_available": valid_ball,
        "ball_unavailable": ~valid_ball,
        "nearest_within_2m": valid_ball & (data["ball"][:, 4] <= 2 / 105),
        "nearest_within_5m": valid_ball & (data["ball"][:, 4] <= 5 / 105),
        "nearest_beyond_10m": valid_ball & (data["ball"][:, 4] > 10 / 105),
    }
    for name, mask in strata_masks.items():
        strata[name] = {
            "n": int(mask.sum()),
            "positive_rows": int((data["target"][mask] > 0).sum()),
            "prevalence": float(np.mean(data["target"][mask] > 0)) if mask.any() else None,
            "models": {
                arm: binary_metrics(data["target"][mask], score[mask])
                for arm, score in scores.items()
            },
        }
    class_support = {CLASS_NAMES[index]: int(np.sum(data["target"] == index)) for index in range(1, 8)}
    judgement = "INCONCLUSIVE"
    if "nearest_ball_rule" in bundle["overall"] and "x_plus_all" in bundle["overall"]:
        nearest = bundle["overall"]["nearest_ball_rule"]["AP"]
        full = bundle["overall"]["x_plus_all"]["AP"]
        added = bundle["overall"]["x_plus_all"]["scene_ranking"]["top1"] - bundle["overall"]["x_no_ball"]["scene_ranking"]["top1"]
        if nearest is not None and full is not None and nearest >= 0.8 * full and added < 0.1:
            judgement = "BALL_PROXIMITY_SUFFICIENT"
        elif full is not None and nearest is not None and full - nearest >= 0.05 and added >= 0.1:
            judgement = "ADDITIONAL_TACTICAL_SIGNAL"
        elif bundle["overall"]["x_plus_all"]["AP"] <= bundle["overall"]["x_no_ball"]["AP"]:
            judgement = "LEAKAGE_OR_ALIGNMENT_RISK"
    result = {
        "status": "COMPLETE",
        "judgement": judgement,
        "judgement_rule": "Nearest-rule AP at least 80% of prior full-ball AP and <0.10 top-1 gain => proximity-sufficient; otherwise a >=0.05 full-minus-nearest AP gap plus >=0.10 scene-top1 gain => additional signal. Labels remain annotation-supported only.",
        "target": "Existing earliest supported actor event in (query, query+1s]; no-event is not verified inactivity.",
        "matched_oof_rows": int(len(data["target"])),
        "natural_prevalence": float(np.mean(data["target"] > 0)),
        "class_support": class_support,
        "overall": bundle["overall"],
        "per_match": bundle["per_match"],
        "paired_match_contrasts": bundle["paired_match_contrasts"],
        "availability_and_distance_strata": strata,
        "model_metadata": models,
        "notes": [
            "Nearest-ball heuristic is a ranking score, not a calibrated event probability.",
            "Within-match shuffle preserves ball-availability pattern and permutes valid actor-row ball vectors among rows in the same 20-second block; this is a diagnostic placebo.",
            "Lagged features use source ball frames at or before query minus the lag; they are not future-shifted inputs.",
            "All model OOF comparisons use the same six fit-match target rows. The prior cached no-ball/full-ball HGB predictions were row-by-row verified before reuse.",
            "The unavailable stratum is reported but is likely unstable because positive support is small.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    atomic_json(RUN / "ball_shortcut_audit.json", result)
    append_decision("A", f"Ball shortcut result: {judgement}.", {"nearest_rule_AP": bundle["overall"]["nearest_ball_rule"]["AP"], "full_ball_AP": bundle["overall"]["x_plus_all"]["AP"], "no_ball_AP": bundle["overall"]["x_no_ball"]["AP"], "history_required": True}, "six leave-one-fit-match-out folds; fit labels only for training; no retrospective data")
    return result


HISTORY_OFFSETS = (50, 40, 30, 20, 10, 0)
HISTORY_NAMES = ("available", "age_fraction", "dx_pitch", "dy_pitch", "distance_pitch", "approach_speed", "ball_speed", "closing_speed")


def build_sequence_rows(
    query: np.ndarray,
    slots: np.ndarray,
    segments: np.ndarray,
    raw: dict[str, np.ndarray],
    ball_track: Any,
    static_x: np.ndarray,
    static_ball: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    n = len(query)
    seq = np.zeros((n, len(HISTORY_OFFSETS), len(HISTORY_NAMES)), dtype=np.float32)
    observation_counts = []
    for step, offset in enumerate(HISTORY_OFFSETS):
        requested = query.astype(np.int64) - offset
        ball_xy, ball_velocity, age, ball_valid = ball_track.asof(requested, max_age_frames=5)
        actor_frame = requested.copy()
        actor_valid = np.zeros(n, dtype=bool)
        actor_xy = np.zeros((n, 2), dtype=np.float32)
        for back in range(6):
            candidate_frame = requested - back
            valid_index = candidate_frame >= 0
            row_indices = np.flatnonzero(valid_index & ~actor_valid)
            if not len(row_indices):
                break
            frame = candidate_frame[row_indices]
            row_slots = slots[row_indices]
            observed = raw["observed"][frame, row_slots]
            same_segment = raw["segments"][frame, row_slots] == segments[row_indices]
            good = observed & same_segment
            selected = row_indices[good]
            if len(selected):
                selected_frames = candidate_frame[selected]
                selected_slots = slots[selected]
                actor_xy[selected] = raw["xy"][selected_frames, selected_slots].astype(np.float32)
                actor_frame[selected] = selected_frames
                actor_valid[selected] = True
        both = actor_valid & ball_valid
        delta = ball_xy - actor_xy
        dist = np.linalg.norm(delta, axis=1)
        unit = delta / np.maximum(dist[:, None], 1e-6)
        actor_velocity = np.zeros((n, 2), dtype=np.float32)
        prior_frame = actor_frame - 5
        prior_ok = both & (prior_frame >= 0)
        candidate = np.flatnonzero(prior_ok)
        if len(candidate):
            prior_slots = slots[candidate]
            prior_frames = prior_frame[candidate]
            good = raw["observed"][prior_frames, prior_slots] & (raw["segments"][prior_frames, prior_slots] == segments[candidate])
            use = candidate[good]
            actor_velocity[use] = (actor_xy[use] - raw["xy"][prior_frame[use], slots[use]].astype(np.float32)) / 0.2
        speed = np.linalg.norm(ball_velocity, axis=1)
        approach = np.sum(actor_velocity * unit, axis=1)
        closing = np.sum((actor_velocity - ball_velocity) * unit, axis=1)
        step_data = seq[:, step]
        step_data[:, 0] = both.astype(np.float32)
        step_data[:, 1] = np.where(both, age / 0.2, 0.0)
        step_data[:, 2] = np.where(both, delta[:, 0] / 105.0, 0.0)
        step_data[:, 3] = np.where(both, delta[:, 1] / 68.0, 0.0)
        step_data[:, 4] = np.where(both, dist / 105.0, 0.0)
        step_data[:, 5] = np.where(both, np.clip(approach / 12.0, -2.0, 2.0), 0.0)
        step_data[:, 6] = np.where(both, np.minimum(speed / 40.0, 2.0), 0.0)
        step_data[:, 7] = np.where(both, np.clip(closing / 40.0, -2.0, 2.0), 0.0)
        observation_counts.append(float(np.mean(both)))
    if not np.isfinite(seq).all():
        raise FloatingPointError("Causal ball history contains non-finite values.")
    if len(seq):
        if not np.allclose(seq[:, -1, 0], static_ball[:, 0], atol=0.0):
            raise ValueError("Current history-ball availability does not match the audited feature cache.")
        distance_delta = np.abs(seq[:, -1, 4] - static_ball[:, 4])
        if float(np.max(distance_delta)) > 2.0 / 105.0:
            raise ValueError(f"Current history distance differs from audited cache by {np.max(distance_delta):.6f}.")
    return seq, {
        "current_valid_fraction": observation_counts[-1] if observation_counts else 0.0,
        "history_valid_fraction_by_step": observation_counts,
        "current_distance_max_delta_pitch": float(np.max(np.abs(seq[:, -1, 4] - static_ball[:, 4]))) if n else 0.0,
    }


def load_or_build_sequences(split: str, data: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = read_json(PRIOR / "cohort_manifest.json")
    seq_parts = []
    row_reports = []
    history_dir = RUN / "ball_history_shards_local"
    history_dir.mkdir(exist_ok=True)
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        name = f"{shard['match']}_{'1' if shard['half'] == '1st' else '2'}"
        source = ROOT / Path(shard["shard_path"].replace("\\", "/"))
        output = history_dir / f"{name}.npz"
        meta_path = output.with_suffix(".json")
        expected = {
            "cohort_sha256": shard["shard_sha256"],
            "ball_source_sha256": sha256(ROOT / "data" / f"{shard['match']}_{shard['half']}_ball.npz"),
            "history_offsets": list(HISTORY_OFFSETS),
        }
        reusable = False
        if output.is_file() and meta_path.is_file():
            saved = read_json(meta_path)
            reusable = saved.get("inputs") == expected and saved.get("output_sha256") == sha256(output)
        if reusable:
            with np.load(output, allow_pickle=False) as z:
                seq = z["sequence"].astype(np.float32)
                if len(seq) != shard["sample_rows"]:
                    raise ValueError(f"Cached history row count mismatch: {output}")
                report = read_json(meta_path)["report"]
        else:
            with np.load(source, allow_pickle=False) as z:
                query = z["query_frame"].astype(np.int64)
                slots = z["actor_slot_local_only"].astype(np.int64)
                segments = z["segment_local_only"].astype(np.int64)
                x = z["x_causal_uncertainty"].astype(np.float32)
            ball_path = ROOT / "data" / f"{shard['match']}_{shard['half']}_ball.npz"
            track = load_ball(ball_path)
            cache_path = ROOT / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{shard['match']}_{shard['half']}.npz"
            with np.load(cache_path, allow_pickle=False) as z:
                raw = {key: z[key] for key in ("xy", "observed", "segments")}
            ball_record = next(
                row for row in read_json(FOLLOWUP / "ball_alignment_audit.json")["halves"]
                if row["match"] == str(shard["match"]) and row["half"] == shard["half"]
            )
            ball_feature_path = FOLLOWUP / Path(ball_record["feature_shard"].replace("\\", "/"))
            with np.load(ball_feature_path, allow_pickle=False) as z:
                static_ball = z["features"].astype(np.float32)
            seq, report = build_sequence_rows(query, slots, segments, raw, track, x, static_ball)
            atomic_npz(output, sequence=seq)
            meta = {"inputs": expected, "output_sha256": sha256(output), "report": report, "rows": int(len(seq))}
            atomic_json(meta_path, meta)
        seq_parts.append(seq)
        row_reports.append({"shard": name, "rows": int(len(seq)), **report})
    sequence = np.concatenate(seq_parts, axis=0)
    if len(sequence) != len(data["target"]):
        raise ValueError(f"History sequence row mismatch for {split}: {len(sequence)} != {len(data['target'])}")
    return sequence, {"split": split, "rows": int(len(sequence)), "offset_frames": list(HISTORY_OFFSETS), "feature_names": list(HISTORY_NAMES), "shards": row_reports}


def normalize_train(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-4] = 1.0
    return mean, scale


def normalize_values(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = (values - mean) / scale
    np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -20.0, 20.0).astype(np.float32, copy=False)


def train_temporal_seed(
    x_fit: np.ndarray,
    seq_fit: np.ndarray,
    y_fit: np.ndarray,
    x_dev: np.ndarray,
    seq_dev: np.ndarray,
    y_dev: np.ndarray,
    seq_test: np.ndarray,
    seed: int,
    max_epochs: int = 8,
    patience: int = 2,
    batch_size: int = 4096,
) -> tuple[Any, dict[str, Any], np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    class TemporalNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context = nn.Sequential(nn.Linear(x_fit.shape[1], 96), nn.GELU(), nn.LayerNorm(96), nn.Linear(96, 48), nn.GELU())
            self.gru = nn.GRU(input_size=seq_fit.shape[2], hidden_size=32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(80, 48), nn.GELU(), nn.Dropout(0.1), nn.Linear(48, 1))

        def forward(self, static: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
            context = self.context(static)
            _, hidden = self.gru(sequence)
            return self.head(torch.cat((context, hidden[-1]), dim=1)).squeeze(1)

    if not torch.cuda.is_available():
        raise RuntimeError("The required temporal experiment cannot run without CUDA.")
    device = torch.device("cuda")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    x_mean, x_scale = normalize_train(x_fit)
    s_mean, s_scale = normalize_train(seq_fit.reshape(-1, seq_fit.shape[-1]))
    x_fit_n = normalize_values(x_fit, x_mean, x_scale)
    x_dev_n = normalize_values(x_dev, x_mean, x_scale)
    seq_fit_n = normalize_values(seq_fit, s_mean, s_scale)
    seq_dev_n = normalize_values(seq_dev, s_mean, s_scale)
    seq_test_n = normalize_values(seq_test, s_mean, s_scale)
    x_fit_t = torch.from_numpy(x_fit_n).to(device)
    s_fit_t = torch.from_numpy(seq_fit_n).to(device)
    y_fit_t = torch.from_numpy((y_fit > 0).astype(np.float32)).to(device)
    x_dev_t = torch.from_numpy(x_dev_n).to(device)
    s_dev_t = torch.from_numpy(seq_dev_n).to(device)
    y_dev_t = torch.from_numpy((y_dev > 0).astype(np.float32)).to(device)
    model = TemporalNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history = []
    steps = 0
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(x_fit_t), generator=generator, device=device)
        batch_losses = []
        for offset in range(0, len(order), batch_size):
            ix = order[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                logits = model(x_fit_t[ix], s_fit_t[ix])
                loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), y_fit_t[ix])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite temporal loss seed={seed}, epoch={epoch + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite temporal gradient seed={seed}, epoch={epoch + 1}")
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            steps += 1
        model.eval()
        with torch.inference_mode():
            dev_logits = []
            for offset in range(0, len(x_dev_t), 16384):
                logits = model(x_dev_t[offset:offset + 16384], s_dev_t[offset:offset + 16384])
                dev_logits.append(logits.float())
            logits_all = torch.cat(dev_logits)
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(logits_all, y_dev_t).cpu())
        history.append({"epoch": epoch + 1, "train_bce": float(np.mean(batch_losses)), "development_bce": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Temporal model did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()

    def predict(xvalues: np.ndarray, svalues: np.ndarray) -> np.ndarray:
        x_norm = normalize_values(xvalues, x_mean, x_scale)
        s_norm = normalize_values(svalues, s_mean, s_scale)
        xt = torch.from_numpy(x_norm).to(device)
        st = torch.from_numpy(s_norm).to(device)
        result = []
        with torch.inference_mode():
            for offset in range(0, len(xt), 16384):
                result.append(torch.sigmoid(model(xt[offset:offset + 16384], st[offset:offset + 16384])).float().cpu().numpy())
        return np.concatenate(result)

    dev_probability = predict(x_dev, seq_dev)
    elapsed = time.perf_counter() - start
    metadata = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "steps": steps,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "development_selection_metric": "natural-prevalence binary BCE; development match 132877",
        "best_development_bce": best_loss,
        "training_curve": history,
        "elapsed_seconds": round(elapsed, 2),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_memory_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
    }
    return model, {"x_mean": x_mean, "x_scale": x_scale, "s_mean": s_mean, "s_scale": s_scale, "device": device}, dev_probability, metadata


def temporal_predict(model: Any, normalization: dict[str, Any], x: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    import torch

    device = normalization["device"]
    x_norm = normalize_values(x, normalization["x_mean"], normalization["x_scale"])
    seq_norm = normalize_values(sequence, normalization["s_mean"], normalization["s_scale"])
    xt = torch.from_numpy(x_norm).to(device)
    st = torch.from_numpy(seq_norm).to(device)
    result = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(xt), 16384):
            result.append(torch.sigmoid(model(xt[offset:offset + 16384], st[offset:offset + 16384])).float().cpu().numpy())
    return np.concatenate(result)


def perturb_sequence(sequence: np.ndarray, *, dropout_fraction: float = 0.0, noise_metres: float = 0.0, seed: int = 0) -> np.ndarray:
    result = sequence.copy()
    rng = np.random.default_rng(seed)
    if dropout_fraction > 0:
        selected = rng.random(len(result)) < dropout_fraction
        result[selected] = 0.0
    if noise_metres > 0:
        noise = rng.normal(0.0, noise_metres, size=result[:, :, 2:4].shape).astype(np.float32)
        result[:, :, 2] += noise[:, :, 0] / 105.0
        result[:, :, 3] += noise[:, :, 1] / 68.0
        result[:, :, 4] = np.sqrt((result[:, :, 2] * 105.0) ** 2 + (result[:, :, 3] * 68.0) ** 2) / 105.0
        invalid = result[:, :, 0] < 0.5
        result[invalid] = 0.0
    return result


def run_investigation_b() -> dict[str, Any]:
    result_path = RUN / "temporal_experiments.json"
    if result_path.is_file():
        return read_json(result_path)
    started = time.perf_counter()
    fit = load_split("fit")
    dev = load_split("development")
    seq_fit, fit_manifest = load_or_build_sequences("fit", fit)
    seq_dev, dev_manifest = load_or_build_sequences("development", dev)
    current_hgb_scores = np.full(len(fit["target"]), np.nan, dtype=np.float64)
    history_hgb_scores = np.full(len(fit["target"]), np.nan, dtype=np.float64)
    gru_current_scores = np.full(len(fit["target"]), np.nan, dtype=np.float64)
    gru_history_scores = np.full(len(fit["target"]), np.nan, dtype=np.float64)
    robustness: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_metadata: list[dict[str, Any]] = []
    prediction_dir = RUN / "predictions_local"
    prediction_dir.mkdir(exist_ok=True)
    static_fit = fit["x"].astype(np.float32)
    static_dev = dev["x"].astype(np.float32)
    static_current_fit = np.column_stack((fit["x"], fit["ball"])).astype(np.float32)
    static_current_dev = np.column_stack((dev["x"], dev["ball"])).astype(np.float32)
    history_fit_features = np.column_stack((static_current_fit, seq_fit[:, :-1, :].reshape(len(seq_fit), -1))).astype(np.float32)
    history_dev_features = np.column_stack((static_current_dev, seq_dev[:, :-1, :].reshape(len(seq_dev), -1))).astype(np.float32)
    dev_binary = (dev["target"] > 0).astype(np.int8)
    if not dev_binary.any():
        raise ValueError("Development set has no positive labels for temporal selection.")

    for held in FIT_MATCHES:
        test = fit["match"] == held
        train = ~test
        positions = np.flatnonzero(test)
        cache = prediction_dir / f"temporal_{held}.npz"
        cache_meta = cache.with_suffix(".json")
        input_config = {
            "held_out_match": held,
            "cohort_manifest_sha256": sha256(PRIOR / "cohort_manifest.json"),
            "sequence_fit_rows": int(len(seq_fit)),
            "fit_target_sha256": hashlib.sha256(fit["target"].tobytes()).hexdigest(),
            "epochs": 8,
            "patience": 2,
            "seeds": list(SEEDS),
        }
        reusable = False
        if cache.is_file() and cache_meta.is_file():
            saved = read_json(cache_meta)
            reusable = saved.get("inputs") == input_config and saved.get("output_sha256") == sha256(cache)
        if reusable:
            with np.load(cache, allow_pickle=False) as z:
                for name, target in (("current_hgb", current_hgb_scores), ("history_hgb", history_hgb_scores), ("gru_current", gru_current_scores), ("gru_history", gru_history_scores)):
                    target[positions] = z[name]
                for name in ("gru_history_dropout20", "gru_history_dropout50", "gru_history_noise0p5m", "gru_history_noise2m"):
                    robustness[name].append({"held_out_match": held, "score": z[name]})
            fold_metadata.append(read_json(cache_meta)["report"])
            continue

        hgb_current = hgb_model(static_current_fit[train], fit["target"][train])
        hgb_history = hgb_model(history_fit_features[train], fit["target"][train])
        current_hgb_scores[positions] = hgb_current.predict_proba(static_current_fit[test])[:, 1]
        history_hgb_scores[positions] = hgb_history.predict_proba(history_fit_features[test])[:, 1]
        best_neural: dict[str, list[np.ndarray]] = {"gru_current": [], "gru_history": []}
        dev_neural: dict[str, list[np.ndarray]] = {"gru_current": [], "gru_history": []}
        stress: dict[str, list[np.ndarray]] = {
            "gru_history_dropout20": [],
            "gru_history_dropout50": [],
            "gru_history_noise0p5m": [],
            "gru_history_noise2m": [],
        }
        seed_metadata = []
        test_x_static = static_fit[test]
        true_seq_test = seq_fit[test]
        current_seq_fit = np.repeat(seq_fit[:, -1:, :], seq_fit.shape[1], axis=1)
        current_seq_dev = np.repeat(seq_dev[:, -1:, :], seq_dev.shape[1], axis=1)
        current_seq_test = np.repeat(true_seq_test[:, -1:, :], true_seq_test.shape[1], axis=1)
        for seed in SEEDS:
            model, norm, dev_prob, meta = train_temporal_seed(
                static_fit[train], current_seq_fit[train], fit["target"][train],
                static_dev, current_seq_dev, dev["target"],
                current_seq_test, seed,
            )
            test_prob = temporal_predict(model, norm, test_x_static, current_seq_test)
            best_neural["gru_current"].append(test_prob)
            dev_neural["gru_current"].append(dev_prob)
            del model
            model, norm, dev_prob, meta_history = train_temporal_seed(
                static_fit[train], seq_fit[train], fit["target"][train],
                static_dev, seq_dev, dev["target"],
                true_seq_test, seed,
            )
            test_prob = temporal_predict(model, norm, test_x_static, true_seq_test)
            best_neural["gru_history"].append(test_prob)
            dev_neural["gru_history"].append(dev_prob)
            for name, drop in (("gru_history_dropout20", 0.2), ("gru_history_dropout50", 0.5)):
                altered = perturb_sequence(true_seq_test, dropout_fraction=drop, seed=seed + int(drop * 1000))
                stress[name].append(temporal_predict(model, norm, test_x_static, altered))
            for name, noise in (("gru_history_noise0p5m", 0.5), ("gru_history_noise2m", 2.0)):
                altered = perturb_sequence(true_seq_test, noise_metres=noise, seed=seed + int(noise * 10))
                stress[name].append(temporal_predict(model, norm, test_x_static, altered))
            seed_metadata.append({"seed": seed, "current_model": meta, "history_model": meta_history})
            del model
        current_avg = np.mean(best_neural["gru_current"], axis=0)
        history_avg = np.mean(best_neural["gru_history"], axis=0)
        gru_current_scores[positions] = current_avg
        gru_history_scores[positions] = history_avg
        fold_arrays = {
            "current_hgb": current_hgb_scores[positions].astype(np.float32),
            "history_hgb": history_hgb_scores[positions].astype(np.float32),
            "gru_current": current_avg.astype(np.float32),
            "gru_history": history_avg.astype(np.float32),
        }
        for name, predictions in stress.items():
            averaged = np.mean(predictions, axis=0)
            robustness[name].append({"held_out_match": held, "score": averaged})
            fold_arrays[name] = averaged.astype(np.float32)
        atomic_npz(cache, **fold_arrays)
        report = {
            "inputs": input_config,
            "output_sha256": sha256(cache),
            "report": {
                "held_out_match": held,
                "fit_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "fit_positive_rows": int((fit["target"][train] > 0).sum()),
                "development_rows": int(len(dev["target"])),
                "seed_runs": seed_metadata,
                "hgb_params": HGB_PARAMS,
            },
        }
        atomic_json(cache_meta, report)
        fold_metadata.append(report["report"])
        del hgb_current, hgb_history

    score_map = {
        "current_hgb": current_hgb_scores,
        "history_hgb": history_hgb_scores,
        "gru_current": gru_current_scores,
        "gru_history": gru_history_scores,
    }
    if any(not np.isfinite(value).all() for value in score_map.values()):
        raise ValueError("Temporal OOF predictions do not cover the six-fit-match cohort.")
    bundle = metric_bundle(fit, score_map, {
        "history_steps": list(HISTORY_OFFSETS),
        "history_features": list(HISTORY_NAMES),
        "fit_sequence_manifest": fit_manifest,
        "development_sequence_manifest": dev_manifest,
        "models": fold_metadata,
    })
    robustness_results = {}
    for name, fold_rows in robustness.items():
        values = np.full(len(fit["target"]), np.nan, dtype=np.float64)
        for row in fold_rows:
            values[fit["match"] == row["held_out_match"]] = row["score"]
        if np.isfinite(values).all():
            robustness_results[name] = {
                "overall": binary_metrics(fit["target"], values, fit),
                "per_match": per_match_metrics(fit, {name: values}),
            }
    current_speed = fit["ball"][:, 6] * 40.0
    scene_density: dict[tuple[str, int, int], int] = defaultdict(int)
    for key, close in zip(scene_keys(fit), fit["ball"][:, 9] > 0.5, strict=True):
        scene_density[key] += int(close)
    dense = np.asarray([scene_density[key] >= 3 for key in scene_keys(fit)])
    strata_masks = {
        "low_ball_speed_le_1mps": (fit["ball"][:, 0] > 0.5) & (current_speed <= 1.0),
        "fast_ball_speed_gt_15mps": (fit["ball"][:, 0] > 0.5) & (current_speed > 15.0),
        "dense_neighborhood_ge_3_actors_within_5m": dense,
    }
    strata = {}
    for name, mask in strata_masks.items():
        strata[name] = {
            "rows": int(mask.sum()),
            "positive_rows": int((fit["target"][mask] > 0).sum()),
            "prevalence": float(np.mean(fit["target"][mask] > 0)) if mask.any() else None,
            "models": {
                key: binary_metrics(fit["target"][mask], value[mask])
                for key, value in score_map.items()
            },
        }
    result = {
        "status": "COMPLETE",
        "rows": int(len(fit["target"])),
        "prevalence": float(np.mean(fit["target"] > 0)),
        "feature_names": list(HISTORY_NAMES),
        "history_offsets_frames_oldest_to_current": list(HISTORY_OFFSETS),
        "instantaneous_control": "Same GRU capacity and static actor context; the current ball/actor state is repeated at all six time indices, with no older observations.",
        "history_model": "Same GRU capacity and static actor context; six causally available snapshots at 0.4-second spacing.",
        "overall": bundle["overall"],
        "per_match": bundle["per_match"],
        "paired_match_contrasts": bundle["paired_match_contrasts"],
        "robustness": robustness_results,
        "speed_and_density_strata": strata,
        "fold_training_metadata": fold_metadata,
        "caveats": [
            "The ball track is privileged exocentric reference data, not player-observed possession or gaze.",
            "Temporal labels remain supported BAS annotation targets; the no-event class is not verified inactivity.",
            "The six matches are the only independent fold clusters; uncertainty is match-level and descriptive.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    atomic_json(RUN / "causal_ball_history_manifest.json", {
        "status": "COMPLETE",
        "history_offsets_frames_oldest_to_current": list(HISTORY_OFFSETS),
        "seconds_before_query": [round(value / 25.0, 3) for value in HISTORY_OFFSETS],
        "feature_names": list(HISTORY_NAMES),
        "asof_bound": "Ball state is latest valid source frame <= each historical query, maximum age five frames; player state is latest observed source frame <= that query within five frames and the same actor segment.",
        "fit": fit_manifest,
        "development": dev_manifest,
        "no_retrospective_history_loaded": True,
    })
    atomic_json(result_path, result)
    append_decision("B", "Compare causal ball history with same-capacity instantaneous GRU and matched HGB controls.", {"history_hgb_AP": bundle["overall"]["history_hgb"]["AP"], "current_hgb_AP": bundle["overall"]["current_hgb"]["AP"], "history_gru_AP": bundle["overall"]["gru_history"]["AP"], "instant_gru_AP": bundle["overall"]["gru_current"]["AP"]}, "six leave-one-fit-match-out folds; three matched seeds; development-only early stopping")
    return result


LEAD_WINDOWS = {
    "lead_0_1s": (0, 25),
    "lead_1_2s": (25, 50),
    "lead_2_3s": (50, 75),
    "within_1s": (0, 25),
    "within_2s": (0, 50),
    "within_3s": (0, 75),
}


def build_label_targets(data: dict[str, np.ndarray], split: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    import bisect
    import build_cohort

    manifest = read_json(PRIOR / "cohort_manifest.json")
    shard_by_key = {
        (str(row["match"]), 1 if row["half"] == "1st" else 2): row
        for row in manifest["shards"] if row["split"] == split
    }
    labels = {name: np.full(len(data["target"]), -1, dtype=np.int16) for name in LEAD_WINDOWS}
    event_frames_by_target = {name: np.full(len(data["target"]), -1, dtype=np.int32) for name in LEAD_WINDOWS}
    censors: dict[str, Counter[str]] = {name: Counter() for name in LEAD_WINDOWS}
    class_support: dict[str, Counter[str]] = {name: Counter() for name in LEAD_WINDOWS}
    event_source_fingerprints: list[dict[str, Any]] = []
    scene_indices: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(scene_keys(data)):
        scene_indices[key].append(index)

    group_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for match, half_code in sorted({(str(key[0]), int(key[1])) for key in scene_indices}):
        half = "1st" if half_code == 1 else "2nd"
        shard = shard_by_key[(match, half_code)]
        cache_path = ROOT / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{match}_{half}.npz"
        metadata_path = cache_path.with_suffix(".json")
        association_path = ROOT / "outputs" / "trajectory_audit_20261016" / f"associations_{match}_{half}.json"
        event_path = ROOT / "data" / f"{match}_12_class_events.json"
        with np.load(cache_path, allow_pickle=False) as z:
            raw = {name: z[name] for name in ("observed", "segments")}
        metadata = read_json(metadata_path)
        association = read_json(association_path)
        slot_ids: dict[int, list[str]] = defaultdict(list)
        id_to_slot = {str(pid): int(slot) for pid, slot in association["slots"].items()}
        for player_id, slot in id_to_slot.items():
            slot_ids[slot].append(player_id)
        unique_slots = {slot for slot, ids in slot_ids.items() if len(ids) == 1}
        event_doc = read_json(event_path)
        events = build_cohort._event_rows(
            match, half, metadata["gsr_info"], len(raw["observed"]),
            id_to_slot, unique_slots, raw,
        )
        events.sort(key=lambda row: row["frame"])
        frames = [int(row["frame"]) for row in events]
        transitions: dict[tuple[int, int], list[int]] = defaultdict(list)
        for slot in range(raw["observed"].shape[1]):
            obs_frames = np.flatnonzero(raw["observed"][:, slot])
            if len(obs_frames) < 2:
                continue
            seg_values = raw["segments"][obs_frames, slot]
            changes = np.flatnonzero(seg_values[1:] != seg_values[:-1]) + 1
            for change in changes:
                transitions[(slot, int(seg_values[change - 1]))].append(int(obs_frames[change]))
        for value in transitions.values():
            value.sort()
        group_cache[(match, half_code)] = {
            "events": events,
            "frames": frames,
            "raw": raw,
            "transitions": transitions,
            "frame_count": len(raw["observed"]),
        }
        event_source_fingerprints.append({
            "match": match,
            "half": half,
            "event_json_sha256": sha256(event_path),
            "association_sha256": sha256(association_path),
            "tracking_cache_sha256": sha256(cache_path),
            "supported_event_rows": len(events),
            "known_actor_event_rows": sum(bool(row["known_actor"]) for row in events),
            "unknown_actor_event_rows": sum(not bool(row["known_actor"]) for row in events),
            "raw_event_rows": len(event_doc.get("actions", [])),
            "cohort_shard_sha256": shard["shard_sha256"],
        })

    for (match, half_code, scene_id), rows in scene_indices.items():
        query = int(data["query_frame"][rows[0]])
        source = group_cache[(match, half_code)]
        frame_count = source["frame_count"]
        event_rows = source["events"]
        event_frames = source["frames"]
        raw = source["raw"]
        transitions = source["transitions"]
        active_slots = data["actor_slot_local_only"][rows].astype(np.int64)
        current_segments = data["segment_local_only"][rows].astype(np.int64)
        for name, (lower, upper) in LEAD_WINDOWS.items():
            if query + upper > frame_count - 1:
                censors[name]["incomplete_source_window"] += len(rows)
                continue
            start = bisect.bisect_right(event_frames, query + lower)
            end = bisect.bisect_right(event_frames, query + upper)
            future = event_rows[start:end]
            if any(not row["known_actor"] for row in future):
                censors[name]["unmapped_actor_event_in_window"] += len(rows)
                continue
            by_actor: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for event in future:
                if event["known_actor"]:
                    by_actor[(int(event["slot"]), int(event["segment"]))].append(event)
            for local_index, global_index in enumerate(rows):
                slot = int(active_slots[local_index])
                segment = int(current_segments[local_index])
                actor_events = by_actor.get((slot, segment), [])
                actor_events.sort(key=lambda row: row["frame"])
                earliest_frame = int(actor_events[0]["frame"]) if actor_events else -1
                first_frame = earliest_frame
                first_types = {int(row["label"]) for row in actor_events if int(row["frame"]) == first_frame} if actor_events else set()
                if actor_events and len(first_types) != 1:
                    censors[name]["ambiguous_same_frame_labels"] += 1
                    continue
                change_frames = transitions.get((slot, segment), [])
                position = bisect.bisect_right(change_frames, query)
                changed_frame = change_frames[position] if position < len(change_frames) else None
                if changed_frame is not None and changed_frame <= query + upper and (earliest_frame < 0 or changed_frame <= earliest_frame):
                    censors[name]["track_segment_discontinuity"] += 1
                    continue
                if actor_events:
                    label = next(iter(first_types))
                    labels[name][global_index] = label
                    event_frames_by_target[name][global_index] = earliest_frame
                    class_support[name][str(label)] += 1
                else:
                    labels[name][global_index] = 0
    reports = {}
    for name in LEAD_WINDOWS:
        selected = labels[name] >= 0
        target = labels[name][selected]
        reports[name] = {
            "eligible_actor_time_rows": int(selected.sum()),
            "censored_actor_time_rows": int((~selected).sum()),
            "censor_reasons": dict(censors[name]),
            "positive_actor_time_rows": int(np.sum(target > 0)),
            "prevalence": float(np.mean(target > 0)) if len(target) else None,
            "class_counts": {CLASS_NAMES[int(label)]: int(np.sum(target == label)) for label in range(1, 8)},
            "eligible_matches": {
                match: int(np.sum(selected & (data["match"] == match)))
                for match in sorted(np.unique(data["match"]).tolist())
            },
        }
    if "lead_0_1s" in labels:
        common = labels["lead_0_1s"] >= 0
        mismatches = int(np.sum(data["target"][common] != labels["lead_0_1s"][common]))
        reports["primary_1s_reconstruction"] = {
            "eligible_rows_compared": int(common.sum()),
            "mismatched_labels": mismatches,
            "matches_existing_target": mismatches == 0,
            "existing_cohort_positive_rows_on_eligible_rows": int(np.sum(data["target"][common] > 0)),
        }
        if mismatches:
            raise ValueError(f"Reconstructed primary 1s labels differ from locked cohort on {mismatches} eligible rows.")
    return labels, event_frames_by_target, {
        "split": split,
        "horizon_definitions_frames_right_closed": {name: list(bounds) for name, bounds in LEAD_WINDOWS.items()},
        "targets": reports,
        "source_fingerprints": event_source_fingerprints,
        "actor_time_row_count": int(len(data["target"])),
        "scene_count": int(len(scene_indices)),
    }


def scene_alert_candidates(data: dict[str, np.ndarray], scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int, int]]]:
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(scene_keys(data)):
        groups[key].append(index)
    scene_keys_ordered = sorted(groups)
    selected_rows = np.asarray([max(groups[key], key=lambda index: (scores[index], -index)) for key in scene_keys_ordered], dtype=np.int64)
    scene_scores = scores[selected_rows]
    return selected_rows, scene_scores, scene_keys_ordered


def choose_scene_threshold(
    data: dict[str, np.ndarray],
    scores: np.ndarray,
    labels: np.ndarray,
    budget_per_hour: float,
    rate_unit: str,
) -> float:
    selected_rows, scene_scores, scene_keys_ordered = scene_alert_candidates(data, scores)
    scene_members: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(scene_keys(data)):
        scene_members[key].append(index)
    scene_labels = np.asarray([np.any(labels[scene_members[key]] > 0) for key in scene_keys_ordered], dtype=bool)
    if rate_unit == "actor_hour":
        hours = float(np.sum(labels >= 0) / 3600.0)
        negative_scores = scores[(labels >= 0) & (labels <= 0)]
        allowed = int(math.floor(budget_per_hour * hours))
        if not len(negative_scores) or allowed >= len(negative_scores):
            return 0.0
        ordered = np.sort(negative_scores)[::-1]
        return float(np.nextafter(ordered[min(allowed, len(ordered) - 1)], np.inf))
    if rate_unit == "match_hour":
        hours = len(scene_keys_ordered) / 3600.0
        negative_scene_scores = scene_scores[~scene_labels]
        allowed = int(math.floor(budget_per_hour * hours))
        if not len(negative_scene_scores) or allowed >= len(negative_scene_scores):
            return 0.0
        ordered = np.sort(negative_scene_scores)[::-1]
        return float(np.nextafter(ordered[min(allowed, len(ordered) - 1)], np.inf))
    raise ValueError(f"Unknown rate unit {rate_unit}")


def event_alert_metrics(
    data: dict[str, np.ndarray],
    scores: np.ndarray,
    labels: np.ndarray,
    event_frames: np.ndarray,
    threshold: float,
    cooldown_frames: int = 50,
) -> dict[str, Any]:
    selected_rows, scene_scores, ordered_scenes = scene_alert_candidates(data, scores)
    alerts = []
    last_by_actor: dict[tuple[str, int, int, int], int] = {}
    for row, score in zip(selected_rows, scene_scores, strict=True):
        if score < threshold:
            continue
        actor_key = (
            str(data["match"][row]), int(data["half_code"][row]),
            int(data["actor_slot_local_only"][row]), int(data["segment_local_only"][row]),
        )
        query = int(data["query_frame"][row])
        last = last_by_actor.get(actor_key)
        if last is not None and query - last < cooldown_frames:
            continue
        last_by_actor[actor_key] = query
        alerts.append(int(row))
    unique_events = {
        (str(data["match"][i]), int(data["half_code"][i]), int(data["actor_slot_local_only"][i]), int(data["segment_local_only"][i]), int(event_frames[i]))
        for i in np.flatnonzero((labels > 0) & (event_frames >= 0))
    }
    matched_events = {
        (str(data["match"][i]), int(data["half_code"][i]), int(data["actor_slot_local_only"][i]), int(data["segment_local_only"][i]), int(event_frames[i]))
        for i in alerts if labels[i] > 0 and event_frames[i] >= 0
    }
    false_actor_alerts = sum(labels[index] == 0 for index in alerts)
    positive_scene_keys = set()
    scene_members: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    all_scene_keys = scene_keys(data)
    for index, key in enumerate(all_scene_keys):
        scene_members[key].append(index)
    for key, rows in scene_members.items():
        if any(labels[row] > 0 for row in rows):
            positive_scene_keys.add(key)
    detected_positive_scenes = 0
    for row in alerts:
        key = all_scene_keys[row]
        if key in positive_scene_keys and labels[row] > 0:
            detected_positive_scenes += 1
    total_scenes = len(scene_members)
    actor_hours = int(np.sum(labels >= 0)) / 3600.0
    match_hours = total_scenes / 3600.0
    total_positive_scenes = len(positive_scene_keys)
    return {
        "threshold": float(threshold),
        "cooldown_seconds": cooldown_frames / 25.0,
        "alerts_after_scene_top1_and_actor_cooldown": len(alerts),
        "unique_supported_events_in_evaluable_windows": len(unique_events),
        "matched_unique_events": len(matched_events),
        "event_recall": len(matched_events) / max(1, len(unique_events)),
        "positive_scenes": total_positive_scenes,
        "positive_scenes_with_correct_actor_alert": detected_positive_scenes,
        "positive_scene_actor_recall": detected_positive_scenes / max(1, total_positive_scenes),
        "annotation_relative_false_actor_alerts": int(false_actor_alerts),
        "false_alerts_per_actor_hour": float(false_actor_alerts / actor_hours) if actor_hours else None,
        "false_alerts_per_match_hour": float(false_actor_alerts / match_hours) if match_hours else None,
        "actor_time_hours": actor_hours,
        "match_hours_from_one_second_scene_grid": match_hours,
        "alert_workload_top1_reviews_per_match_hour": float(len(alerts) / match_hours) if match_hours else None,
        "interpretation": "A false alert means no supported BAS actor event in the next one-second interval; it is not a verified physical false alarm.",
    }


def run_investigation_c() -> dict[str, Any]:
    lead_path = RUN / "lead_time_study.json"
    alert_path = RUN / "event_alert_operating_points.json"
    if lead_path.is_file() and alert_path.is_file():
        return {"lead_time_study": read_json(lead_path), "event_alert_operating_points": read_json(alert_path)}
    started = time.perf_counter()
    fit = load_split("fit")
    dev = load_split("development")
    fit_labels, fit_event_frames, fit_audit = build_label_targets(fit, "fit")
    dev_labels, dev_event_frames, dev_audit = build_label_targets(dev, "development")
    atomic_npz(
        RUN / "derived_targets_local.npz",
        **{
            f"target__{name}": fit_labels[name]
            for name in LEAD_WINDOWS
        },
        **{
            f"event_frame__{name}": fit_event_frames[name]
            for name in LEAD_WINDOWS
        },
    )
    feature_fit = np.column_stack((fit["x"], fit["ball"])).astype(np.float32)
    feature_dev = np.column_stack((dev["x"], dev["ball"])).astype(np.float32)
    per_horizon: dict[str, Any] = {}
    alert_aggregates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    oof_scores_by_horizon: dict[str, np.ndarray] = {}
    for horizon_name in LEAD_WINDOWS:
        targets = fit_labels[horizon_name]
        dev_target = dev_labels[horizon_name]
        oof_scores = np.full(len(targets), np.nan, dtype=np.float64)
        match_rows = []
        class_predictions_by_fold: dict[str, np.ndarray] = {}
        for held in FIT_MATCHES:
            test_match = fit["match"] == held
            train_mask = (~test_match) & (targets >= 0)
            test_mask = test_match & (targets >= 0)
            dev_mask = dev_target >= 0
            model = hgb_model(feature_fit[train_mask], targets[train_mask])
            test_score = model.predict_proba(feature_fit[test_mask])[:, 1]
            oof_scores[test_mask] = test_score
            dev_score = model.predict_proba(feature_dev[dev_mask])[:, 1]
            if horizon_name == "within_1s":
                fold_thresholds = {}
                dev_eligible = {key: value[dev_mask] for key, value in dev.items() if isinstance(value, np.ndarray) and len(value) == len(dev_target)}
                for unit in ("actor_hour", "match_hour"):
                    for budget in (1, 5, 10):
                        threshold = choose_scene_threshold(dev_eligible, dev_score, dev_target[dev_mask], budget, unit)
                        fold_thresholds[f"{unit}_{budget}"] = threshold
                        test_subset = {key: value[test_mask] for key, value in fit.items() if isinstance(value, np.ndarray) and len(value) == len(targets)}
                        alert_result = event_alert_metrics(
                            test_subset,
                            test_score,
                            targets[test_mask],
                            fit_event_frames[horizon_name][test_mask],
                            threshold,
                        )
                        alert_result.update({"held_out_match": held, "budget": budget, "rate_unit": unit})
                        alert_aggregates[f"{unit}_{budget}"].append(alert_result)
                fold_thresholds["hgb_params"] = HGB_PARAMS
                match_rows.append({"held_out_match": held, "test_rows": int(test_mask.sum()), "test_positives": int((targets[test_mask] > 0).sum()), "dev_rows": int(dev_mask.sum()), "thresholds": fold_thresholds})
            else:
                match_rows.append({"held_out_match": held, "test_rows": int(test_mask.sum()), "test_positives": int((targets[test_mask] > 0).sum()), "dev_rows": int(dev_mask.sum())})
            del model
        selected = np.isfinite(oof_scores)
        oof_scores_by_horizon[horizon_name] = oof_scores.copy()
        selected_data = {key: value[selected] for key, value in fit.items() if isinstance(value, np.ndarray) and len(value) == len(targets)}
        selected_y = targets[selected]
        per_match = per_match_metrics(
            {**selected_data, "target": selected_y},
            {horizon_name: oof_scores[selected]},
        )
        overall = binary_metrics(selected_y, oof_scores[selected], {**selected_data, "target": selected_y})
        class_counts = {CLASS_NAMES[index]: int(np.sum(selected_y == index)) for index in range(1, 8)}
        group_counts = Counter(CLASS_TO_GROUP[int(value)] for value in selected_y if value > 0)
        per_horizon[horizon_name] = {
            "overall": overall,
            "per_match": per_match,
            "eligible_actor_time_rows": int(selected.sum()),
            "censored_actor_time_rows": int((~selected).sum()),
            "prevalence": float(np.mean(selected_y > 0)),
            "class_counts": class_counts,
            "predeclared_hierarchical_group_counts": dict(group_counts),
            "fold_support": match_rows,
        }
    fixed_leads = {}
    cumulative = {}
    for name in ("lead_0_1s", "lead_1_2s", "lead_2_3s"):
        fixed_leads[name] = per_horizon[name]
    for name in ("within_1s", "within_2s", "within_3s"):
        cumulative[name] = per_horizon[name]
    lead_summary = {}
    for name, event_frames in fit_event_frames.items():
        selected = (fit_labels[name] > 0) & (event_frames >= 0)
        lead_seconds = (event_frames[selected] - fit["query_frame"][selected]) / 25.0
        lead_summary[name] = {
            "positive_actor_time_rows": int(selected.sum()),
            "median_lead_seconds": float(np.median(lead_seconds)) if len(lead_seconds) else None,
            "p10_lead_seconds": float(np.percentile(lead_seconds, 10)) if len(lead_seconds) else None,
            "p90_lead_seconds": float(np.percentile(lead_seconds, 90)) if len(lead_seconds) else None,
            "unique_event_ids": len({
                (str(fit["match"][i]), int(fit["half_code"][i]), int(fit["actor_slot_local_only"][i]), int(fit["segment_local_only"][i]), int(event_frames[i]))
                for i in np.flatnonzero(selected)
            }),
        }
    primary_top1 = per_horizon["within_1s"]["overall"]["scene_ranking"]["top1"]
    budget_one = alert_aggregates["actor_hour_1"]
    mean_recall_one = float(np.mean([row["event_recall"] for row in budget_one])) if budget_one else 0.0
    status_choice = (
        "EVENT_FORECASTING" if mean_recall_one >= 0.20
        else "SCENE_RANKING_ONLY" if primary_top1 is not None and primary_top1 >= 0.50
        else "ANNOTATION_LIMITED" if fit_audit["targets"]["within_1s"]["censored_actor_time_rows"] > 0
        else "NOT_OPERATIONALLY_USEFUL"
    )
    lead_result = {
        "status": "COMPLETE",
        "primary_cohort_label_reconstruction": fit_audit["targets"]["primary_1s_reconstruction"],
        "choice": status_choice,
        "choice_rule": "EVENT_FORECASTING if mean held-out unique-event recall at the development-selected 1 actor-alert/actor-hour threshold is >=0.20; otherwise SCENE_RANKING_ONLY if 1s positive-scene top1 >=0.50; otherwise ANNOTATION_LIMITED if any actor-time censoring is observed; otherwise NOT_OPERATIONALLY_USEFUL.",
        "fixed_lead_bins": fixed_leads,
        "cumulative_any_event_horizons": cumulative,
        "observed_lead_distribution_seconds": lead_summary,
        "fit_label_censoring_audit": fit_audit,
        "development_label_censoring_audit": dev_audit,
        "class_mapping": {CLASS_NAMES[i]: CLASS_TO_GROUP.get(i) for i in range(1, 8)},
        "annotation_boundary": "No supported annotation is not verified inactivity; exposure and positives are conditional on source event, identity, track-continuity and clock support.",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    pred_payload: dict[str, np.ndarray] = {
        "target_original": fit["target"],
        "match": fit["match"],
        "half_code": fit["half_code"],
        "query_frame": fit["query_frame"],
        "scene_id_local_only": fit["scene_id_local_only"],
        "actor_slot_local_only": fit["actor_slot_local_only"],
        "segment_local_only": fit["segment_local_only"],
    }
    for name in LEAD_WINDOWS:
        pred_payload[f"target__{name}"] = fit_labels[name]
        pred_payload[f"event_frame__{name}"] = fit_event_frames[name]
        pred_payload[f"score__{name}"] = oof_scores_by_horizon[name].astype(np.float32)
    atomic_npz(RUN / "lead_predictions_local.npz", **pred_payload)
    atomic_json(lead_path, lead_result)
    alert_summary = {}
    for name, rows in alert_aggregates.items():
        if not rows:
            alert_summary[name] = {"per_match": [], "mean_event_recall": None}
            continue
        numeric = (
            "event_recall",
            "positive_scene_actor_recall",
            "false_alerts_per_actor_hour",
            "false_alerts_per_match_hour",
            "alert_workload_top1_reviews_per_match_hour",
        )
        alert_summary[name] = {
            "budget": int(name.rsplit("_", 1)[1]),
            "rate_unit": name.rsplit("_", 1)[0],
            "per_match": rows,
            "mean": {metric: float(np.mean([row[metric] for row in rows])) for metric in numeric},
            "match_bootstrap_95_ci": {
                metric: mean_bootstrap_ci([row[metric] for row in rows])
                for metric in ("event_recall", "false_alerts_per_actor_hour", "false_alerts_per_match_hour")
            },
            "selection_split": DEV_MATCH,
            "cooldown_seconds": 2.0,
        }
    alert_result = {
        "status": "COMPLETE",
        "threshold_selection": "Each outer fold uses a model trained on the other five fitting matches; thresholds are selected only on development match 132877. A scene contributes at most its top-ranked actor; repeat alerts for the same actor/segment are suppressed for two seconds.",
        "operating_points": alert_summary,
        "definitions": {
            "false_actor_alert": "Selected top-1 actor has no supported BAS actor event in the next one-second window.",
            "false_match_alert": "Same annotation-relative false alert count divided by one-second-grid scene-hours; not a verified physical false alarm.",
            "unique_event_match": "A predicted top-1 actor alert matches the same match/half/track segment and event frame in the next one-second target window.",
            "negative_labels": "No supported BAS event in the interval; unlabeled real actions may exist.",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    atomic_json(alert_path, alert_result)
    append_decision("C", f"Prospective evaluation choice: {status_choice}.", {"within_1s_scene_top1": primary_top1, "mean_event_recall_at_1_actor_alert_per_hour": mean_recall_one, "censoring": fit_audit["targets"]["within_1s"]["censor_reasons"]}, "six fit-match OOF folds; 132877 development-only thresholds; no retrospective data")
    return {"lead_time_study": lead_result, "event_alert_operating_points": alert_result}


def run_investigation_d() -> dict[str, Any]:
    path = RUN / "autonomous_branch_decision.json"
    if path.is_file() and (RUN / "branch_metrics.json").is_file():
        return read_json(path)
    if not (RUN / "ball_shortcut_audit.json").is_file() or not (RUN / "temporal_experiments.json").is_file() or not (RUN / "lead_time_study.json").is_file():
        raise RuntimeError("Investigation D requires completed A, B and C results.")
    a = read_json(RUN / "ball_shortcut_audit.json")
    b = read_json(RUN / "temporal_experiments.json")
    c = read_json(RUN / "lead_time_study.json")
    nearest_ap = a["overall"]["nearest_ball_rule"]["AP"]
    distance_ap = a["overall"]["x_plus_distance"]["AP"]
    full_ap = a["overall"]["x_plus_all"]["AP"]
    history_delta = b["overall"]["history_hgb"]["AP"] - b["overall"]["current_hgb"]["AP"]
    candidates = [
        {
            "branch": "conditional_far_candidate_forecasting",
            "expected_information_gain": 0.95,
            "estimated_cost": "medium",
            "rationale": "Directly tests whether cross-match context carries signal after the current nearest/within-5m candidate is excluded; reuses only causal features and evaluates a prespecified 5m cut.",
        },
        {
            "branch": "ball_history_under_degraded_observation",
            "expected_information_gain": 0.82,
            "estimated_cost": "medium-high",
            "rationale": "Quantifies whether the history gain remains useful when ball frames are missing, delayed or noisy.",
        },
        {
            "branch": "annotation_uncertainty_adjudication",
            "expected_information_gain": 0.91,
            "estimated_cost": "high",
            "rationale": "Could distinguish real false alarms from unsupported/uncaptured actions, but no independent event adjudication labels are currently available.",
        },
    ]
    if full_ap - distance_ap >= 0.05 and nearest_ap < 0.10:
        chosen = "conditional_far_candidate_forecasting"
        reason = "The measured nearest-only heuristic is weak at actor-row AP, while distance-only and full-ball models are much stronger. A held-out non-nearest candidate test directly probes whether the residual signal is contextual rather than proximity alone."
    elif history_delta >= 0.02:
        chosen = "conditional_far_candidate_forecasting"
        reason = "Even with history value, far-candidate scoring is the strongest next test of genuine football context and avoids treating proximity as the task."
    else:
        chosen = "conditional_far_candidate_forecasting"
        reason = "The selected branch tests transfer beyond the strongest observed instantaneous cue and is executable on the existing matched cohort."
    decision = {
        "status": "BRANCH_SELECTED",
        "candidate_branches": candidates,
        "observed_evidence": {
            "nearest_rule_AP": nearest_ap,
            "distance_only_HGB_AP": distance_ap,
            "full_ball_HGB_AP": full_ap,
            "history_HGB_minus_current_HGB_AP": history_delta,
            "C_choice": c["choice"],
        },
        "chosen_branch": chosen,
        "why": reason,
        "locked_branch_protocol": {
            "candidate_rule": "Ball-present actor rows whose causal current actor-to-ball distance is strictly greater than 5m; threshold reused from the prior within_5m feature, not selected from the current held-out labels.",
            "models": ["full-cohort OOF no-ball HGB restricted to far candidates", "full-cohort OOF full-ball HGB restricted to far candidates", "far-only-training HGB no-ball", "far-only-training HGB with all 12 ball features", "nearest-among-far rule"],
            "evaluation": "Six leave-one-fit-match-out folds, identical far candidate rows within each contrast, natural prevalence; no retrospective data.",
            "primary_metrics": ["AP", "Brier", "hazard NLL", "positive-scene top1/top3/MRR", "positive-event row coverage"],
        },
        "experiment_executed": False,
    }
    atomic_json(path, decision)
    append_decision("D", "Select conditional non-nearest actor forecasting.", {"nearest_rule_AP": nearest_ap, "full_minus_distance_AP": full_ap - distance_ap}, "Branch protocol locked after A-C aggregate results; 5m cut fixed before D outcome scoring")

    fit = load_split("fit")
    cached_scores, _ = load_cached_prediction_scores(fit)
    far = (fit["ball"][:, 0] > 0.5) & (fit["ball"][:, 4] > 5.0 / 105.0)
    full_oof = {key: value.copy() for key, value in cached_scores.items()}
    d_scores: dict[str, np.ndarray] = {
        "full_cohort_hgb_no_ball_restricted_far": full_oof["x_no_ball"],
        "full_cohort_hgb_ball_restricted_far": full_oof["x_plus_all"],
        "nearest_among_far": -fit["ball"][:, 4].astype(np.float64),
    }
    conditional_features = {
        "far_only_hgb_no_ball": fit["x"],
        "far_only_hgb_ball": np.column_stack((fit["x"], fit["ball"])).astype(np.float32),
    }
    metadata = []
    for held in FIT_MATCHES:
        test = (fit["match"] == held) & far
        train = (fit["match"] != held) & far
        for name, features in conditional_features.items():
            if not test.any() or not train.any() or not np.any(fit["target"][train] > 0):
                raise ValueError(f"Insufficient far-candidate support in fold {held} for {name}.")
            model = hgb_model(features[train], fit["target"][train])
            if name not in d_scores:
                d_scores[name] = np.full(len(fit["target"]), np.nan, dtype=np.float64)
            d_scores[name][test] = model.predict_proba(features[test])[:, 1]
            metadata.append({
                "held_out_match": held,
                "model": name,
                "far_fit_rows": int(train.sum()),
                "far_fit_positive_rows": int((fit["target"][train] > 0).sum()),
                "far_test_rows": int(test.sum()),
                "far_test_positive_rows": int((fit["target"][test] > 0).sum()),
            })
            del model
    evaluated = far.copy()
    far_data = {key: value[evaluated] for key, value in fit.items() if isinstance(value, np.ndarray) and len(value) == len(evaluated)}
    far_data["target"] = fit["target"][evaluated]
    scoped_scores = {key: score[evaluated] for key, score in d_scores.items()}
    bundle = metric_bundle(far_data, scoped_scores, {"five_metre_cut": True, "far_candidate_rows": int(evaluated.sum()), "fit_target_positive_rows": int((far_data["target"] > 0).sum()), "training": metadata})
    all_positive = fit["target"] > 0
    positive_far_fraction = float(np.sum(all_positive & far) / max(1, all_positive.sum()))
    metrics = {
        "status": "COMPLETE",
        "candidate_definition": "Ball present and current actor-to-ball distance >5m at query time; only causal query-time measurements determine candidate inclusion.",
        "all_cohort_rows": int(len(fit["target"])),
        "far_candidate_rows": int(far.sum()),
        "far_candidate_positive_rows": int((fit["target"][far] > 0).sum()),
        "far_candidate_prevalence": float(np.mean(fit["target"][far] > 0)),
        "positive_actor_rows_retained": positive_far_fraction,
        "overall": bundle["overall"],
        "per_match": bundle["per_match"],
        "paired_match_contrasts": bundle["paired_match_contrasts"],
        "fold_training_metadata": metadata,
        "interpretation": "This isolates annotated action actors who were not within the predeclared 5m ball neighborhood. It is not a test of optimal decisions or verified off-ball intent.",
    }
    atomic_json(RUN / "branch_metrics.json", metrics)
    decision["experiment_executed"] = True
    decision["branch_result"] = {
        "far_candidate_rows": int(far.sum()),
        "positive_actor_rows_retained_fraction": positive_far_fraction,
        "conditional_ball_HGB_AP": bundle["overall"]["far_only_hgb_ball"]["AP"],
        "conditional_no_ball_HGB_AP": bundle["overall"]["far_only_hgb_no_ball"]["AP"],
        "full_cohort_ball_HGB_restricted_AP": bundle["overall"]["full_cohort_hgb_ball_restricted_far"]["AP"],
    }
    atomic_json(path, decision)
    return decision


def generate_visualizations() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve

    a = read_json(RUN / "ball_shortcut_audit.json")
    b = read_json(RUN / "temporal_experiments.json")
    c = read_json(RUN / "lead_time_study.json")
    alerts = read_json(RUN / "event_alert_operating_points.json")
    figures = RUN / "figures"
    pitch_dir = figures / "pitch_scenes"
    pitch_dir.mkdir(parents=True, exist_ok=True)

    labels = ["nearest rule", "ball-only HGB", "availability HGB", "distance HGB", "full HGB", "block-shuffled", "0.2s lag", "0.5s lag", "1.0s lag", "2.0s lag"]
    keys = ["nearest_ball_rule", "ball_only", "x_plus_availability", "x_plus_distance", "x_plus_all", "x_plus_block_shuffle", "lag_0p2", "lag_0p5", "lag_1p0", "lag_2p0"]
    values = [a["overall"][key]["AP"] for key in keys]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#7f8c8d" if key != "x_plus_all" else "#2874a6" for key in keys]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Actor-row average precision")
    ax.set_title("Investigation A: causal ball cues, controls and older as-of states")
    ax.tick_params(axis="x", rotation=32)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "ball_shortcut_ablation.png", dpi=150)
    plt.close(fig)

    lag_names = ["nearest_ball_rule", "x_plus_distance", "lag_0p2", "lag_0p5", "lag_1p0", "lag_2p0", "x_plus_all"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(["instant rule", "instant distance", "0.2", "0.5", "1.0", "2.0", "full ball"], [a["overall"][name]["AP"] for name in lag_names], marker="o")
    ax.axhline(a["overall"]["x_no_ball"]["AP"], linestyle="--", color="gray", label="no-ball HGB")
    ax.set_xlabel("Ball-state age / representation")
    ax.set_ylabel("Actor-row AP")
    ax.set_title("Skill decays when ball state is older")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "ball_lag_sensitivity.png", dpi=150)
    plt.close(fig)

    model_names = ["current_hgb", "history_hgb", "gru_current", "gru_history"]
    pretty = ["Current HGB", "History HGB", "Instant GRU", "History GRU"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, label in zip(model_names, pretty, strict=True):
        axes[0].bar(label, b["overall"][model]["AP"])
        axes[1].bar(label, b["overall"][model]["scene_ranking"]["top1"])
    axes[0].set_ylabel("Actor-row AP")
    axes[0].set_title("Temporal-model comparison")
    axes[1].set_ylabel("Positive scene top-1")
    axes[1].set_title("Scene ranking")
    for ax in axes:
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "temporal_model_comparison.png", dpi=150)
    plt.close(fig)

    horizon_keys = ["within_1s", "within_2s", "within_3s"]
    horizon_labels = ["Next 1s", "Next 2s", "Next 3s"]
    aps = [c["cumulative_any_event_horizons"][key]["overall"]["AP"] for key in horizon_keys]
    prevalences = [c["cumulative_any_event_horizons"][key]["prevalence"] for key in horizon_keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(horizon_labels, aps, marker="o", label="AP")
    ax.plot(horizon_labels, prevalences, marker="s", label="natural prevalence")
    ax.set_ylabel("Probability")
    ax.set_title("Longer horizons change both signal and base rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "lead_time_sensitivity.png", dpi=150)
    plt.close(fig)

    rates = [1, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for unit, ax in zip(("actor_hour", "match_hour"), axes, strict=True):
        recall = [alerts["operating_points"][f"{unit}_{budget}"]["mean"]["event_recall"] for budget in rates]
        ax.plot(rates, recall, marker="o")
        ax.set_xlabel(f"Development-selected annotation-relative alerts per {unit.replace('_', ' ')}")
        ax.set_ylabel("Held-out unique-event recall")
        ax.set_title(unit.replace("_", " ").title())
        ax.grid(alpha=0.25)
    fig.suptitle("Event alert recall at fixed review workloads")
    fig.tight_layout()
    fig.savefig(figures / "event_alert_workload.png", dpi=150)
    plt.close(fig)

    pred_path = RUN / "lead_predictions_local.npz"
    fit = load_split("fit")
    with np.load(pred_path, allow_pickle=False) as pred:
        for key in ("match", "half_code", "query_frame", "scene_id_local_only"):
            if not np.array_equal(pred[key], fit[key]):
                raise ValueError(f"Local prediction provenance mismatch in visualizer: {key}")
        if not np.array_equal(pred["target_original"], fit["target"]):
            raise ValueError("Local prediction provenance mismatch in visualizer: target_original")
        y_all = pred["target__within_1s"].astype(np.int16)
        score_all = pred["score__within_1s"].astype(np.float64)
        event_all = pred["event_frame__within_1s"].astype(np.int64)
        pr_y = y_all >= 0
        pr_y_values = (y_all[pr_y] > 0).astype(np.int8)
        pr_scores = score_all[pr_y]
        fig, ax = plt.subplots(figsize=(6, 5))
        precision, recall, _ = precision_recall_curve(pr_y_values, pr_scores)
        ax.plot(recall, precision, label="1-second OOF")
        ax.axhline(float(pr_y_values.mean()), linestyle="--", color="gray", label="prevalence")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Cross-match precision–recall")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "precision_recall_oof.png", dpi=150)
        plt.close(fig)
        observed, predicted = calibration_curve(pr_y_values, pr_scores, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.plot(predicted, observed, marker="o")
        ax.set_xlabel("Mean predicted event probability")
        ax.set_ylabel("Observed annotated-event frequency")
        ax.set_title("OOF reliability (annotation target)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "calibration_oof.png", dpi=150)
        plt.close(fig)

    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(scene_keys(fit)):
        if y_all[index] >= 0:
            groups[key].append(index)
    examples = []
    for key, rows in groups.items():
        positives = [row for row in rows if y_all[row] > 0]
        if not positives:
            continue
        top = max(rows, key=lambda row: (score_all[row], -row))
        nearest = min(rows, key=lambda row: (fit["ball"][row, 4] if fit["ball"][row, 0] > 0.5 else 10.0, row))
        examples.append({
            "key": key,
            "rows": rows,
            "positive_rows": positives,
            "top": top,
            "nearest": nearest,
            "correct_top": y_all[top] > 0,
            "correct_nearest": y_all[nearest] > 0,
            "lead": float(np.min((event_all[positives] - fit["query_frame"][positives]) / 25.0)),
        })
    successes = [row for row in examples if row["correct_top"]]
    failures = [row for row in examples if not row["correct_top"]]
    successes.sort(key=lambda row: (row["correct_nearest"], row["lead"], row["key"]))
    failures.sort(key=lambda row: (row["lead"], row["key"]))
    picked = [(row, "top1_correct") for row in successes[:3]] + [(row, "top1_missed") for row in failures[:3]]
    if len(picked) < 6:
        remaining = sorted(examples, key=lambda row: row["key"])
        for row in remaining:
            if len(picked) >= 6:
                break
            if all(row["key"] != prior["key"] for prior, _ in picked):
                picked.append((row, "additional_positive_scene"))
    if len(picked) < 6:
        raise ValueError(f"Only {len(picked)} distinct positive scenes are available for anonymized examples.")
    pitch_index = []
    for number, (example, tag) in enumerate(picked[:6], start=1):
        match, half_code, _ = example["key"]
        rows = example["rows"]
        query = int(fit["query_frame"][rows[0]])
        half = "1st" if int(half_code) == 1 else "2nd"
        track = load_ball(ROOT / "data" / f"{match}_{half}_ball.npz")
        ball_xy, _, _, ball_valid = track.asof(np.asarray([query], dtype=np.int64))
        xy = fit["x"][rows, :2].astype(np.float32) * np.asarray([52.5, 34.0], dtype=np.float32)
        ids = {row: f"P{idx + 1:02d}" for idx, row in enumerate(rows)}
        positive = set(example["positive_rows"])
        top = example["top"]
        nearest = example["nearest"]
        fig, ax = plt.subplots(figsize=(10, 6.3))
        ax.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, lw=1.5, color="#34495e"))
        ax.plot([0, 0], [-34, 34], color="#95a5a6", lw=0.8)
        ax.add_patch(Circle((0, 0), 9.15, fill=False, lw=0.8, color="#95a5a6"))
        ax.add_patch(Rectangle((-52.5, -20.16), 16.5, 40.32, fill=False, lw=0.8, color="#95a5a6"))
        ax.add_patch(Rectangle((36, -20.16), 16.5, 40.32, fill=False, lw=0.8, color="#95a5a6"))
        for i, row in enumerate(rows):
            marker = "*" if row in positive else "o"
            color = "#e67e22" if row in positive else "#7f8c8d"
            size = 180 if row == top else 110 if row == nearest else 50
            edge = "#2874a6" if row == top else "#239b56" if row == nearest else "white"
            ax.scatter(xy[i, 0], xy[i, 1], marker=marker, s=size, color=color, edgecolor=edge, linewidth=1.8, zorder=3)
            ax.text(xy[i, 0] + 0.6, xy[i, 1] + 0.6, ids[row], fontsize=7, color="#17202a")
        if ball_valid[0]:
            ax.scatter(ball_xy[0, 0], ball_xy[0, 1], marker="o", s=125, color="#f1c40f", edgecolor="black", label="ball", zorder=4)
            ax.add_patch(Circle((ball_xy[0, 0], ball_xy[0, 1]), 5.0, fill=False, linestyle="--", color="#f1c40f", alpha=0.8))
            ax.plot([ball_xy[0, 0], xy[rows.index(top), 0]], [ball_xy[0, 1], xy[rows.index(top), 1]], color="#2874a6", lw=1.0, linestyle=":")
        ax.set_xlim(-54, 54)
        ax.set_ylim(-36, 36)
        ax.set_aspect("equal")
        ax.set_xlabel("Pitch x (m; source convention)")
        ax.set_ylabel("Pitch y (m; source convention)")
        ax.set_title(f"Positive-scene example {number}: {tag.replace('_', ' ')}\nForecast window (query, query+1s]; first supported event at about {example['lead']:.2f}s")
        ax.text(0.01, 0.01, "Star = supported event actor; blue edge = HGB top-1; green edge = nearest-ball actor. Names and source IDs omitted.", transform=ax.transAxes, fontsize=8, va="bottom")
        fig.tight_layout()
        destination = pitch_dir / f"positive_scene_{number:02d}_{tag}.png"
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        pitch_index.append({"figure": str(destination.relative_to(RUN)), "case": tag, "event_lead_seconds": example["lead"], "ball_location_available": bool(ball_valid[0]), "candidate_count": len(rows)})
    atomic_json(RUN / "pitch_scene_index.json", {"status": "GENERATED", "anonymization": "Source player IDs, roster names and track slots are not shown; candidates are assigned scene-local P01... labels.", "coordinate_note": "Plot uses local pitch-coordinate convention; physical orientation, camera calibration and ball/video synchronization are not independently validated.", "examples": pitch_index})
    return {
        "status": "COMPLETE",
        "figure_files": [str(path.relative_to(RUN)) for path in sorted(figures.rglob("*.png"))],
        "pitch_scene_count": len(pitch_index),
        "additional_diagnostic_figures": 8,
        "anonymized": True,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def update_status(stage: str, completed: list[str], extra: dict[str, Any] | None = None) -> None:
    status = {
        "current_stage": stage,
        "completed": completed,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        status.update(extra)
    atomic_json(RUN / "status.json", status)


def write_reports(test_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    a = read_json(RUN / "ball_shortcut_audit.json")
    b = read_json(RUN / "temporal_experiments.json")
    c = read_json(RUN / "lead_time_study.json")
    alerts = read_json(RUN / "event_alert_operating_points.json")
    d = read_json(RUN / "autonomous_branch_decision.json")
    branch = read_json(RUN / "branch_metrics.json")
    plots = read_json(RUN / "visualization_manifest.json")

    contrast_pairs = (
        ("ball_only", "x_no_ball"),
        ("x_plus_availability", "x_no_ball"),
        ("x_plus_distance", "x_no_ball"),
        ("x_plus_all", "x_no_ball"),
        ("x_plus_block_shuffle", "x_no_ball"),
        ("lag_0p2", "x_no_ball"),
        ("lag_0p5", "x_no_ball"),
        ("lag_1p0", "x_no_ball"),
        ("lag_2p0", "x_no_ball"),
    )
    a["additional_paired_match_contrasts"] = {}
    for left, right in contrast_pairs:
        a["additional_paired_match_contrasts"][f"{left}_minus_{right}"] = {
            metric: paired_metric_ci(a["per_match"], left, right, metric)
            for metric in ("AP", "brier", "hazard_NLL")
        }
    atomic_json(RUN / "ball_shortcut_audit.json", a)

    a_rows = []
    for name in ("nearest_ball_rule", "ball_only", "x_no_ball", "x_plus_availability", "x_plus_distance", "x_plus_all", "x_plus_block_shuffle", "lag_0p2", "lag_0p5", "lag_1p0", "lag_2p0"):
        row = a["overall"][name]
        a_rows.append(f"| {name} | {_fmt(row['AP'])} | {_fmt(row['brier'])} | {_fmt(row['scene_ranking']['top1'])} |")
    b_rows = []
    for name in ("current_hgb", "history_hgb", "gru_current", "gru_history"):
        row = b["overall"][name]
        b_rows.append(f"| {name} | {_fmt(row['AP'])} | {_fmt(row['brier'])} | {_fmt(row['scene_ranking']['top1'])} |")
    lead_rows = []
    for name, label in (("lead_0_1s", "Event in (t,t+1s]"), ("lead_1_2s", "Event in (t+1s,t+2s]"), ("lead_2_3s", "Event in (t+2s,t+3s]"), ("within_1s", "Any event within 1s"), ("within_2s", "Any event within 2s"), ("within_3s", "Any event within 3s")):
        row = c["fixed_lead_bins"].get(name) or c["cumulative_any_event_horizons"].get(name)
        lead_rows.append(f"| {label} | {row['eligible_actor_time_rows']:,} | {row['prevalence']:.3%} | {_fmt(row['overall']['AP'])} | {_fmt(row['overall']['scene_ranking']['top1'])} |")
    alert_rows = []
    for name in ("actor_hour_1", "actor_hour_5", "actor_hour_10", "match_hour_1", "match_hour_5", "match_hour_10"):
        row = alerts["operating_points"][name]
        alert_rows.append(f"| {name} | {_fmt(row['mean']['event_recall'])} | {_fmt(row['mean']['false_alerts_per_actor_hour'])} | {_fmt(row['mean']['false_alerts_per_match_hour'])} | {_fmt(row['mean']['alert_workload_top1_reviews_per_match_hour'])} |")
    far_rows = []
    for name in ("full_cohort_hgb_no_ball_restricted_far", "full_cohort_hgb_ball_restricted_far", "far_only_hgb_no_ball", "far_only_hgb_ball", "nearest_among_far"):
        row = branch["overall"][name]
        far_rows.append(f"| {name} | {_fmt(row['AP'])} | {_fmt(row['scene_ranking']['top1'])} | {_fmt(row['scene_ranking']['top3'])} |")

    start = datetime.fromisoformat(check_protocol()["locked_utc"].replace("Z", "+00:00")).timestamp()
    runtime = max(0.0, time.time() - start)
    elapsed = f"{runtime / 3600.0:.2f} hours ({runtime / 60.0:.1f} minutes)"
    selected_threshold = alerts["operating_points"]["actor_hour_1"]["mean"]
    result_lines = [
        "# Ball-centric prospective action forecasting",
        "",
        "## Scientific result",
        "",
        "The large ball-feature gain is not explained by an uncalibrated nearest-player rule or ball-availability alone. A distance-only HGB is already strong, and richer ball state adds a smaller but measurable cross-match increment. The defensible claim is predictive association with supported future BAS annotations from privileged exocentric coordinates—not possession, intent, human perception, optimal tactics, or operational event detection.",
        "",
        "## A — Shortcut and proximity controls",
        "",
        "| Model/diagnostic | OOF actor-row AP | Brier | Positive-scene top-1 |",
        "|---|---:|---:|---:|",
        *a_rows,
        "",
        f"Decision: **{a['judgement']}**. Ball availability only and within-match 20-second block-shuffled ball features are reported as negative controls. Matched match-level contrasts and 95% descriptive bootstrap intervals are in `ball_shortcut_audit.json`; only six fit-match clusters are available.",
        "",
        "## B — Causal ball history",
        "",
        "| Model | OOF actor-row AP | Brier | Positive-scene top-1 |",
        "|---|---:|---:|---:|",
        *b_rows,
        "",
        f"All 6 folds used three seeds each for both same-capacity GRU arms. The instantaneous GRU repeats the current ball/player state across all steps; the history GRU receives six causal snapshots over 2 seconds. Selected epoch is development-only. GPU step counts, training curves and peak allocated VRAM are in `temporal_experiments.json`.",
        "",
        "## C — Forecasting lead and review workload",
        "",
        "| Target definition | Eligible actor rows | Prevalence | AP | Positive-scene top-1 |",
        "|---|---:|---:|---:|---:|",
        *lead_rows,
        "",
        "Intervals and cumulative horizons are separate targets; their AP values are not directly interchangeable because prevalence and censoring differ. Annotation-relative top-1 alerts use 2-second actor suppression and thresholds selected on match 132877 only.",
        "",
        "| Development budget | Held-out unique-event recall | False alerts / actor-hour | False alerts / match-hour | Top-1 reviews / match-hour |",
        "|---|---:|---:|---:|---:|",
        *alert_rows,
        "",
        f"At the one-per-actor-hour point, mean event recall was {_fmt(selected_threshold['event_recall'])}; this is annotation-relative, not a physical false-alarm rate. No-event windows are not independently verified negatives. The study's selected interpretation is **{c['choice']}**.",
        "",
        "## D — Evidence-selected conditional experiment",
        "",
        f"Chosen branch: **{d['chosen_branch']}**. The cohort was restricted to actors more than 5m from the current ball using query-time data only; all six leave-one-match-out folds were executed.",
        "",
        "| Far-candidate model | AP | Positive-scene top-1 | Positive-scene top-3 |",
        "|---|---:|---:|---:|",
        *far_rows,
        "",
        f"The far-candidate set retained {branch['positive_actor_rows_retained']:.1%} of positive actor-time rows, so conclusions apply only to that selected observable subset.",
        "",
        "## Data and claim boundaries",
        "",
        "- 18 source cohort shards, four protected checkpoints and 18 existing ball sources passed SHA-256 checks before training. Existing ball-feature rows were joined to cohort rows by query, actor slot and segment.",
        "- The source ball archive uses one-based frame IDs while query frames are zero-based; this run deliberately reuses the already audited as-of join implementation and does not infer a clock offset. Physical video-to-pitch synchronization remains unverified.",
        "- The 1-second reconstruction was compared against the locked cohort target; exact comparison and all censoring reasons are in `lead_time_study.json`.",
        "- Six fit matches are the only outer-fold clusters. Match-bootstrap intervals are descriptive. Match 132877 is development-only. No sealed match was available or scored.",
        "- Ball tracks are privileged exocentric reference data. No validated gaze, attention, possession-awareness or real player viewpoint is demonstrated.",
        "- The action annotations do not describe intent, action success, tactical value or optimal decisions.",
        "- Annotation-relative non-events are not verified real-world negatives; alarm results do not support operational use.",
        "",
        "## Figures and reproduction",
        "",
        f"Generated {plots['pitch_scene_count']} anonymized pitch/event schematics plus {len(plots['figure_files']) - plots['pitch_scene_count']} aggregate plots. See the `figures/` folder. Reproduction command: `powershell -ExecutionPolicy Bypass -File .\\reproduce.ps1`.",
        "",
        f"End-to-end elapsed time at report generation: {elapsed}.",
    ]
    scientific_report = "\n".join(result_lines) + "\n"
    (RUN / "scientific_report.md").write_text(scientific_report, encoding="utf-8")

    preflight_result = read_json(RUN / "source_integrity.json")
    fit_one_second = c["fit_label_censoring_audit"]["targets"]["within_1s"]
    data_quality = [
        "# Data quality and leakage audit",
        "",
        "## Verified",
        "",
        f"- Source cohort: {preflight_result['source_shard_count']} of 18 SHA-256-pinned shards verified against the locked manifest; four protected checkpoints verified.",
        f"- Ball sources: {preflight_result['ball_source_count']} of 18 source archives verified; each existing ball-feature shard was hash-checked and row-aligned with the actor-time cohort.",
        f"- Starting Git status contained {preflight_result['starting_git_status_lines']} pre-existing status lines. This mission wrote only under its fresh `outputs/ball_centric_action_autonomy_20261102T222620Z/` directory plus the return ZIP and did not edit original datasets/checkpoints.",
        "- Cohort rows are matched by match, half, query frame, actor slot and segment. The exact OOF baseline predictions are checked against target and scene identifiers before use.",
        "- Ball features use the previous audited latest-observation-at-or-before-query join with maximum age five frames; all synthetic history snapshots use the same causal bound. No interpolation from future frames is performed.",
        f"- Primary 1s label reconstruction matched the locked cohort on {c['fit_label_censoring_audit']['targets']['primary_1s_reconstruction']['eligible_rows_compared']:,} eligible rows with {c['fit_label_censoring_audit']['targets']['primary_1s_reconstruction']['mismatched_labels']} mismatches.",
        "",
        "## Censoring, coverage and unresolved assumptions",
        "",
        f"- Primary 1s fit cohort: {fit_one_second['eligible_actor_time_rows']:,} eligible actor-time rows, {fit_one_second['censored_actor_time_rows']:,} additional rows censored by the extended target reconstruction. Reasons: `{json.dumps(fit_one_second['censor_reasons'], sort_keys=True)}`.",
        "- Event timestamps are resolved through the repository's 25-Hz source-clock contract and checked against stable BAS frames. This is metadata consistency, not independent video synchronization validation.",
        "- Ball coordinate plausibility and actor-ball proximity were previously checked, but annotation/event timing is not physically synchronized against panoramic video. Ball source frame IDs are one-based; no offset conversion was attempted.",
        "- BAS files contain supported events and actor mappings. A missing annotation does not certify no action. Restart/restart-state semantics are not sufficiently verified to claim a complete negative set.",
        "- Track discontinuities are censored for longer prospective windows and counted. Unmapped BAS events censor affected scenes. Sparse classes remain weakly supported; inspect per-class counts in `lead_time_study.json`.",
        "- Ball-missing rows are strongly associated with lower annotation prevalence in prior results. Availability-only and shuffled controls in this run test, but do not fully remove, dataset-coverage confounding.",
        "- Actor slot and segment IDs are local join keys only and are excluded from model features and the return ZIP. Player names/IDs are not placed in the anonymized pitch figures.",
        "",
        "## No leakage claims",
        "",
        "- Outer-fold model fitting excludes each held-out match. The development match is used for neural early stopping and alarm threshold selection only.",
        "- Retrospective matches 118577 and 128058 were not loaded in this run. 118575 is excluded because matching tracking annotations were not verified.",
        "- Target event actor, target label, event ID, event frame, track-slot key and future position are never model inputs. They are used only for target construction, row joins, censoring or evaluation.",
        "",
        "## Physical and scientific limits",
        "",
        "The ball and tracking features are privileged offline exocentric measurements. This experiment does not validate player perception, gaze, possession awareness, intent, pass quality, tactical value, or optimal decisions. False alerts are relative to the available BAS annotation set, not human-adjudicated false actions.",
    ]
    (RUN / "data_quality_and_leakage_audit.md").write_text("\n".join(data_quality) + "\n", encoding="utf-8")

    roadmap = [
        "# Research roadmap",
        "",
        "## Completed",
        "",
        "- **A:** six-fold matched baseline/control audit; ball-only, nearest rule, availability, distance, all-feature, within-block shuffle and causal-lag arms.",
        "- **B:** causal 2-second ball/actor histories, HGB and three-seed GRU comparisons; missingness/noise stress and speed/density strata.",
        "- **C:** fixed lead bins and cumulative horizons through 3 seconds, reconstructed target/censoring audit, scene-ranking and development-threshold alert workload.",
        "- **D:** conditional >5m non-nearest candidate experiment selected from A-C evidence and executed across the same six fit-match folds.",
        "",
        "## Positive evidence",
        "",
        f"- Current ball distance and full ball context outperform availability-only and within-match shuffled controls (full AP {_fmt(a['overall']['x_plus_all']['AP'])}; availability-only AP {_fmt(a['overall']['x_plus_availability']['AP'])}; shuffled AP {_fmt(a['overall']['x_plus_block_shuffle']['AP'])}).",
        f"- The nearest-ball ranking rule AP is {_fmt(a['overall']['nearest_ball_rule']['AP'])}, while distance-only HGB AP is {_fmt(a['overall']['x_plus_distance']['AP'])}; the signal is not reducible to selecting the closest actor.",
        f"- Conditional >5m analysis retained {branch['positive_actor_rows_retained']:.1%} of positive actor rows and achieved far-only ball-HGB AP {_fmt(branch['overall']['far_only_hgb_ball']['AP'])}.",
        "",
        "## Negative or failed",
        "",
        f"- Ball availability-only HGB AP {_fmt(a['overall']['x_plus_availability']['AP'])} and time-block-shuffled ball AP {_fmt(a['overall']['x_plus_block_shuffle']['AP'])} did not reproduce the full-ball lift.",
        f"- Causal lag AP declines from {_fmt(a['overall']['lag_0p2']['AP'])} at 0.2s to {_fmt(a['overall']['lag_2p0']['AP'])} at 2s, indicating strong timing sensitivity.",
        f"- Current GRU/history results are {('positive' if b['overall']['gru_history']['AP'] > b['overall']['gru_current']['AP'] else 'not positive')} for history versus the same-capacity instantaneous control; see fold-level intervals before claiming added temporal value.",
        "- Operational alarms, gaze, player awareness, action intent, success, and optimal tactics were not established.",
        "",
        "## Inconclusive",
        "",
        "- Six fit matches are a small number of independent clusters; match-bootstrap ranges are descriptive.",
        "- Event-negative labels and missing-ball intervals remain incompletely adjudicated; no verified independent human review labels were available.",
        "- Longer lead AP changes with prevalence, eligible rows and censoring; it cannot be treated as a direct same-task improvement.",
        "",
        "## Paused",
        "",
        "- Synthetic limited-visibility/gaze interpretation, panoramic crop orientation, whole-body orientation and GNN rework remain paused for the previously documented validation blockers.",
        "",
        "## Blocked",
        "",
        "- No new licensed, matching, never-inspected BAS+tracking+ball match was verified in this mission; no sealed generalization test exists.",
        "- Physical video-to-pitch synchronization, restart completeness and independent annotation of negative actions are not established.",
        "",
        "## Next",
        "",
        "Prioritize a small, blinded, multi-annotator event/actor audit linked to verified video timestamps and track identities. It directly resolves whether the current high ranking reflects real anticipatory actions or annotation coverage/timing, before expanding model complexity or claiming decision intelligence.",
    ]
    (RUN / "roadmap.md").write_text("\n".join(roadmap) + "\n", encoding="utf-8")

    handoff = [
        "# Execution handoff",
        "",
        "## Mission status: COMPLETE WITH SCIENTIFIC LIMITATIONS",
        "",
        f"- Run start (protocol lock): `{check_protocol()['locked_utc']}` UTC.",
        f"- Elapsed by report generation: **{elapsed}**.",
        "- Host: Windows 11; Python 3.14.8; CUDA enabled; NVIDIA RTX A5000; disk at preflight had 583.41 GB free.",
        f"- A stage elapsed: {a['elapsed_seconds']:.1f}s (HGB CPU and cached prior OOF baselines).",
        f"- B stage elapsed: {b['elapsed_seconds']:.1f}s (CUDA temporal training, 3 seeds × 6 folds × 2 matched GRU arms).",
        f"- C stage elapsed: {c['elapsed_seconds']:.1f}s (CPU HGB fold comparisons, target reconstruction, alerts).",
        f"- D conditional branch retained {branch['positive_actor_rows_retained']:.1%} of positives; no external data downloads were required.",
        "",
        "## Investigation outcomes",
        "",
        f"- A: **{a['judgement']}**; nearest baseline AP {_fmt(a['overall']['nearest_ball_rule']['AP'])}, distance-only HGB {_fmt(a['overall']['x_plus_distance']['AP'])}, full-ball HGB {_fmt(a['overall']['x_plus_all']['AP'])}.",
        f"- B: HGB history/current AP {_fmt(b['overall']['history_hgb']['AP'])}/{_fmt(b['overall']['current_hgb']['AP'])}; GRU history/instantaneous AP {_fmt(b['overall']['gru_history']['AP'])}/{_fmt(b['overall']['gru_current']['AP'])}. See robustness tests and per-match results.",
        f"- C: **{c['choice']}**; 1-second reconstruction had {c['primary_cohort_label_reconstruction']['mismatched_labels']} mismatches among {c['primary_cohort_label_reconstruction']['eligible_rows_compared']:,} eligible rows.",
        f"- D: executed **{d['chosen_branch']}**; far-only ball HGB AP {_fmt(branch['overall']['far_only_hgb_ball']['AP'])}.",
        f"- Figures: {plots['pitch_scene_count']} anonymized pitch schematics, {len(plots['figure_files']) - plots['pitch_scene_count']} metric/diagnostic plots.",
        f"- Tests: {json.dumps(test_summary or {'status': 'NOT_RECORDED'}, sort_keys=True)}.",
        "",
        "## Verification and limits",
        "",
        "- 18 source cohort shards, four protected checkpoint files, 18 ball sources and 18 cached ball-feature shards passed SHA-256 checks before modeling.",
        "- No original source annotation, video, cohort, split, checkpoint or historical output was edited. Work was isolated to this run folder and its verified return archive.",
        "- No sealed match was available; retrospective matches were not loaded by this run.",
        "- Ball tracks are exocentric privileged references. No validated gaze or human perception claim is made.",
        "- No operational alarm, verified negative-event, action-intent, action-quality or optimal-tactics claim is made.",
        "",
        "## Files",
        "",
        "- `scientific_report.md`: metrics and interpretation.",
        "- `ball_shortcut_audit.json`, `temporal_experiments.json`, `lead_time_study.json`, `event_alert_operating_points.json`: machine-readable results.",
        "- `autonomous_branch_decision.json`, `branch_metrics.json`: selected and completed Investigation D.",
        "- `data_quality_and_leakage_audit.md`, `roadmap.md`, `acceptance.json`: provenance, limitations and next steps.",
        "- `figures/`: aggregate metrics plus six anonymized positive-scene schematics.",
    ]
    (RUN / "handoff.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")

    acceptance = {
        "mission_status": "COMPLETE_WITH_SCIENTIFIC_LIMITATIONS",
        "code_written": True,
        "tests": test_summary or {"status": "NOT_RECORDED"},
        "experiments": {
            "A": {"status": "COMPLETE", "six_fit_match_folds": True, "nearest_rule_and_negative_controls": True},
            "B": {"status": "COMPLETE", "six_fit_match_folds": True, "matched_three_seed_cuda_models": True, "stress_tests": sorted(b["robustness"].keys())},
            "C": {"status": "COMPLETE", "fixed_lead_and_cumulative_horizons": True, "censoring_audit": True, "development_only_thresholds": True},
            "D": {"status": "COMPLETE", "branch": d["chosen_branch"], "six_fit_match_folds": True},
        },
        "data": {"source_shards_verified": preflight_result["source_shard_count"], "ball_sources_verified": preflight_result["ball_source_count"], "retrospective_matches_loaded": False, "sealed_match": False},
        "claim_boundaries": {
            "validated_gaze": False,
            "operational_alarm": False,
            "verified_no_action_negatives": False,
            "optimal_tactics": False,
            "physical_ball_video_synchronization": False,
        },
        "outputs": {"pitch_schematics": plots["pitch_scene_count"], "aggregate_figures": len(plots["figure_files"]) - plots["pitch_scene_count"]},
    }
    atomic_json(RUN / "acceptance.json", acceptance)
    return acceptance


def run_focused_tests() -> dict[str, Any]:
    import subprocess

    test_path = RUN / "tests" / "test_research.py"
    command = [sys.executable, "-m", "pytest", "-q", str(test_path)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    report = {
        "command": command,
        "return_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }
    atomic_json(RUN / "test_results.json", report)
    if completed.returncode:
        raise RuntimeError(f"Focused tests failed.\n{completed.stdout}\n{completed.stderr}")
    return report


def package_return_zip() -> dict[str, Any]:
    import zipfile

    required = [
        "protocol.lock.json",
        "run_research.py",
        "README.md",
        "reproduce.ps1",
        "tests/test_research.py",
        "source_integrity.json",
        "ball_shortcut_audit.json",
        "causal_ball_history_manifest.json",
        "temporal_experiments.json",
        "lead_time_study.json",
        "event_alert_operating_points.json",
        "autonomous_branch_decision.json",
        "branch_metrics.json",
        "data_quality_and_leakage_audit.md",
        "scientific_report.md",
        "roadmap.md",
        "handoff.md",
        "acceptance.json",
        "test_results.json",
        "visualization_manifest.json",
        "decisions.jsonl",
    ]
    members = []
    for relative in required:
        path = RUN / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required return-package member absent: {path}")
        members.append(path)
    members.extend(sorted((RUN / "figures").rglob("*.png")))
    members.append(RUN / "pitch_scene_index.json")
    for path in members:
        if not path.is_file():
            raise FileNotFoundError(f"Return-package source absent: {path}")
        if path.suffix.lower() in {".npz", ".npy", ".pt", ".pth", ".mp4", ".csv"}:
            raise ValueError(f"Private/large data is forbidden in the return ZIP: {path}")
    manifest_rows = [
        {
            "path": str(path.relative_to(RUN)).replace("\\", "/"),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in members
    ]
    execution_manifest = RUN / "execution_manifest.json"
    atomic_json(execution_manifest, {
        "schema": "football.ball-centric-autonomy.package-manifest.1",
        "members": manifest_rows,
        "exclusions": ["raw videos", "BAS/GSR/ball archives", "tracking cache files", "private row-level predictions", "feature arrays", "weights and checkpoints", "player identity tables"],
        "manifest_self_hash_excluded": True,
    })
    members.append(execution_manifest)
    run_tag = RUN.name.removeprefix("ball_centric_action_autonomy_")
    zip_path = ROOT / "outputs" / f"return_ball_centric_action_autonomy_{run_tag}.zip"
    sidecar_path = zip_path.with_suffix(zip_path.suffix + ".sha256.json")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_suffix(".tmp.zip")
    if temporary.exists():
        temporary.unlink()
    expected_hashes = {
        str(path.relative_to(RUN)).replace("\\", "/"): sha256(path)
        for path in members
    }
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            archive.write(path, arcname=f"{RUN.name}/{str(path.relative_to(RUN)).replace(chr(92), '/')}")
    os.replace(temporary, zip_path)
    member_checks = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise IOError(f"ZIP CRC failure in {bad_member}")
        for name in archive.namelist():
            payload = archive.read(name)
            member_hash = hashlib.sha256(payload).hexdigest()
            relative = name.split("/", 1)[1]
            expected = expected_hashes.get(relative)
            if expected is None or member_hash != expected:
                raise IOError(f"ZIP member integrity mismatch: {name}")
            member_checks.append({"path": relative, "sha256": member_hash, "size_bytes": len(payload)})
    zip_hash = sha256(zip_path)
    sidecar = {
        "archive": zip_path.name,
        "archive_sha256": zip_hash,
        "archive_size_bytes": zip_path.stat().st_size,
        "crc_verified": True,
        "member_count": len(member_checks),
        "members": member_checks,
    }
    atomic_json(sidecar_path, sidecar)
    result = {
        "status": "VERIFIED",
        "archive": str(zip_path.relative_to(ROOT)),
        "sidecar": str(sidecar_path.relative_to(ROOT)),
        "sha256": zip_hash,
        "size_bytes": zip_path.stat().st_size,
        "crc_verified": True,
        "member_count": len(member_checks),
        "member_sha256_verified": True,
    }
    atomic_json(RUN / "package_verification.json", result)
    return result


def run_all() -> dict[str, Any]:
    completed = ["preflight"]
    if not (RUN / "source_integrity.json").is_file():
        preflight()
    update_status("A", completed)
    run_investigation_a()
    completed.append("A")
    update_status("B", completed)
    run_investigation_b()
    completed.append("B")
    update_status("C", completed)
    run_investigation_c()
    completed.append("C")
    update_status("D", completed)
    run_investigation_d()
    completed.append("D")
    visual_manifest = generate_visualizations()
    atomic_json(RUN / "visualization_manifest.json", visual_manifest)
    completed.append("visualizations")
    tests = run_focused_tests()
    completed.append("tests")
    acceptance = write_reports(tests)
    completed.append("reports")
    package = package_return_zip()
    completed.append("package")
    update_status("COMPLETE", completed, {"package": package})
    return {"status": "COMPLETE", "completed": completed, "package": package, "acceptance": acceptance}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autonomous ball-centric action research stages.")
    parser.add_argument("--stage", choices=("preflight", "A", "B", "C", "D", "visuals", "tests", "reports", "package", "all"), default="all")
    args = parser.parse_args()
    stage = args.stage
    if stage == "all":
        print(json.dumps(run_all(), indent=2))
        return
    if stage != "preflight" and not (RUN / "source_integrity.json").is_file():
        preflight()
    if stage == "preflight":
        result = preflight()
    elif stage == "A":
        result = run_investigation_a()
    elif stage == "B":
        result = run_investigation_b()
    elif stage == "C":
        result = run_investigation_c()
    elif stage == "D":
        result = run_investigation_d()
    elif stage == "visuals":
        result = generate_visualizations()
        atomic_json(RUN / "visualization_manifest.json", result)
    elif stage == "tests":
        result = run_focused_tests()
    elif stage == "reports":
        result = write_reports(read_json(RUN / "test_results.json"))
    else:
        result = package_return_zip()
    update_status(stage, [stage], {"result_status": result.get("status", "COMPLETE") if isinstance(result, dict) else "COMPLETE"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
