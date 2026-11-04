from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, precision_recall_curve

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OLD = ROOT / "outputs" / "ball_centric_action_autonomy_20261102T222620Z"
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location("previous_ball_research", OLD / "run_research.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
FIT = old.FIT_MATCHES
SEEDS = (42, 137, 901)
PARAMS = dict(old.HGB_PARAMS)
os.environ.setdefault("OMP_NUM_THREADS", "8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def decision(stage, text, evidence):
    with (RUN / "decisions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc(), "stage": stage, "decision": text, "evidence": evidence, "boundary": "Fit OOF evidence; 132877 development-only; no retrospective selection."}) + "\n")


def status(stage, state, extra=None):
    path = RUN / "status.json"
    values = read(path) if path.exists() else {}
    values[str(stage)] = {"state": state, "utc": utc(), **(extra or {})}
    write(path, values)
    with (RUN / "execution_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": str(stage), **values[str(stage)]}) + "\n")
    lines = ["# Continuous research mission status", "", "| Stage | State | Updated UTC |", "|---|---|---|"]
    lines += [f"| {name} | {value['state']} | {value['utc']} |" for name, value in values.items()]
    (RUN / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def groups(data):
    keys = np.rec.fromarrays([data["match"].astype(str), data["half_code"], data["query_frame"]], names="match,half,frame")
    _, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, len(order)]
    return [order[a:b] for a, b in zip(starts[:-1], starts[1:])]


def scene_rank(y, score, scene_groups):
    ranks = []
    for rows in scene_groups:
        positives = y[rows] > 0
        if not positives.any():
            continue
        order = np.argsort(-score[rows], kind="stable")
        rank = int(np.flatnonzero(positives[order])[0]) + 1
        ranks.append(rank)
    r = np.asarray(ranks)
    return {"positive_scenes": len(ranks), "top1": float(np.mean(r <= 1)) if len(r) else None, "top3": float(np.mean(r <= 3)) if len(r) else None, "mrr": float(np.mean(1 / r)) if len(r) else None}


def metrics(y, score, scene_groups=None, probability=None):
    binary = np.asarray(y) > 0
    score = np.asarray(score, dtype=np.float64)
    if not len(binary) or score.shape != binary.shape or not np.isfinite(score).all():
        raise ValueError("Nonempty labels and finite, aligned raw ranking scores are required.")
    prevalence = float(binary.mean())
    result = {
        "rows": len(binary), "positives": int(binary.sum()), "prevalence": prevalence,
        "AP": float(average_precision_score(binary, score)) if binary.any() else None,
        "AUC": float(roc_auc_score(binary, score)) if binary.any() and not binary.all() else None,
        "brier": None, "NLL": None,
    }
    result["AP_over_prevalence"] = result["AP"] / prevalence if prevalence else None
    if binary.any():
        precision, recall, _ = precision_recall_curve(binary, score)
        result["recall_at_annotation_precision"] = {str(level): float(recall[precision >= level].max(initial=0)) for level in (.5, .8)}
    else:
        result["recall_at_annotation_precision"] = {"0.5": None, "0.8": None}
    if probability is not None:
        p = np.asarray(probability, dtype=np.float64)
        if p.shape != binary.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Brier/NLL require valid probabilities separately from ranking scores.")
        result["brier"] = float(np.mean((p - binary) ** 2))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        result["NLL"] = float(-np.mean(binary * np.log(p) + ~binary * np.log1p(-p)))
    if scene_groups is not None:
        result["scene"] = scene_rank(np.asarray(y), score, scene_groups)
    return result


def subset(data, mask):
    return {k: v[mask] for k, v in data.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}


def summary(data, predictions, probabilistic=()):
    y = data["target"]
    g = groups(data)
    overall = {name: metrics(y, p, g, p if name in probabilistic else None) for name, p in predictions.items()}
    per_match = {}
    for match in FIT:
        mask = data["match"] == match
        if not mask.any():
            continue
        sg = groups(subset(data, mask))
        per_match[match] = {name: metrics(y[mask], p[mask], sg, p[mask] if name in probabilistic else None) for name, p in predictions.items()}
    macro = {name: {metric: float(np.mean([m[name][metric] for m in per_match.values() if m[name][metric] is not None])) if any(m[name][metric] is not None for m in per_match.values()) else None for metric in ("AP", "brier", "NLL")} for name in predictions}
    macro_scene = {name: {metric: float(np.mean([m[name]["scene"][metric] for m in per_match.values() if m[name]["scene"][metric] is not None])) if any(m[name]["scene"][metric] is not None for m in per_match.values()) else None for metric in ("top1", "top3", "mrr")} for name in predictions}
    return {"pooled": overall, "macro_match": macro, "macro_scene": macro_scene, "per_match": per_match}


def contrast(result, left, right, metric="AP"):
    delta = np.asarray([m[left][metric] - m[right][metric] for m in result["per_match"].values()])
    rng = np.random.default_rng(20261103)
    draws = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {"mean_delta": float(delta.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(), "per_match_delta": delta.tolist(), "clusters": len(delta), "interpretation": "Descriptive match-bootstrap, not population significance."}


def preflight():
    import torch
    from threadpoolctl import threadpool_limits

    threadpool_limits(8)
    provenance = read(OLD / "source_integrity.json")
    paths = {}
    for key in ("cohort_shards_verified", "protected_checkpoints_verified", "ball_sources_verified", "existing_ball_feature_shards_verified"):
        for row in provenance[key]:
            path = ROOT / row["path"]
            got = digest(path)
            if got != row["sha256"]:
                raise ValueError(f"Protected input changed: {path}")
            paths[str(path.relative_to(ROOT))] = got
    for path in OLD.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            paths[str(path.relative_to(ROOT))] = digest(path)
    original = ROOT / "outputs" / "action_opportunity_research_20261102_144500"
    for shard in read(original / "cohort_manifest.json")["shards"]:
        for path, key in (
            (ROOT / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{shard['match']}_{shard['half']}.npz", "source_cache_sha256"),
            (ROOT / "outputs" / "trajectory_audit_20261016" / f"associations_{shard['match']}_{shard['half']}.json", "association_sha256"),
        ):
            got = digest(path)
            if got != shard[key]:
                raise ValueError(f"Source tracking identity fingerprint failure: {path}")
            paths[str(path.relative_to(ROOT))] = got
        path = ROOT / "data" / f"{shard['match']}_12_class_events.json"
        paths[str(path.relative_to(ROOT))] = digest(path)
    git = subprocess.run(["git", "--no-pager", "status", "--porcelain=v1"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    (RUN / "starting_git_status.txt").write_text(git, encoding="utf-8")
    write(RUN / "input_hashes_before.json", paths)
    free = __import__("shutil").disk_usage(ROOT).free
    memory = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], capture_output=True, text=True, check=True).stdout.strip()
    write(RUN / "environment.json", {"utc": utc(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "platform": platform.platform(), "disk_free_bytes": free, "free_physical_memory_kb": memory, "input_files_hashed": len(paths)})
    decision("preflight", "Proceed with immutable cached cohorts, ball histories and outer folds; original files remain untouched.", {"input_files": len(paths), "free_disk_GB": free / 1e9})
    return {"status": "COMPLETE", "input_files": len(paths)}


def data(split="fit"):
    result = old.load_split(split)
    return result


def sequence(split="fit"):
    manifest = read(old.PRIOR / "cohort_manifest.json")
    pieces = []
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        key = f"{shard['match']}_{1 if shard['half'] == '1st' else 2}"
        path = OLD / "ball_history_shards_local" / f"{key}.npz"
        meta = read(path.with_suffix(".json"))
        if digest(path) != meta["output_sha256"]:
            raise ValueError("History cache fingerprint failure.")
        with np.load(path, allow_pickle=False) as z:
            pieces.append(z["sequence"])
    return np.concatenate(pieces).astype(np.float32)


def cached_temporal(d):
    result = {k: np.zeros(len(d["target"])) for k in ("current_hgb", "history_hgb", "gru_current", "gru_history")}
    for match in FIT:
        mask = d["match"] == match
        with np.load(OLD / "predictions_local" / f"temporal_{match}.npz", allow_pickle=False) as z:
            for name in result:
                if len(z[name]) != mask.sum():
                    raise ValueError("Temporal prediction denominator mismatch.")
                result[name][mask] = z[name]
    return result


def task1():
    d = data()
    cache, _ = old.load_cached_prediction_scores(d)
    learned = cached_temporal(d)
    distance = d["ball"][:, 4].astype(np.float64) * 105
    available = d["ball"][:, 0] > .5
    raw = np.where(available, -distance, -10000)
    predictions = {
        "nearest_ball": raw,
        "inverse_distance": np.where(available, 1 / (1 + distance), 0),
        "within_2m": (available & (distance <= 2)).astype(float),
        "within_5m": (available & (distance <= 5)).astype(float),
        "no_ball_hgb": cache["x_no_ball"],
        "current_hgb": learned["current_hgb"],
        "history_hgb": learned["history_hgb"],
    }
    predictions["ball_only_hgb"], ball_meta = fit_oof("control_ball_only", d["ball"][:, [0, 1, 2, 3, 4, 6, 7]], d)
    predictions["distance_only_hgb"], distance_meta = fit_oof("control_distance_only", d["ball"][:, [0, 4]], d)
    probability_names = ("no_ball_hgb", "current_hgb", "history_hgb", "ball_only_hgb", "distance_only_hgb")
    result = summary(d, predictions, probability_names)
    result["additional_control_training"] = {"ball_only_hgb": ball_meta, "distance_only_hgb": distance_meta}
    result["control_semantics"] = {"ball_only_hgb": "Seven cached relative-ball channels [availability,age,dx,dy,distance,speed,closing], no 196-dimensional context.", "distance_only_hgb": "Availability plus source distance only, unlike the old mislabeled context-plus-distance variant."}
    result["nearest_vs_current_AP"] = contrast(result, "nearest_ball", "current_hgb")
    result["history_vs_nearest_AP"] = contrast(result, "history_hgb", "nearest_ball")
    result["subgroups"] = {}
    for metres in (5, 10):
        mask = available & (distance > metres)
        result["subgroups"][f"beyond_{metres}m"] = summary(subset(d, mask), {k: v[mask] for k, v in predictions.items()}, probability_names)
    corrected = result["pooled"]["nearest_ball"]
    old_result = read(OLD / "ball_shortcut_audit.json")["overall"]["nearest_ball_rule"]
    result.update({
        "status": "COMPLETE",
        "bug": "Negative distances were clipped before AP and scene ranking, producing tied scores. Fixed in this run only: raw scores drive ranking; probabilities are separate.",
        "before": old_result,
        "raw_vs_monotonic_AP_difference": corrected["AP"] - result["pooled"]["inverse_distance"]["AP"],
        "distance_units": "metres, source normalized distance times 105; missing ball gets -10000m ranking score below any valid pitch distance",
        "team_variant": "NOT_RUN: causal ball-team possession is not source-verified; no future action used to infer team.",
        "old_conclusion": "REQUIRES_QUALIFICATION: predictive residual must be judged against corrected proximity, not the invalid near-chance control; tactical value is not labeled.",
    })
    write(RUN / "nearest_ball_correction.json", result)
    lines = [
        "# Nearest-ball correction: before and after", "",
        f"Old nearest AP {old_result['AP']:.5f}; corrected pooled AP {corrected['AP']:.5f}.",
        f"Old nearest scene top1 {old_result['scene_ranking']['top1']:.3%}; corrected {corrected['scene']['top1']:.3%}.",
        f"Current HGB AP {result['pooled']['current_hgb']['AP']:.5f}; history HGB {result['pooled']['history_hgb']['AP']:.5f}.",
        "The old claim that the nearest rule was near chance is invalid and retracted. Historical artifacts are preserved.",
        "Whether a learned model exceeds proximity is assessed from these corrected six-fold comparisons. A performance residual is not proof of tactics, intent or optimal decisions.",
        "No Brier/NLL is reported for raw proximity. Learned probability predictions are unchanged.",
    ]
    (RUN / "before_after.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision(1, "Replace invalid proximity inference with corrected raw-score evidence.", {"nearest_AP": corrected["AP"], "nearest_top1": corrected["scene"]["top1"], "history_minus_nearest": result["history_vs_nearest_AP"]})
    return result


def fit_oof(tag, x, d, y=None, hold_mask=None):
    from threadpoolctl import threadpool_limits
    threadpool_limits(8)
    y = d["target"] if y is None else y
    valid = y >= 0
    output = np.full(len(y), np.nan, dtype=np.float32)
    metadata = []
    for held in FIT:
        path = RUN / "predictions_local" / f"{tag}_{held}.npz"
        train = (d["match"] != held) & valid
        test = (d["match"] == held) & valid
        if hold_mask is not None:
            train &= hold_mask
            test &= hold_mask
        config = hashlib.sha256((tag + held + json.dumps(PARAMS, sort_keys=True)).encode() + np.asarray(y).tobytes() + np.asarray(x).tobytes()).hexdigest()
        meta_path = path.with_suffix(".json")
        if path.exists() and meta_path.exists() and read(meta_path)["fingerprint"] == config and read(meta_path)["sha256"] == digest(path):
            with np.load(path, allow_pickle=False) as z:
                output[test] = z["p"]
            metadata.append(read(meta_path))
            continue
        if not train.any() or not test.any() or np.unique(y[train] > 0).size != 2:
            raise ValueError(f"Insufficient support in {tag}:{held}")
        start = time.perf_counter()
        model = HistGradientBoostingClassifier(**PARAMS).fit(x[train], y[train] > 0)
        p = model.predict_proba(x[test])[:, 1]
        save(path, p=p.astype(np.float32), row=np.flatnonzero(test))
        meta = {"fingerprint": config, "sha256": digest(path), "held_out_match": held, "training_matches": [m for m in FIT if m != held], "train_rows": int(train.sum()), "test_rows": int(test.sum()), "seconds": time.perf_counter() - start}
        write(meta_path, meta)
        metadata.append(meta)
        output[test] = p
        print(f"{tag}: {held} complete ({meta['seconds']:.1f}s)", flush=True)
    return output, metadata


def aligned_state(raw, track, query, slots, segments, lag=0, noise=None):
    if lag < 0:
        raise ValueError("Future shifts are not legal causal inputs.")
    q = query.astype(np.int64) - int(lag)
    safe = np.maximum(q, 0)
    bxy, bv, age, bvld = track.asof(q)
    valid = (q >= 0) & raw["observed"][safe, slots] & (raw["segments"][safe, slots] == segments) & bvld
    axy = raw["xy"][safe, slots].astype(np.float32)
    av = np.zeros_like(axy)
    past = np.maximum(safe - 5, 0)
    vgood = (q >= 5) & raw["observed"][past, slots] & (raw["segments"][past, slots] == segments)
    av[vgood] = (axy[vgood] - raw["xy"][past[vgood], slots[vgood]]) / .2
    if noise is not None:
        bxy = bxy + noise
    delta = bxy - axy
    dist = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(dist[:, None], 1e-6)
    state = np.column_stack((
        valid, np.where(valid, age / .2, 0), delta[:, 0] / 105, delta[:, 1] / 68,
        dist / 105, np.sum(av * unit, axis=1) / 12,
        np.linalg.norm(bv, axis=1) / 40,
        np.sum((av - bv) * unit, axis=1) / 40,
        axy[:, 0] / 52.5, axy[:, 1] / 34, av[:, 0] / 12, av[:, 1] / 12,
    )).astype(np.float32)
    state[~valid] = 0
    state = np.clip(state, -20, 20)
    return state


def load_aligned(d, lag: int = 0, noise_metres: float = 0.0, noise_bound_metres: float = 0.0):
    if lag < 0:
        raise ValueError("Future shifts are not legal causal inputs.")
    tag = f"aligned_lag{lag}_noise{noise_metres:g}"
    if noise_bound_metres:
        if noise_bound_metres < 0:
            raise ValueError("Noise radius must be positive.")
        tag += f"_bounded{noise_bound_metres}"
    path = RUN / "features_local" / f"{tag}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return z["state"]
    out = np.zeros((len(d["target"]), 12), dtype=np.float32)
    for match in np.unique(d["match"]):
        for half in (1, 2):
            mask = (d["match"] == match) & (d["half_code"] == half)
            if not mask.any():
                continue
            half_name = "1st" if half == 1 else "2nd"
            cp = ROOT / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{match}_{half_name}.npz"
            with np.load(cp, allow_pickle=False) as z:
                raw = {k: z[k] for k in ("observed", "xy", "segments")}
            track = old.load_ball(ROOT / "data" / f"{match}_{half_name}_ball.npz")
            q = d["query_frame"][mask].astype(int)
            slots = d["actor_slot_local_only"][mask].astype(int)
            seg = d["segment_local_only"][mask].astype(int)
            if not raw["observed"][q, slots].all() or not np.array_equal(raw["segments"][q, slots], seg):
                raise ValueError(f"Actor-to-source join invalid: {match}:{half}")
            error = np.linalg.norm(raw["xy"][q, slots] - d["x"][mask, :2] * [52.5, 34], axis=1)
            if error.max() > .2:
                raise ValueError("Current actor position join differs by >0.2m.")
            noise = None
            if noise_metres:
                uq, inv = np.unique(q, return_inverse=True)
                noise = np.random.default_rng(int(match) + half).normal(0, noise_metres, (len(uq), 2))[inv]
                if noise_bound_metres:
                    length = np.linalg.norm(noise, axis=1, keepdims=True)
                    noise *= np.minimum(1, noise_bound_metres / np.maximum(length, 1e-12))
            out[mask] = aligned_state(raw, track, q, slots, seg, lag, noise)
    save(path, state=out)
    return out


def task2():
    d = data()
    predictions = {}
    metadata = {}
    for lag in (0, 5, 12, 25, 50):
        state = load_aligned(d, lag)
        name = f"joint_lag_{lag}frames"
        predictions[name], metadata[name] = fit_oof("timing_" + name, np.column_stack((d["x"], state)), d)
    for noise in (.5, 2):
        name = f"noise_{noise}m"
        state = load_aligned(d, 0, noise)
        predictions[name], metadata[name] = fit_oof("timing_" + name, np.column_stack((d["x"], state)), d)
    for radius in (.5, 2):
        name = f"bounded_noise_{radius}m"
        state = load_aligned(d, 0, float(radius), float(radius))
        predictions[name], metadata[name] = fit_oof("timing_" + name, np.column_stack((d["x"], state)), d)
    for name, x in (
        ("ball_withheld", d["x"]),
        ("missingness_only", np.column_stack((d["x"], d["ball"][:, :2]))),
        ("blocked_shuffle", np.column_stack((d["x"], old.block_shuffle_ball(d)))),
    ):
        predictions[name], metadata[name] = fit_oof("timing_" + name, x, d)
    result = summary(d, predictions, tuple(predictions))
    result["contrasts"] = {name: contrast(result, name, "joint_lag_0frames") for name in predictions if name != "joint_lag_0frames"}
    state = load_aligned(d)
    masks = {
        "ball_absent": state[:, 0] < .5,
        "fresh": (state[:, 0] > .5) & (state[:, 1] == 0),
        "stale": (state[:, 0] > .5) & (state[:, 1] > 0),
        "slow_le_1mps": (state[:, 0] > .5) & (state[:, 6] * 40 <= 1),
        "fast_gt_15mps": (state[:, 0] > .5) & (state[:, 6] * 40 > 15),
        "beyond_5m": (state[:, 0] > .5) & (state[:, 4] * 105 > 5),
    }
    result["strata"] = {k: {name: metrics(d["target"][mask], p[mask], probability=p[mask]) for name, p in predictions.items()} for k, mask in masks.items() if mask.any()}
    result.update({
        "status": "COMPLETE", "training": metadata,
        "clock_contract": "Ball archive integer frame IDs start at 1; source player queries start at 0. We preserve released frame labels and require raw frame ID <= query. A possible physical one-frame convention offset is unresolved; metadata cannot prove synchronization.",
        "lag_frames": [0, 5, 12, 25, 50],
        "lag_seconds": [0, .2, .48, 1, 2],
        "rounding": "0.5s is implemented as 12 frames = 0.48s (nearest causal source grid).",
        "causal_scope": "All historical ball and actor components are aligned to the same requested past timestamp. Other 196 context channels remain causal at current query; these are stale-ball-actor information tests, not whole-scene time travel.",
        "noise_scope": "Same coordinate perturbation for all actors in a scene; noise_ arms use Gaussian per-axis standard deviation in metres. bounded_noise_ arms clip that vector's norm to the named radius. Separately trained folds, not claimed unseen-noise robustness.",
    })
    save(RUN / "timing_predictions_local.npz", **predictions)
    write(RUN / "timing_and_leakage_tests.json", result)
    decision(2, "Use jointly aligned causal history; distinguish stale pair information from whole-scene staleness.", {"macro_AP": result["macro_match"], "physical_sync": "UNVERIFIED"})
    return result


def dynamics(d, seq):
    valid = seq[:, :, 0] > .5
    dist = seq[:, :, 4] * 105
    rel = seq[:, :, 2:4] * np.array([105, 68])
    dv = (rel[:, -1] - rel[:, -2]) / .4
    pair = valid[:, -1] & valid[:, -2]
    speed2 = np.sum(dv ** 2, axis=1)
    defined = pair & (speed2 > .04)
    tca = np.zeros(len(dist))
    tca[defined] = np.clip(-np.sum(rel[defined, -1] * dv[defined], axis=1) / speed2[defined], 0, 3)
    cpa = np.linalg.norm(rel[:, -1] + dv * tca[:, None], axis=1)
    first_last = valid[:, 0] & valid[:, -1]
    previous_current = valid[:, -2] & valid[:, -1]
    kinematics = np.column_stack((
        valid.mean(axis=1),
        np.where(first_last, (dist[:, -1] - dist[:, 0]) / 2 / 12, 0),
        np.where(previous_current, (dist[:, -1] - dist[:, -2]) / .4 / 12, 0),
        np.where(first_last, (rel[:, -1, 0] - rel[:, 0, 0]) / 105, 0),
        np.where(first_last, (rel[:, -1, 1] - rel[:, 0, 1]) / 68, 0),
        np.where(previous_current, (seq[:, -1, 6] - seq[:, -2, 6]) / .4, 0),
        np.where(defined, tca / 3, 0), defined.astype(float),
        np.where(defined, cpa / 105, 0),
        np.where(valid, dist, np.inf).min(axis=1) / 105,
        ((dist < 2) & valid).mean(axis=1),
        ((dist < 5) & valid).mean(axis=1),
        np.where(pair[:, None], dv / 40, 0),
    ))
    np.nan_to_num(kinematics, copy=False, nan=0, posinf=0, neginf=0)
    geometry = np.zeros((len(dist), 9), dtype=np.float32)
    for rows in groups(d):
        current = valid[rows, -1]
        dd = np.where(current, dist[rows, -1], 10000)
        order = np.argsort(dd, kind="stable")
        n = len(rows)
        nearest = dd[order[0]]
        second = dd[order[1]] if n > 1 else nearest
        logits = np.where(current, -dd / 2, -10000)
        weights = np.exp(logits - logits.max()) * current
        weights /= max(weights.sum(), 1e-12)
        rank = np.empty(n)
        rank[order] = np.arange(n)
        geometry[rows] = np.column_stack((
            np.full(n, nearest / 105), np.full(n, (second - nearest) / 105),
            rank / max(n - 1, 1), weights,
            np.full(n, np.sum(dd < 2) / 22), np.full(n, np.sum(dd < 5) / 22),
            (dd - nearest) / 105,
            np.full(n, -np.sum(weights * np.log(np.maximum(weights, 1e-12))) / np.log(max(n, 2))),
            np.full(n, np.sum(current) / max(n, 1)),
        ))
        geometry[rows[~current]] = 0
    return np.clip(kinematics, -20, 20).astype(np.float32), np.clip(geometry, -20, 20)


def feature_sets(d, seq):
    kin, geo = dynamics(d, seq)
    current = np.column_stack((d["x"], d["ball"])).astype(np.float32)
    return {
        "current": current,
        "raw_history": np.column_stack((current, seq[:, :-1].reshape(len(seq), -1))),
        "kinematics": np.column_stack((current, kin)),
        "competition": np.column_stack((current, geo)),
        "interpretable": np.column_stack((current, kin, geo)),
        "ball_light": np.column_stack((d["x"][:, :16], geo)),
    }


def task3():
    d = data()
    seq = sequence()
    fs = feature_sets(d, seq)
    preds = {}
    metadata = {}
    for name in ("current", "raw_history", "kinematics", "competition", "interpretable", "ball_light"):
        preds[name], metadata[name] = fit_oof("temporal_" + name, fs[name], d)
    valid = d["ball"][:, 0] > .5
    preds["proximity"] = np.where(valid, -d["ball"][:, 4] * 105, -10000)
    result = summary(d, preds, tuple(k for k in preds if k != "proximity"))
    result["contrasts"] = {name: contrast(result, name, "current") for name in preds if name != "current"}
    far = valid & (d["ball"][:, 4] * 105 > 5)
    result["far_5m"] = summary(subset(d, far), {k: v[far] for k, v in preds.items()}, tuple(k for k in preds if k != "proximity"))
    previous = cached_temporal(d)
    result["prior_gru_same_rows"] = summary(d, {k: previous[k] for k in ("gru_current", "gru_history")}, ("gru_current", "gru_history"))
    result["training"] = metadata
    result["group_ablation"] = "Current versus +kinematics, +competition, +both, and +raw history are separately fitted, fixed-recipe matched ablations, not test-selected permutation tuning."
    result["proxy"] = "Softmax(-distance/2m) among causal tracked candidates, plus entropy and contest count. It is an uncalibrated geometry proxy, not verified possession or next touch."
    result["feature_manifest"] = {
        "kinematics": ["history_valid_fraction", "distance_slope_2s", "distance_slope_0p4s", "relative_dx_2s", "relative_dy_2s", "ball_speed_delta", "time_closest_approach_3s", "tca_defined", "closest_approach_distance", "history_min_distance", "fraction_within2m", "fraction_within5m", "relative_vx", "relative_vy"],
        "competition": ["nearest_distance", "first_second_gap", "rank", "carrier_proxy_probability", "count2m", "count5m", "distance_to_nearest_gap", "carrier_entropy", "scene_valid_fraction"],
        "team_relative_geometry": "Not claimed: this ablation uses anonymous candidate geometry, not inferred possession-team labels.",
    }
    result["status"] = "COMPLETE"
    save(RUN / "temporal_predictions_local.npz", **preds)
    write(RUN / "temporal_feature_ablation.json", result)
    decision(3, "Evaluate interpretable possession/dynamics proxies as predictive geometry, not ground-truth possession.", {"macro_AP": result["macro_match"], "contrasts": result["contrasts"]})
    return result


def load_targets(d):
    with np.load(OLD / "lead_predictions_local.npz", allow_pickle=False) as z:
        for key in ("match", "half_code", "query_frame", "scene_id_local_only", "actor_slot_local_only", "segment_local_only"):
            if not np.array_equal(d[key], z[key]):
                raise ValueError(f"Cached lead target row mismatch: {key}")
        targets = {name: z[f"target__{name}"] for name in old.LEAD_WINDOWS}
        event_frames = {name: z[f"event_frame__{name}"] for name in old.LEAD_WINDOWS}
    return targets, event_frames


def task4():
    d = data()
    seq = sequence()
    fs = feature_sets(d, seq)
    targets, frames = load_targets(d)
    result = {"status": "COMPLETE", "horizons": {}}
    payload = {}
    for horizon in ("lead_0_1s", "lead_1_2s", "lead_2_3s", "within_2s", "within_3s"):
        y = targets[horizon]
        mask = y >= 0
        predictions = {}
        meta = {}
        for name in ("current", "raw_history", "ball_light", "interpretable"):
            predictions[name], meta[name] = fit_oof(horizon + "_" + name, fs[name], d, y)
        predictions["proximity"] = np.where(d["ball"][:, 0] > .5, -d["ball"][:, 4] * 105, -10000)
        eligible = subset(d, mask)
        eligible["target"] = y[mask]
        sm = summary(eligible, {k: v[mask] for k, v in predictions.items()}, ("current", "raw_history", "ball_light", "interpretable"))
        sm["contrasts"] = {k: contrast(sm, k, "current") for k in ("raw_history", "interpretable", "ball_light", "proximity")}
        sm["class_support"] = {old.CLASS_NAMES[k]: int(np.sum(y[mask] == k)) for k in range(1, 8)}
        sm["censoring"] = read(OLD / "lead_time_study.json")["fit_label_censoring_audit"]["targets"][horizon]
        sm["training"] = meta
        sm["actual_lead_seconds"] = {
            "median": float(np.median((frames[horizon][y > 0] - d["query_frame"][y > 0]) / 25)),
            "min": float(np.min((frames[horizon][y > 0] - d["query_frame"][y > 0]) / 25)),
        }
        result["horizons"][horizon] = sm
        for name, p in predictions.items():
            payload[horizon + "__" + name] = p
        write(RUN / "lead_anticipation_progress.json", {"status": "RUNNING", **{k: v for k, v in result.items() if k != "status"}})
        print(horizon, "macro AP", sm["macro_match"], flush=True)
    result["target_boundary"] = "Disjoint one-second bins and cumulative intervals are separate targets; source-window truncation, unknown-actor and segment-change censoring retained. Restart completeness is not independently known."
    save(RUN / "anticipation_predictions_local.npz", **payload)
    write(RUN / "lead_anticipation_progress.json", result)
    write(RUN / "lead_anticipation_study.json", result)
    decision(4, "Distinguish disjoint anticipation from cumulative horizon prevalence; advance to learned scene detection.", {name: value["macro_match"] for name, value in result["horizons"].items()})
    return result


def scene_features(d, fs):
    gg = groups(d)
    compact = np.column_stack((d["x"][:, :16], d["ball"], fs["interpretable"][:, 208:]))
    pooled = []
    sy = []
    sm = []
    for rows in gg:
        values = compact[rows]
        pooled.append(np.concatenate((values.mean(axis=0), values.min(axis=0), values.max(axis=0), values.std(axis=0))))
        sy.append(int(np.any(d["target"][rows] > 0)))
        sm.append(str(d["match"][rows[0]]))
    return np.asarray(pooled, np.float32), np.asarray(sy), np.asarray(sm), gg


def normalize_actor_scores(score, gg, temperature=1):
    output = np.zeros(len(score), np.float64)
    for rows in gg:
        z = np.log(np.maximum(score[rows], 1e-8)) / temperature
        weight = np.exp(z - z.max())
        output[rows] = weight / weight.sum()
    return output


def actor_conditional_nll(y, p, gg):
    losses = [-float(np.mean(np.log(np.maximum(p[rows][y[rows] > 0], 1e-12)))) for rows in gg if np.any(y[rows] > 0)]
    return float(np.mean(losses))


def fit_class_head(xtrain, ytrain, xtest):
    params = dict(PARAMS, min_samples_leaf=20)
    model = HistGradientBoostingClassifier(**params).fit(xtrain, ytrain)
    p = np.full((len(xtest), 7), 1e-8)
    predicted = model.predict_proba(xtest)
    for j, cls in enumerate(model.classes_):
        p[:, int(cls) - 1] = predicted[:, j]
    return p / p.sum(axis=1, keepdims=True)


def task5():
    from threadpoolctl import threadpool_limits
    threadpool_limits(8)
    d = data()
    dev = data("development")
    fs = feature_sets(d, sequence())
    fd = feature_sets(dev, sequence("development"))
    sx, sy, sm, gg = scene_features(d, fs)
    dx, dy, dm, dg = scene_features(dev, fd)
    allx = np.column_stack((fs["raw_history"], fs["interpretable"][:, 208:])).astype(np.float32)
    devx = np.column_stack((fd["raw_history"], fd["interpretable"][:, 208:])).astype(np.float32)
    scene_p = np.zeros(len(sy))
    conditional_p = np.zeros(len(d["target"]))
    joint_p = np.zeros(len(d["target"]))
    class_p = np.zeros((len(d["target"]), 7))
    records = []
    scene_positive_rows = np.zeros(len(d["target"]), bool)
    for rows in gg:
        scene_positive_rows[rows] = np.any(d["target"][rows] > 0)
    for held in FIT:
        start = time.perf_counter()
        tr = d["match"] != held
        te = ~tr
        strn = sm != held
        ste = ~strn
        fold_path = RUN / "predictions_local" / f"hierarchy_fold_{held}.npz"
        dev_path = RUN / "predictions_local" / f"hierarchy_dev_{held}.npz"
        fold_meta = fold_path.with_suffix(".json")
        fingerprint = hashlib.sha256(allx.tobytes() + devx.tobytes() + d["target"].tobytes() + dev["target"].tobytes() + json.dumps(PARAMS, sort_keys=True).encode()).hexdigest()
        if fold_path.exists() and dev_path.exists() and fold_meta.exists():
            saved = read(fold_meta)
            if saved["fingerprint"] == fingerprint and saved["sha256"] == digest(fold_path) and saved["development_sha256"] == digest(dev_path):
                with np.load(fold_path, allow_pickle=False) as z:
                    scene_p[ste] = z["scene"]
                    conditional_p[te] = z["conditional"]
                    joint_p[te] = z["joint"]
                    class_p[te] = z["action"]
                records.append(saved)
                continue
        scene_model = HistGradientBoostingClassifier(**PARAMS).fit(sx[strn], sy[strn])
        scene_p[ste] = scene_model.predict_proba(sx[ste])[:, 1]
        dev_scene = scene_model.predict_proba(dx)[:, 1]
        actor_model = HistGradientBoostingClassifier(**PARAMS).fit(allx[tr & scene_positive_rows], d["target"][tr & scene_positive_rows] > 0)
        dev_actor_raw = actor_model.predict_proba(devx)[:, 1]
        temperature_losses = {}
        for temp in (.5, 1, 2):
            candidate = normalize_actor_scores(dev_actor_raw, dg, temp)
            temperature_losses[str(temp)] = actor_conditional_nll(dev["target"], candidate, dg)
        temp = float(min(temperature_losses, key=temperature_losses.get))
        raw = actor_model.predict_proba(allx[te])[:, 1]
        test_gg = groups(subset(d, te))
        global_keys = [(int(d["half_code"][r[0]]), int(d["query_frame"][r[0]])) for r in gg if str(d["match"][r[0]]) == held]
        td = subset(d, te)
        local_keys = [(int(td["half_code"][r[0]]), int(td["query_frame"][r[0]])) for r in test_gg]
        if global_keys != local_keys:
            raise ValueError("Held-out scene pooling and actor grouping do not align.")
        cond = normalize_actor_scores(raw, test_gg, temp)
        conditional_p[te] = cond
        dev_cond = normalize_actor_scores(dev_actor_raw, dg, temp)
        dev_joint = np.zeros(len(dev["target"]))
        for j, rows in enumerate(dg):
            dev_joint[rows] = dev_scene[j] * dev_cond[rows]
        cprob = fit_class_head(allx[tr & (d["target"] > 0)], d["target"][tr & (d["target"] > 0)], allx[te])
        class_p[te] = cprob
        test_scene_p = scene_p[ste]
        for j, rows in enumerate(test_gg):
            joint_p[np.flatnonzero(te)[rows]] = test_scene_p[j] * cond[rows]
        save(dev_path, scene=dev_scene, actor=dev_cond, joint=dev_joint)
        save(fold_path, scene=scene_p[ste], conditional=conditional_p[te], joint=joint_p[te], action=class_p[te])
        record = {"held_out_match": held, "training_matches": [m for m in FIT if m != held], "train_rows": int(tr.sum()), "test_rows": int(te.sum()), "scene_train": int(strn.sum()), "scene_test": int(ste.sum()), "temperature": temp, "development_temperature_losses": temperature_losses, "seconds": time.perf_counter() - start, "fingerprint": fingerprint, "sha256": digest(fold_path), "development_sha256": digest(dev_path)}
        write(fold_meta, record)
        records.append(record)
        print("hierarchy", held, record["seconds"], flush=True)
    for j, rows in enumerate(gg):
        if not np.isclose(conditional_p[rows].sum(), 1):
            raise ValueError("Conditional actor probability is not normalized.")
    baseline, bmeta = fit_oof("hierarchy_actor_row_full", allx, d)
    baseline_scene = np.array([baseline[rows].max() for rows in gg])
    noisy_or = np.array([1 - np.prod(1 - baseline[rows]) for rows in gg])
    physics = np.array([-(d["ball"][rows, 11].min() * 105) + np.mean(d["ball"][rows, 6]) if np.any(d["ball"][rows, 0] > .5) else -10000 for rows in gg])
    scene_result = {}
    for name, p in (("factorized_scene", scene_p), ("actor_max", baseline_scene), ("actor_noisy_or", noisy_or), ("physics", physics)):
        per = {match: metrics(sy[sm == match], p[sm == match], probability=p[sm == match] if name != "physics" else None) for match in FIT}
        scene_result[name] = {"pooled": metrics(sy, p, probability=p if name != "physics" else None), "per_match": per, "macro_AP": float(np.mean([m["AP"] for m in per.values()]))}
    pred = {"factorized_joint": joint_p, "actor_conditional": conditional_p, "actor_row_full": baseline}
    result = summary(d, pred, tuple(pred))
    result["scene_hazard"] = scene_result
    result["actor_conditional_NLL"] = actor_conditional_nll(d["target"], conditional_p, gg)
    positive = d["target"] > 0
    result["action_head"] = {
        "support": {old.CLASS_NAMES[k]: int(np.sum(d["target"] == k)) for k in range(1, 8)},
        "positive_only_conditional_NLL": float(-np.log(np.maximum(class_p[positive, d["target"][positive] - 1], 1e-12)).mean()),
        "positive_only_accuracy": float(np.mean(np.argmax(class_p[positive], axis=1) + 1 == d["target"][positive])),
        "per_class_AP_end_to_end": {old.CLASS_NAMES[k]: float(average_precision_score(d["target"] == k, joint_p * class_p[:, k - 1])) for k in range(1, 8)},
        "per_class_conditional_AP": {old.CLASS_NAMES[k]: float(average_precision_score(d["target"][positive] == k, class_p[positive, k - 1])) for k in range(1, 8)},
    }
    result["training"] = records
    result["multiple_positive_scenes"] = int(sum(np.sum(d["target"][rows] > 0) > 1 for rows in gg))
    result["status"] = "COMPLETE"
    result["error_taxonomy"] = ["scene event not prioritized", "scene event detected but wrong actor", "correct actor but wrong action", "sparse class uncertainty", "annotation-negative high-score (unadjudicated)", "ball missing or track segment unsupported"]
    result["optional_receiver"] = "NOT_RUN: end-to-end pass decision evaluation prioritized; prospective receiver candidate cache not established for uniform scene grid."
    save(RUN / "hierarchy_predictions_local.npz", scene_p=scene_p, scene_y=sy, scene_match=sm, actor_cond=conditional_p, joint=joint_p, action=class_p, row_baseline=baseline)
    write(RUN / "hierarchical_forecast.json", result)
    decision(5, "Separate event detection, conditional actor selection and action class; do not assume event presence.", {"scene_macro_AP": {k: v["macro_AP"] for k, v in scene_result.items()}, "actor_macro_AP": result["macro_match"], "action": result["action_head"]})
    return result


def select_reviews(scene_scores, gg, d, budget, suppression=125):
    sm = np.array([str(d["match"][r[0]]) for r in gg])
    half = np.array([int(d["half_code"][r[0]]) for r in gg])
    frame = np.array([int(d["query_frame"][r[0]]) for r in gg])
    selected = []
    suppressed = 0
    for match in np.unique(sm):
        indexes = np.flatnonzero(sm == match)
        allowance = max(0, int(np.floor(len(indexes) / 3600 * budget)))
        order = sorted(indexes, key=lambda j: (-float(scene_scores[j]), int(half[j]), int(frame[j])))
        chosen = []
        for j in order:
            if len(chosen) >= allowance:
                break
            if not np.isfinite(scene_scores[j]):
                continue
            if any(half[j] == half[p] and abs(frame[j] - frame[p]) < suppression for p in chosen):
                suppressed += 1
                continue
            chosen.append(j)
        selected.extend(chosen)
    return np.asarray(selected, int), suppressed


def review_metrics(selected, gg, d, actor_scores, events, k=1):
    all_events = set()
    for i in np.flatnonzero(events >= 0):
        all_events.add((str(d["match"][i]), int(d["half_code"][i]), int(d["actor_slot_local_only"][i]), int(d["segment_local_only"][i]), int(events[i])))
    hits = set()
    scene_hits = []
    leads = []
    by_match = {}
    for j in selected:
        rows = gg[j]
        order = rows[np.argsort(-actor_scores[rows], kind="stable")[:k]]
        matched = []
        for row in order:
            if events[row] >= 0:
                key = (str(d["match"][row]), int(d["half_code"][row]), int(d["actor_slot_local_only"][row]), int(d["segment_local_only"][row]), int(events[row]))
                hits.add(key)
                matched.append(row)
                leads.append((events[row] - d["query_frame"][row]) / 25)
        scene_hits.append(bool(matched))
        match = str(d["match"][rows[0]])
        by_match.setdefault(match, {"reviewed": 0, "hit_scenes": 0})
        by_match[match]["reviewed"] += 1
        by_match[match]["hit_scenes"] += int(bool(matched))
    return {
        "reviewed_scenes": len(selected), "reviewed_actor_candidates": len(selected) * k,
        "unique_events_recalled": len(hits), "unique_supported_events": len(all_events),
        "event_recall": len(hits) / max(len(all_events), 1),
        "unique_events_per_reviewed_scene": len(hits) / max(len(selected), 1),
        "actor_hit_at_k_per_review": float(np.mean(scene_hits)) if len(selected) else None,
        "matches_with_recovered_event": sum(value["hit_scenes"] > 0 for value in by_match.values()),
        "evaluated_matches": len(set(d["match"])),
        "fraction_matches_with_event_recovered": sum(value["hit_scenes"] > 0 for value in by_match.values()) / max(len(set(d["match"])), 1),
        "median_event_lead_seconds": float(np.median(leads)) if leads else None,
        "false_review_scenes_annotation_relative": int(len(selected) - sum(scene_hits)),
        "per_match": by_match,
    }


def task6():
    d = data()
    targets, frames = load_targets(d)
    gg = groups(d)
    with np.load(RUN / "hierarchy_predictions_local.npz", allow_pickle=False) as z:
        hp = z["scene_p"].copy()
        cond = z["actor_cond"].copy()
        rowp = z["row_baseline"].copy()
    sy = np.array([int(np.any(d["target"][r] > 0)) for r in gg])
    scene_eligible = np.array([np.all(targets["within_1s"][r] >= 0) for r in gg])
    sm = np.array([str(d["match"][r[0]]) for r in gg])
    rowmax = np.array([rowp[r].max() for r in gg])
    physics = np.array([-(d["ball"][r, 11].min() * 105) + np.mean(d["ball"][r, 6]) if np.any(d["ball"][r, 0] > .5) else -10000 for r in gg])
    rawdistance = np.where(d["ball"][:, 0] > .5, -d["ball"][:, 4] * 105, -10000)
    uncertainty = hp.copy()
    selection = []
    dev = data("development")
    dg = groups(dev)
    dt, de, da = old.build_label_targets(dev, "development")
    dev_scene_eligible = np.array([np.all(dt["within_1s"][r] >= 0) for r in dg])
    for held in FIT:
        with np.load(RUN / "predictions_local" / f"hierarchy_dev_{held}.npz", allow_pickle=False) as z:
            ds = z["scene"]
            dp = z["actor"]
        entropy = -(ds * np.log(np.maximum(ds, 1e-12)) + (1 - ds) * np.log(np.maximum(1 - ds, 1e-12))) / np.log(2)
        candidates = {}
        for coefficient in (0, .1, .25):
            ids, _ = select_reviews(np.where(dev_scene_eligible, ds + coefficient * entropy, -np.inf), dg, dev, 10)
            candidates[str(coefficient)] = review_metrics(ids, dg, dev, dp, de["within_1s"], 3)["unique_events_per_reviewed_scene"]
        coefficient = float(max(candidates, key=candidates.get))
        mask = sm == held
        ee = -(hp[mask] * np.log(np.maximum(hp[mask], 1e-12)) + (1 - hp[mask]) * np.log(np.maximum(1 - hp[mask], 1e-12))) / np.log(2)
        uncertainty[mask] = hp[mask] + coefficient * ee
        selection.append({"held_out_match": held, "development_candidates": candidates, "selected_coefficient": coefficient})
    result = {"status": "COMPLETE", "strategies": {}, "uncertainty_development_selection": selection}
    for name, scene_scores, actor_scores in (
        ("hierarchy_top1", hp, cond),
        ("actor_max_top1", rowmax, rowp),
        ("hierarchy_top3", hp, cond),
        ("uncertainty_top3", uncertainty, cond),
        ("physics_nearest_top1", physics, rawdistance),
        ("physics_nearest_top3", physics, rawdistance),
    ):
        k = 3 if name.endswith("top3") else 1
        result["strategies"][name] = {}
        for budget in (5, 10, 20):
            selected, overlap = select_reviews(np.where(scene_eligible, scene_scores, -np.inf), gg, d, budget)
            values = review_metrics(selected, gg, d, actor_scores, frames["within_1s"], k)
            values.update({"budget_scenes_per_match_hour": budget, "suppression_seconds": 5, "suppressed_nearby_candidates": overlap, "annotation_relative_false_per_match_hour": values["false_review_scenes_annotation_relative"] / (len(gg) / 3600), "annotation_relative_false_per_actor_hour": values["false_review_scenes_annotation_relative"] / (len(d["target"]) / 3600)})
            result["strategies"][name][str(budget)] = values
    result["protocol"] = "Uniform scene grid including annotation-negative scenes; within-match offline fixed quotas, five-second suppression. Budgets are comparable scenes, top3 costs three actor reviews. No real-time threshold or verified physical false-alarm claim."
    result["support"] = {"scene_rows": len(gg), "positive_scenes": int(sy.sum()), "scene_hours": len(gg) / 3600, "actor_hours": len(d["target"]) / 3600, "censored_scene_queries": int(np.sum(~scene_eligible)), "target_cache_censored_rows": int(np.sum(targets["within_1s"] < 0)), "eligibility": "Unknown one-second source-target scenes cannot be selected as negative reviews. Workload hours retain the complete original uniform-grid exposure."}
    save(RUN / "review_scene_scores_local.npz", hazard=hp, actor_max=rowmax, physics=physics, uncertainty=uncertainty, scene_match=sm)
    write(RUN / "scene_review_utility.json", result)
    decision(6, "Compare offline scene review quotas rather than treating event-conditioned actor ranking as honest detection.", {name: v["10"] for name, v in result["strategies"].items()})
    return result


def task8():
    before = read(RUN / "input_hashes_before.json")
    changed = []
    missing = []
    for relative, expected in before.items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        elif digest(path) != expected:
            changed.append(relative)
    if changed or missing:
        raise ValueError(f"Protected input changed/missing: {changed}, {missing}")
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", str(RUN / "test_mission.py")], cwd=ROOT, capture_output=True, text=True, check=False)
    tests = {"return_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "command": [sys.executable, "-m", "pytest", "-q", str(RUN / "test_mission.py")]}
    write(RUN / "test_results.json", tests)
    if completed.returncode:
        raise RuntimeError(f"Causal/metric tests failed: {completed.stdout}\n{completed.stderr}")
    manifests = list((RUN / "predictions_local").glob("*.json"))
    fold_checks = []
    for path in manifests:
        meta = read(path)
        if "training_matches" in meta:
            if meta["held_out_match"] in meta["training_matches"]:
                raise ValueError("Held-out match leaked into fit metadata.")
            if set(meta["training_matches"]) != set(FIT) - {meta["held_out_match"]}:
                raise ValueError("Unexpected fold membership.")
            fold_checks.append(path.name)
    env = read(RUN / "environment.json")
    result = {
        "status": "COMPLETE", "protected_files_verified_unchanged": len(before),
        "changed": changed, "missing": missing, "tests": tests,
        "fold_membership_manifests_checked": len(fold_checks),
        "generalization": "NO_NEW_SEALED_MATCH",
        "evidence": "The existing 18-shard manifest covers six fit, one development and two previously exposed retrospective matches. Recovered match 118575 has no verified equivalent tracking cohort. No new licensed, equivalent, unexposed match has been established; no speculative downloads or access bypass attempted.",
        "retrospective_scored": False, "development_used_for_selection_only": True,
        "cpu": "Fixed HGB, 8 thread limit; per-fold seconds recorded.",
        "gpu": env["gpu"], "restart_censoring": "Source restart-state completeness unverified; cannot count all ambiguous restarts or certify annotation-negative windows.",
        "physical_synchronization": "UNVERIFIED; source-clock consistency only.",
    }
    write(RUN / "generalization_and_integrity.json", result)
    decision(8, "Verify original artifacts unchanged and restrict generalization to six fit-match clusters.", {"files": len(before), "tests_returncode": completed.returncode, "sealed_match": False})
    return result


def task9():
    from adaptive import temporal_multihorizon, proximity_transition
    c = read(RUN / "lead_anticipation_study.json")
    h = read(RUN / "hierarchical_forecast.json")
    r = read(RUN / "scene_review_utility.json")
    selected = {
        "stage": "9", "status": "BRANCHES_LOCKED",
        "evidence": {
            "long_lead_macro": {k: v["macro_match"] for k, v in c["horizons"].items()},
            "scene_hazard": {k: v["macro_AP"] for k, v in h["scene_hazard"].items()},
            "review_top3_10": r["strategies"]["hierarchy_top3"]["10"],
        },
        "candidates": [
            {"name": "multihorizon_capacity_matched_TCN", "information_gain": "Tests learned joint temporal targets versus instantaneous TCN and strong HGB dynamics; 36 CUDA fits, 3 seeds x 6 folds x 2 arms."},
            {"name": "future_geometric_proximity_switch", "information_gain": "Changes supervision to annotation-independent future nearest-player proxy, isolates ball-carrier-like switches without claiming real possession; two HGB variants plus distance/intercept controls."},
            {"name": "larger_transformer", "information_gain": "Lower priority: no evidence that larger model depth is the bottleneck; HGB dynamics are strong."},
        ],
        "selected": ["multihorizon_capacity_matched_TCN", "future_geometric_proximity_switch"],
        "hypotheses": [
            "A compact TCN sharing three disjoint hazard heads may improve longer-lead generalization; compare matched instantaneous model using only fit normalization and development stopping.",
            "Causal motion history should improve future-nearest-player switching relative to current distance; future geometry is label-only and never a model input.",
        ],
    }
    lock_path = RUN / "adaptive_branch_lock.json"
    if lock_path.exists():
        if read(lock_path) != selected:
            raise ValueError("Adaptive branch lock differs from current primary evidence; do not silently relock after training.")
    else:
        write(lock_path, selected)
    (RUN / "branch_selection.md").write_text(
        "# Evidence-driven branches selected after tasks 1–6\n\n"
        "## A: Matched multihorizon temporal learning\n\n"
        "Temporal HGB dynamics improved annotation anticipation; test a compact causal TCN jointly trained on the three disjoint windows, with an identical-capacity instantaneous TCN and three seeds per six match-excluded folds. "
        "This can reject neural scaling if the strong interpretable HGB remains better. Training stops on match 132877 only.\n\n"
        "## B: Annotation-independent future proximity switches\n\n"
        "BAS completeness limits event intelligence. Predict the nearest tracked player to the ball two seconds later using frozen causal inputs, compare current/dynamics HGB with corrected distance and constant-velocity intercept rules, then evaluate switched-nearest scenes. "
        "Future positions define labels only; roster continuity and future missing-ball cases are censored. This is geometric proximity, not a possession or next-touch label.\n\n"
        "A larger transformer was rejected as lower information gain without evidence of a capacity bottleneck. Both selected comparisons are new targets/models beyond the fixed tasks.\n",
        encoding="utf-8",
    )
    a = temporal_multihorizon()
    b = proximity_transition()
    result = {"status": "COMPLETE", "branch_A": {"status": a["status"], "artifact": "adaptive_multihorizon.json"}, "branch_B": {"status": b["status"], "artifact": "adaptive_proximity_transition.json"}, "decisions": selected}
    write(RUN / "autonomous_continuation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    status(args.stage, "RUNNING", {"command": sys.argv})
    from review_tools import task7
    from adaptive import intercept_fusion
    functions = {"preflight": preflight, "1": task1, "2": task2, "3": task3, "4": task4, "5": task5, "6": task6, "7": task7, "8": task8, "9": task9, "9C": intercept_fusion}
    if args.stage not in functions:
        raise ValueError("Stage is not implemented yet: " + args.stage)
    try:
        result = functions[args.stage]()
    except Exception as exc:
        status(args.stage, "FAILED", {"seconds": time.perf_counter() - start, "exception_type": type(exc).__name__, "error": str(exc)})
        raise
    status(args.stage, "COMPLETE", {"seconds": time.perf_counter() - start})
    print(json.dumps({"stage": args.stage, "status": result.get("status", "COMPLETE"), "seconds": time.perf_counter() - start}), flush=True)


if __name__ == "__main__":
    main()
