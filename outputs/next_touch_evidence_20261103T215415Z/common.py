from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PRIOR = ROOT / "outputs" / "ball_anticipation_20261103T000141Z"
sys.path.insert(0, str(PRIOR))
spec = importlib.util.spec_from_file_location("inherited_mission", PRIOR / "mission.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
sys.path.insert(0,str(RUN))
FIT = prior.FIT
PARAMS = dict(prior.PARAMS)
read, write, save, digest = prior.read, prior.write, prior.save, prior.digest
groups, subset, metrics, summary, contrast = prior.groups, prior.subset, prior.metrics, prior.summary, prior.contrast
threadpool_limits(8)


def event_rows(match, half, raw):
    import build_cohort as build
    meta = read(ROOT / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{match}_{half}.json")
    assoc = read(ROOT / "outputs" / "trajectory_audit_20261016" / f"associations_{match}_{half}.json")
    ids = {str(k): int(v) for k, v in assoc["slots"].items()}
    counts = {slot: list(ids.values()).count(slot) for slot in set(ids.values())}
    return build._event_rows(match, half, meta["gsr_info"], len(raw["xy"]), ids, {k for k, v in counts.items() if v == 1}, raw)


def source(match, half):
    path = ROOT / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{match}_{half}.npz"
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    track = prior.old.load_ball(ROOT / "data" / f"{match}_{half}_ball.npz")
    return raw, track


def log(stage, text, evidence):
    record = {"utc": datetime.now(timezone.utc).isoformat(), "stage": str(stage), "interpretation": text, "evidence": evidence}
    with (RUN / "decisions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


def status(stage, state, **extra):
    p = RUN / "stage_status.json"
    values = read(p) if p.exists() else {}
    values[str(stage)] = {"status": state, "utc": datetime.now(timezone.utc).isoformat(), **extra}
    write(p, values)


def fit(tag, x, d, y, train_extra=None, dev=None, sample_weight=None, all_queries=False):
    """Matched fold fits; future-derived validity is target censoring, never an input."""
    valid = y >= 0
    p = np.full(len(y), np.nan, np.float32)
    devp = []
    fingerprint = hashlib.sha256(json.dumps(PARAMS, sort_keys=True).encode() + tag.encode())
    for array in (x, y, d["match"], train_extra if train_extra is not None else np.zeros(0), sample_weight if sample_weight is not None else np.zeros(0)):
        fingerprint.update(np.asarray(array).tobytes())
    if dev is not None:
        fingerprint.update(dev.tobytes())
    if all_queries:
        fingerprint.update(b":score-all-current-queries-v1")
    signature = fingerprint.hexdigest()
    for held in FIT:
        path = RUN / "fits_local" / f"{tag}_{held}.npz"
        mp = path.with_suffix(".json")
        tr = (d["match"] != held) & valid
        if train_extra is not None:
            tr &= train_extra
        te = (d["match"] == held) & (np.ones(len(y),bool) if all_queries else valid)
        if not tr.any() or not te.any() or np.unique(y[tr] > 0).size != 2:
            raise ValueError(f"Insufficient training/test support {tag}:{held}")
        if path.exists() and mp.exists() and read(mp)["fingerprint"] == signature and read(mp)["sha256"] == digest(path):
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z["rows"], np.flatnonzero(te))
                p[te] = z["p"]
                if dev is not None:
                    devp.append(z["dev"])
            continue
        start = time.perf_counter()
        model = HistGradientBoostingClassifier(**PARAMS)
        model.fit(x[tr], y[tr] > 0, sample_weight=sample_weight[tr] if sample_weight is not None else None)
        p[te] = model.predict_proba(x[te])[:, 1]
        payload = {"p": p[te], "rows": np.flatnonzero(te)}
        if dev is not None:
            payload["dev"] = model.predict_proba(dev)[:, 1]
            devp.append(payload["dev"])
        save(path, **payload)
        write(mp, {"fingerprint": signature, "sha256": digest(path), "training_matches": [m for m in FIT if m != held], "held": held, "train_rows": int(tr.sum()), "test_rows": int(te.sum()), "seconds": time.perf_counter() - start, "features": x.shape[1], "seed": 42})
        print(tag, held, "done", flush=True)
    return p, np.stack(devp) if dev is not None else None


def evaluate(d, y, predictions, probability=()):
    valid = y >= 0
    if not valid.any():
        return {"status": "BLOCKED", "reason": "zero eligible target rows"}
    dd = subset(d, valid)
    dd["target"] = y[valid]
    r = summary(dd, {k: v[valid] for k, v in predictions.items()}, probability)
    r["censored_rows"] = int((~valid).sum())
    r["per_half"] = {}
    for match in FIT:
        for half in (1, 2):
            m = valid & (d["match"] == match) & (d["half_code"] == half)
            if m.any():
                r["per_half"][f"{match}_{half}"] = {k: metrics(y[m], p[m], probability=p[m] if k in probability else None) for k, p in predictions.items()}
    return r


def bas_targets(d):
    return prior.load_targets(d)


def preflight():
    import torch
    manifest = dict(read(PRIOR / "input_hashes_before.json"))
    for p in PRIOR.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            manifest[str(p.relative_to(ROOT))] = digest(p)
    for match in (*FIT, "132877"):
        p = ROOT / "data" / f"{match}_12_class_events.json"
        manifest[str(p.relative_to(ROOT))] = digest(p)
        for half in ("1st", "2nd"):
            p = ROOT / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{match}_{half}.json"
            manifest[str(p.relative_to(ROOT))] = digest(p)
    pack = ROOT / "analysis" / "pass_video_review_20261015_204039_494955"
    for p in pack.rglob("*"):
        if p.is_file() and p.suffix in (".csv", ".json"):
            manifest[str(p.relative_to(ROOT))] = digest(p)
    for relative, expected in manifest.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Protected input changed: {relative}")
    write(RUN / "source_hashes_before.json", manifest)
    hashes = []
    for name in ("return_ball_centric_action_autonomy_20261102T222620Z.zip", "return_ball_anticipation_20261103T000141Z.zip"):
        p = ROOT / "outputs" / name
        import zipfile
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
        if bad:
            raise ValueError(f"Historical ZIP CRC failure: {bad}")
        hashes.append({"path": str(p.relative_to(ROOT)), "sha256": digest(p), "bytes": p.stat().st_size, "crc": "PASS"})
    write(RUN / "archive_reconciliation.json", {"artifacts": hashes, "conclusion": "66f9... refers to the older ball-centric archive; a903... refers to the later ball-anticipation archive. Different filenames/runs, not the same bytes. Actual current hashes above were recomputed."})
    git = subprocess.run(["git", "--no-pager", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    (RUN / "git_status_before.txt").write_text(git, encoding="utf-8")
    d = prior.data()
    write(RUN / "environment.json", {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "rows": len(d["target"]), "scenes": len(groups(d)), "positive_rows": int(np.sum(d["target"] > 0)), "hashes_verified": len(manifest)})
    write(RUN / "dependency_graph.json", {"source": "hash-pinned native tracking+ball+BAS and prior immutable loaders", "edges": [["0","1"],["1","2"],["1","3"],["3","4"],["4","5"],["5","6"],["6","7"],["8","9"],["7","9"],["9","10"]]})
    log(0, "Prior archive discrepancy is reconciled as two distinct archives. Pin inherited sources and the fresh protocol before new fits; historical reviews are audited separately.", {"protected_files": len(manifest), "rows": len(d["target"]), "archives": hashes})
    return {"status": "COMPLETE", "protected_files": len(manifest)}
