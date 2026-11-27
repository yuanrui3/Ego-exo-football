"""Synthetic causal-pretraining fixture, not a goalkeeper outcome experiment.

``python -m src.synthetic_goalkeeper_pilot --checkpoint ../receipt.json``
trains a masked-history next-position predictor and measures development
prediction error. It contains no action evaluator or held-scene simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

# In a fresh CLI process these are set before NumPy loads. An embedding process
# may already have initialized BLAS; environment variables cannot verify it.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np

from .player_centric_visibility import simulate_visibility

VERSION = 1
SEED = 864203
HISTORY = 6
TRAIN_FAMILIES = tuple(range(48))
DEV_FAMILIES = tuple(range(48, 60))
# Reserved only: this implementation never generates or evaluates these scenes.
RESERVED_HELD_FAMILIES = tuple(range(60, 84))
TRAIN_SCENES = 12
DEV_SCENES = 10
RESERVED_HELD_SCENES_PER_FAMILY = 10
RIDGE = 1.0
FEATURES = 1 + 2 * HISTORY


def protocol() -> dict:
    """Pretraining fixture protocol; this is not a registered held experiment."""
    return {
        "version": VERSION, "seed": SEED, "history": HISTORY, "ridge": RIDGE,
        "numpy_version": np.__version__, "blas_thread_env_requested": 2,
        "blas_thread_limit_verified": False,
        "train_families": list(TRAIN_FAMILIES), "train_scenes": TRAIN_SCENES,
        "dev_families": list(DEV_FAMILIES), "dev_scenes": DEV_SCENES,
        "reserved_held_families": list(RESERVED_HELD_FAMILIES),
        "reserved_held_scenes_per_family": RESERVED_HELD_SCENES_PER_FAMILY,
        "task": "predict next lateral residual over constant velocity from contiguous-masked prefix",
        "visibility": "fixed forward heading, 90 degree field of view, 30m range",
        "outcome_evaluation": "unavailable",
    }


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("player_centric_visibility.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


_IMPORT_SOURCE_HASH = _source_hash()


def _assert_unchanged(frozen_protocol: dict) -> None:
    if _source_hash() != _IMPORT_SOURCE_HASH or protocol() != frozen_protocol:
        raise ValueError("checkpoint protocol or source mismatch; start a new fixture run")


def _rng(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256(json.dumps([SEED, *parts], separators=(",", ":")).encode())
    return np.random.default_rng(int.from_bytes(digest.digest()[:8], "little"))


def generate_history(family: int, scene: int) -> np.ndarray:
    """Generate a toy lateral attacker trajectory, not a football simulation."""
    if family not in TRAIN_FAMILIES and family not in DEV_FAMILIES:
        raise ValueError("only training and development families are available")
    f = _rng("family", family)
    bias, drift = f.uniform(-1.4, 1.4), f.uniform(-0.2, 0.2)
    r = _rng("scene", family, scene)
    y0 = bias + r.normal(0, 0.7)
    velocity = drift + r.normal(0, 0.12)
    return y0 + velocity * np.arange(HISTORY) + r.normal(0, 0.08, HISTORY)


def contiguous_mask(end: int, *, family: int, scene: int) -> np.ndarray:
    """Mask one nonempty contiguous span in the available prefix."""
    if not 1 <= end < HISTORY - 1:
        raise ValueError("prefix must leave a next-frame target")
    r = _rng("mask", family, scene, end)
    start = int(r.integers(0, end + 1))
    stop = int(r.integers(start + 1, end + 2))
    observed = np.zeros(HISTORY, dtype=bool)
    observed[:end + 1] = True
    observed[start:stop] = False
    return observed


def prefix_features(history: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Construct features from a pre-sliced prefix, ignoring masked values."""
    history = np.asarray(history, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    if history.ndim != 1 or observed.shape != history.shape or not 1 <= len(history) < HISTORY:
        raise ValueError("invalid sliced history or observation mask")
    values = np.zeros(HISTORY)
    mask = np.zeros(HISTORY)
    valid = np.flatnonzero(observed)
    values[valid] = history[valid]
    mask[valid] = 1.0
    return np.concatenate(([1.0], values, mask))


def cv_prediction(history: np.ndarray, observed: np.ndarray) -> float:
    """Predict next frame from the last two observed prefix positions."""
    history = np.asarray(history, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    if history.ndim != 1 or observed.shape != history.shape or not 1 <= len(history) < HISTORY:
        raise ValueError("invalid sliced history or observation mask")
    valid = np.flatnonzero(observed)
    if not len(valid):
        return 0.0
    last = valid[-1]
    if len(valid) == 1:
        return float(history[last])
    previous = valid[-2]
    velocity = (history[last] - history[previous]) / (last - previous)
    return float(history[last] + velocity * (len(history) - last))


def visibility_mask(prefix: np.ndarray) -> np.ndarray:
    """Synthetic goalkeeper's causal field-of-view visibility of the attacker."""
    n = len(prefix)
    xy = np.zeros((n, 2, 2))
    xy[:, 1, 0] = 15.0 - 0.7 * np.arange(n)
    xy[:, 1, 1] = prefix
    observed = np.ones((n, 2), dtype=bool)
    visible, _, _ = simulate_visibility(xy, observed, 0, fov_degrees=90,
                                        max_range_m=30, seed=SEED, heading_mode="fixed_x")
    return visible[:, 1]


def examples(family: int, scene: int):
    history = generate_history(family, scene)
    for end in range(1, HISTORY - 1):
        prefix = history[:end + 1]
        observed = contiguous_mask(end, family=family, scene=scene)[:end + 1]
        observed &= visibility_mask(prefix)
        baseline = cv_prediction(prefix, observed)
        yield prefix_features(prefix, observed), float(history[end + 1]), baseline


def fit_moments(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    """Fit ridge regression; intercept is not penalized."""
    penalty = RIDGE * np.eye(FEATURES)
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, xty)


def _atomic_write(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"state": state, "sha256": _digest(state)}
    fd, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    pending = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        # Windows does not support opening a directory this way. Atomic replace
        # remains available there, but directory durability is not guaranteed.
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        pending.unlink(missing_ok=True)


def run(checkpoint: str | Path, *, max_new_families: int | None = None) -> dict:
    """Resume deterministic training by family; one writer per checkpoint.

    On completion, pretraining weights are persisted before development
    prediction metrics are calculated. The latter are not model selection.
    """
    path = Path(checkpoint)
    if path.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("checkpoint must be outside the source tree")
    if max_new_families is not None and max_new_families < 0:
        raise ValueError("max_new_families must be nonnegative")
    frozen_protocol = protocol()
    _assert_unchanged(frozen_protocol)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            envelope = json.load(handle)
        state = envelope["state"]
        if envelope["sha256"] != _digest(state):
            raise ValueError("checkpoint checksum mismatch")
        if state["protocol"] != frozen_protocol or state["source_sha256"] != _IMPORT_SOURCE_HASH:
            raise ValueError("checkpoint protocol or source mismatch; start a new fixture run")
    else:
        state = {"protocol": frozen_protocol, "source_sha256": _IMPORT_SOURCE_HASH,
                 "families_completed": 0, "examples": 0,
                 "xtx": np.zeros((FEATURES, FEATURES)).tolist(),
                 "xty": np.zeros(FEATURES).tolist(),
                 "pretraining_receipt": None, "development": None}
        _assert_unchanged(frozen_protocol)
        _atomic_write(path, state)

    if not 0 <= state["families_completed"] <= len(TRAIN_FAMILIES):
        raise ValueError("invalid checkpoint progress")
    count = 0
    for index in range(state["families_completed"], len(TRAIN_FAMILIES)):
        if max_new_families is not None and count >= max_new_families:
            return state
        xtx, xty = np.asarray(state["xtx"]), np.asarray(state["xty"])
        for scene in range(TRAIN_SCENES):
            for x, y, baseline in examples(TRAIN_FAMILIES[index], scene):
                xtx += np.outer(x, x)
                xty += x * (y - baseline)
                state["examples"] += 1
        state["xtx"], state["xty"] = xtx.tolist(), xty.tolist()
        state["families_completed"] = index + 1
        _assert_unchanged(frozen_protocol)
        _atomic_write(path, state)
        count += 1

    if state["pretraining_receipt"] is None:
        receipt = {"protocol_sha256": _digest(frozen_protocol), "source_sha256": _IMPORT_SOURCE_HASH,
                   "weights": fit_moments(np.asarray(state["xtx"]), np.asarray(state["xty"])).tolist(),
                   "training_examples": state["examples"]}
        state["pretraining_receipt"] = {"contents": receipt, "sha256": _digest(receipt)}
        _assert_unchanged(frozen_protocol)
        _atomic_write(path, state)

    if state["development"] is None:
        weights = np.asarray(state["pretraining_receipt"]["contents"]["weights"])
        squared_errors = []
        for family in DEV_FAMILIES:
            for scene in range(DEV_SCENES):
                for x, y, baseline in examples(family, scene):
                    squared_errors.append((baseline + float(x @ weights) - y) ** 2)
        state["development"] = {"families": len(DEV_FAMILIES),
                                "scenes": len(DEV_FAMILIES) * DEV_SCENES,
                                "examples": len(squared_errors),
                                "mse": float(np.mean(squared_errors))}
        _assert_unchanged(frozen_protocol)
        _atomic_write(path, state)
    _assert_unchanged(frozen_protocol)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="output path outside the source tree")
    parser.add_argument("--max-new-families", type=int,
                        help="stop after this many additional training families (resume test)")
    args = parser.parse_args()
    state = run(args.checkpoint, max_new_families=args.max_new_families)
    print(json.dumps({"families_completed": state["families_completed"],
                      "training_examples": state["examples"],
                      "pretraining_receipt": state["pretraining_receipt"],
                      "development": state["development"],
                      "held_protocol": "unavailable; no held outcomes accessed",
                      "action_outcomes": "unavailable"}, indent=2))


if __name__ == "__main__":
    main()
