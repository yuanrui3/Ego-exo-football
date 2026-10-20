"""Synthetic player-centric field-of-view simulation from tracking coordinates."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

HeadingMode = Literal["motion_random", "random", "fixed_x"]


def estimate_heading(
    xy: np.ndarray,
    observed: np.ndarray,
    actor_index: int,
    *,
    seed: int,
    mode: HeadingMode = "motion_random",
    min_displacement_m: float = 0.85,
) -> tuple[float, bool, float]:
    """Estimate movement heading from a causal prefix; never treat it as gaze."""
    if mode not in ("motion_random", "random", "fixed_x"):
        raise ValueError(f"Unsupported heading assumption: {mode}")
    rng = np.random.default_rng(seed)
    if mode == "random":
        return float(rng.uniform(-math.pi, math.pi)), False, 0.0
    valid = np.flatnonzero(observed[:, actor_index])[-6:]
    if len(valid) >= 3:
        first, last = xy[valid[0], actor_index], xy[valid[-1], actor_index]
        delta = last - first
        displacement = float(np.linalg.norm(delta))
        elapsed = max((int(valid[-1]) - int(valid[0])) * 0.2, 1e-6)
        speed = displacement / elapsed
        if displacement >= min_displacement_m:
            return float(math.atan2(float(delta[1]), float(delta[0]))), True, speed
    if mode == "fixed_x":
        return 0.0, False, 0.0
    return float(rng.uniform(-math.pi, math.pi)), False, 0.0


def visible_players(
    xy: np.ndarray,
    observed: np.ndarray,
    actor_index: int,
    heading_radians: float,
    fov_degrees: float,
    max_range_m: float,
) -> np.ndarray:
    """Return visible player mask for one time step with no synthetic occlusion."""
    if not 0 < fov_degrees <= 360 or max_range_m <= 0:
        raise ValueError("FOV must be in (0, 360] and range must be positive")
    if not observed[actor_index]:
        return np.zeros_like(observed, dtype=bool)
    delta = xy - xy[actor_index]
    distance = np.linalg.norm(delta, axis=-1)
    if fov_degrees == 360:
        angular = np.ones_like(observed, dtype=bool)
    else:
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        offset = np.arctan2(np.sin(bearing - heading_radians), np.cos(bearing - heading_radians))
        angular = np.abs(offset) <= math.radians(fov_degrees) / 2
    result = observed & angular & (distance <= max_range_m)
    result[actor_index] = True
    return result


def simulate_visibility(
    xy: np.ndarray,
    observed: np.ndarray,
    actor_index: int,
    *,
    fov_degrees: float,
    max_range_m: float,
    seed: int,
    heading_mode: HeadingMode = "motion_random",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate visibility at each historical time using only that time's prefix."""
    if xy.ndim != 3 or xy.shape[-1] != 2 or observed.shape != xy.shape[:2]:
        raise ValueError("Expected xy [time,player,2] and observed [time,player]")
    masks = np.zeros_like(observed, dtype=bool)
    headings = np.zeros(len(xy), dtype=np.float32)
    reliable = np.zeros(len(xy), dtype=bool)
    for frame in range(len(xy)):
        heading, good, _ = estimate_heading(
            xy[:frame + 1], observed[:frame + 1], actor_index,
            seed=seed + frame * 104729, mode=heading_mode,
        )
        headings[frame], reliable[frame] = heading, good
        masks[frame] = visible_players(
            xy[frame], observed[frame], actor_index, heading, fov_degrees, max_range_m,
        )
    return masks, headings, reliable
