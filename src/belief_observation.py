"""Causal anonymous tracking observations and probabilistic heading simulation."""

from __future__ import annotations

import numpy as np

from src.player_centric_visibility import simulate_visibility

GRID_X, GRID_Y = 21, 14
GRID = np.stack(np.meshgrid(
    -52.5 + (np.arange(GRID_X) + .5) * 5,
    -34 + (np.arange(GRID_Y) + .5) * (68 / GRID_Y),
), -1).reshape(-1, 2).astype(np.float32)
SCALE = np.asarray([52.5, 34], np.float32)


def occupancy(xy, selected):
    result = np.zeros(GRID_X * GRID_Y, np.float32)
    points = xy[selected]
    if len(points):
        ix = np.clip(((points[:, 0] + 52.5) / 105 * GRID_X).astype(int), 0, GRID_X - 1)
        iy = np.clip(((points[:, 1] + 34) / 68 * GRID_Y).astype(int), 0, GRID_Y - 1)
        result[iy * GRID_X + ix] = 1
    return result


def heading_visibility_probability(xy, observed, actor, angles, probabilities, fov=120, radius=45):
    """Integrate a supplied causal heading distribution; no hard gaze assertion."""
    p = np.asarray(probabilities, np.float64)
    if np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("Heading distribution must have finite nonnegative mass")
    p = p / p.sum()
    delta = xy - xy[actor]
    bearings = np.arctan2(delta[:, 1], delta[:, 0])
    difference = np.angle(np.exp(1j * (bearings[:, None] - np.asarray(angles)[None])))
    probability = (np.abs(difference) <= np.deg2rad(fov / 2)) @ p
    probability *= observed & (np.linalg.norm(delta, axis=-1) <= radius)
    probability[actor] = float(observed[actor])
    return probability.astype(np.float32)


def encode_observations(xy, observed, segments, teams, roles, actor, visible, dt=.2):
    """Allocate slots only at first sight, resetting memory at known discontinuities."""
    steps, entities = observed.shape
    x = np.zeros((steps, 22, 10), np.float32)
    mask = np.zeros((steps, 22), bool)
    visout = np.zeros_like(mask)
    mapping = np.full(entities, -1, np.int64)
    last_xy = np.zeros((entities, 2), np.float32)
    velocity = np.zeros_like(last_xy)
    last_time = np.full(entities, -1, int)
    last_segment = np.full(entities, -1, int)
    metadata = np.zeros((entities, 2), np.float32)
    for t in range(steps):
        actor_position = xy[t, actor]
        for player in range(entities):
            if not (visible[t, player] and observed[t, player]):
                continue
            if mapping[player] < 0:
                free = set(range(22)) - set(mapping[mapping >= 0].tolist())
                if not free:
                    raise ValueError("More than 22 individually seen tracks in a window")
                mapping[player] = min(free)
            elapsed = (t - last_time[player]) * dt
            same = last_time[player] >= 0 and segments[t, player] == last_segment[player]
            velocity[player] = (
                np.clip((xy[t, player] - last_xy[player]) / max(elapsed, dt), -12, 12)
                if same else 0
            )
            last_xy[player] = xy[t, player]
            last_time[player] = t
            last_segment[player] = segments[t, player]
            metadata[player] = (
                1 if teams[t, player] == teams[t, actor] else -1,
                roles[t, player] == 1,
            )
        for player in np.flatnonzero(mapping >= 0):
            slot = mapping[player]
            age = (t - last_time[player]) * dt
            now = bool(visible[t, player] and observed[t, player])
            x[t, slot] = (
                *((last_xy[player] - actor_position) / SCALE),
                *(velocity[player] / 12), now, 1, min(age, 8) / 8,
                *metadata[player], player == actor,
            )
            mask[t, slot] = True
            visout[t, slot] = now
    last_absolute = np.zeros((22, 2), np.float32)
    last_velocity = np.zeros_like(last_absolute)
    ages = np.zeros(22, np.float32)
    for player in np.flatnonzero(mapping >= 0):
        slot = mapping[player]
        last_absolute[slot] = last_xy[player]
        last_velocity[slot] = velocity[player]
        ages[slot] = (steps - 1 - last_time[player]) * dt
    return {
        "x": x, "mask": mask, "visible": visout, "mapping": mapping,
        "last_xy": last_absolute, "velocity": last_velocity, "age": ages,
    }


def observation_window(raw, anchor, actor, seed, history_frames=150, fov=120,
                       heading_mode="motion_random", hide_intervals=True):
    times = np.arange(anchor - history_frames, anchor + 1, 5)
    if times[0] < 0:
        raise ValueError("Insufficient actor history")
    obs = raw["observed"][times].copy()
    seg = raw["segments"][times]
    if not obs[:, actor].all() or np.any(seg[:, actor] != seg[-1, actor]):
        raise ValueError("Actor history crosses a tracking discontinuity")
    xy = raw["xy"][times]
    vis, _, reliable = simulate_visibility(
        xy, obs, actor, fov_degrees=fov, max_range_m=1000 if fov == 360 else 45,
        seed=seed, heading_mode=heading_mode,
    )
    rng = np.random.default_rng(seed)
    if hide_intervals:
        for player in range(obs.shape[1]):
            if player != actor and rng.random() < .4:
                start = int(rng.integers(0, len(times)))
                length = int(rng.integers(3, len(times) + 1))
                vis[start:start + length, player] = False
    encoded = encode_observations(
        xy, obs, seg, raw["teams"][times], raw["roles"][times], actor, vis,
    )
    current_valid = obs[-1]
    hidden = current_valid & ~vis[-1]
    seen_hidden = np.zeros(22, bool)
    position = np.zeros((22, 2), np.float32)
    future = np.zeros((22, 4, 2), np.float32)
    future_mask = np.zeros((22, 4), bool)
    mapping = encoded.pop("mapping")
    for player in np.flatnonzero(mapping >= 0):
        slot = mapping[player]
        known_segment = seg[-1, player] == seg[max(
            0, len(times) - 1 - int(round(encoded["age"][slot] / .2))
        ), player]
        seen_hidden[slot] = hidden[player] and known_segment
        if current_valid[player] and known_segment:
            position[slot] = raw["xy"][anchor, player]
        for hi, offset in enumerate((10, 25, 50, 75)):
            frame = anchor + offset
            valid = (
                current_valid[player] and known_segment and frame < len(raw["xy"])
                and raw["observed"][frame, player]
                and raw["segments"][frame, player] == raw["segments"][anchor, player]
            )
            if valid:
                future[slot, hi] = raw["xy"][frame, player]
                future_mask[slot, hi] = True
    encoded.update(
        actor_xy=raw["xy"][anchor, actor].copy(),
        occupancy=occupancy(xy[-1], hidden),
        hidden_count=np.float32(hidden.sum()),
        position=position, position_mask=seen_hidden,
        future=future, future_mask=future_mask,
        heading_reliability=np.float32(reliable.mean()),
    )
    return encoded


def memory_belief(last_xy, velocity, age, mask, visible, constant_velocity=False):
    """Same examples/grid as learned belief; missing identities contribute no guessed points."""
    centre = last_xy + velocity * age[..., None] if constant_velocity else last_xy
    sigma = 1.5 + 1.1 * age
    d2 = ((centre[:, None] - GRID[None]) ** 2).sum(-1)
    kernel = np.exp(-d2 / (2 * sigma[:, None] ** 2)) * 25 / (2 * np.pi * sigma[:, None] ** 2)
    kernel *= (mask & ~visible)[:, None]
    return (1 - np.exp(-kernel.sum(0))).astype(np.float32)
