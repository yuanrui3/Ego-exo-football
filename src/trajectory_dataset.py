"""Raw GSR trajectory caches and causal, identity-safe forecasting windows."""

import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import ijson
import numpy as np
from torch.utils.data import Dataset

from src.extract_soccertrack_events import decode_frame

TRAIN_MATCHES = ("117092", "117093", "118576", "118578", "128057", "132831", "132877")
VALIDATION_MATCHES = ("118577", "128058")
DEVELOPMENT_MATCH = "132877"
FPS = 25
HORIZONS = (10, 25, 50, 75)
CACHE_VERSION = 1


def atomic_replace(source, destination, attempts=8):
    source, destination = Path(source), Path(destination)
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.1 * (2 ** attempt), 2.0))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    atomic_replace(tmp, path)


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class BudgetExpired(RuntimeError):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired("Experiment wall-clock budget reached; checkpoint retained.")


def source_signature(path):
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def identity(annotation):
    attrs = annotation.get("attributes") or {}
    player = attrs.get("player_id")
    track = annotation.get("track_id")
    if player in (None, "", -1, "-1") or track is None:
        return None
    return str(player), str(track)


def segment_tracks(xy, observed, tracks, teams, roles):
    """No interpolation: every missing frame, track change or jump starts a segment."""
    clean = observed.copy()
    delta = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    adjacent = observed[1:] & observed[:-1]
    jumps = adjacent & (delta > 1.4)  # >35 m/s for one 40 ms frame.
    changes = adjacent & (
        (tracks[1:] != tracks[:-1]) | (teams[1:] != teams[:-1]) | (roles[1:] != roles[:-1])
    )
    starts = np.zeros_like(observed)
    starts[0] = observed[0]
    starts[1:] = observed[1:] & (~observed[:-1] | jumps | changes)
    segments = np.cumsum(starts, axis=0, dtype=np.int32)
    segments[~clean] = 0
    speeds = delta[adjacent] * FPS
    rounded = np.abs(xy[observed] * 100 - np.round(xy[observed] * 100))
    stats = {
        "observed_player_frames": int(observed.sum()),
        "implausible_adjacent_jumps": int(jumps.sum()),
        "track_team_or_role_changes": int(changes.sum()),
        "segment_count": int(starts.sum()),
        "adjacent_speed_mps_percentiles": np.percentile(speeds, [50, 90, 99, 99.9]).tolist()
        if speeds.size else [],
        "identical_adjacent_position_fraction": float((delta[adjacent] < 1e-6).mean())
        if adjacent.any() else 0.0,
        "coordinates_on_centimeter_grid_fraction": float((rounded < 0.001).mean())
        if rounded.size else 0.0,
        "coordinates_on_1_05m_x_grid_fraction": float(
            (np.abs(xy[observed, 0] / 1.05 - np.round(xy[observed, 0] / 1.05)) < .001).mean()
        ) if observed.any() else 0.0,
        "coordinates_on_0_68m_y_grid_fraction": float(
            (np.abs(xy[observed, 1] / .68 - np.round(xy[observed, 1] / .68)) < .001).mean()
        ) if observed.any() else 0.0,
    }
    return clean, segments, stats


def build_half(source, destination, deadline=float("inf"), frame_limit=None):
    """Scan annotation records once; image metadata is read only up to the image list."""
    source, destination = Path(source), Path(destination)
    meta_path = destination.with_suffix(".json")
    signature = source_signature(source)
    if destination.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["source"] != signature or meta["cache_version"] != CACHE_VERSION:
            raise ValueError(f"Cache/source changed: {source}")
        if file_digest(destination) != meta["cache_sha256"]:
            raise ValueError(f"Cache checksum mismatch: {destination}")
        return meta
    started = time.monotonic()
    with source.open("rb") as stream:
        info = next(ijson.items(stream, "info", use_float=True))
    if float(info["frame_rate"]) != FPS:
        raise ValueError(f"Expected 25 fps GSR, not {info['frame_rate']}")
    n_frames = int(info["seq_length"])
    if frame_limit:
        n_frames = min(n_frames, frame_limit)
    image_ids, seen_frames = {}, set()
    with source.open("rb") as stream:
        for im in ijson.items(stream, "images.item", use_float=True):
            stem = Path(im["file_name"]).stem
            if not stem.isdigit():
                raise ValueError("Non-numeric GSR image filename")
            frame = int(stem) - 1
            if decode_frame(im["image_id"]) != frame:
                raise ValueError("Image-ID decoding disagrees with one-based filename")
            if frame < n_frames:
                if frame in seen_frames or str(im["image_id"]) in image_ids:
                    raise ValueError("Duplicate image frame/ID")
                image_ids[str(im["image_id"])] = frame
                seen_frames.add(frame)
            if len(image_ids) == n_frames:
                break
    if len(image_ids) != n_frames:
        raise ValueError("Incomplete image frame mapping")
    # Slots are identity matching storage only, never learned features.
    capacity = 64
    xy = np.zeros((n_frames, capacity, 2), np.float32)
    observed = np.zeros((n_frames, capacity), bool)
    duplicates = np.zeros_like(observed)
    tracks = np.full((n_frames, capacity), -1, np.int32)
    teams = np.zeros((n_frames, capacity), np.int8)
    roles = np.zeros((n_frames, capacity), np.int8)
    slots, track_codes, counts = {}, {}, Counter()
    last_frame = -1
    with source.open("rb") as stream:
        for index, ann in enumerate(ijson.items(stream, "annotations.item", use_float=True)):
            if index % 10000 == 0:
                check_deadline(deadline)
            iid = str(ann.get("image_id", ""))
            frame = image_ids.get(iid)
            if frame is None:
                if frame_limit and decode_frame(iid) is not None and decode_frame(iid) >= n_frames:
                    # The pilot may stop early only after verifying chronological ordering.
                    if last_frame < n_frames - 2:
                        raise ValueError("Pilot annotations not chronological")
                    break
                continue
            if frame_limit and frame < last_frame:
                raise ValueError("Pilot annotations not chronological")
            last_frame = max(last_frame, frame)
            attrs = ann.get("attributes") or {}
            if ann.get("supercategory") != "object" or attrs.get("role") not in ("player", "goalkeeper"):
                continue
            counts["player_annotations"] += 1
            key = identity(ann)
            if key is None:
                counts["missing_identity"] += 1
                continue
            player, track = key
            if player not in slots:
                if len(slots) >= capacity:
                    raise ValueError("More than 64 player identities in one half")
                slots[player] = len(slots)
            slot = slots[player]
            team = {"left": 1, "right": 2}.get(attrs.get("team"), 0)
            if not team:
                counts["unknown_team"] += 1
                continue
            bp = ann.get("bbox_pitch") or {}
            x, y = bp.get("x_bottom_middle"), bp.get("y_bottom_middle")
            if x is None or y is None or not all(math.isfinite(float(v)) for v in (x, y)):
                counts["missing_or_nonfinite_pitch"] += 1
                continue
            if abs(float(x)) > 65 or abs(float(y)) > 45:
                counts["outside_generous_pitch_bounds"] += 1
                continue
            if observed[frame, slot] or duplicates[frame, slot]:
                duplicates[frame, slot] = True
                observed[frame, slot] = False
                counts["duplicate_identity_records"] += 1
                continue
            code = track_codes.setdefault((player, track), len(track_codes))
            xy[frame, slot] = (x, y)
            observed[frame, slot] = True
            tracks[frame, slot] = code
            teams[frame, slot] = team
            roles[frame, slot] = int(attrs["role"] == "goalkeeper")
    n = len(slots)
    xy, observed, tracks, teams, roles = (
        a[:, :n].copy() for a in (xy, observed, tracks, teams, roles)
    )
    observed, segments, quality = segment_tracks(xy, observed, tracks, teams, roles)
    team_counts = np.stack([(observed & (teams == t)).sum(1) for t in (1, 2)], axis=-1)
    observed_counts = observed.sum(1)
    quality.update({
        **dict(counts), "unique_player_ids": n, "unique_player_track_pairs": len(track_codes),
        "frames": n_frames, "frames_exactly_22": int((observed_counts == 22).sum()),
        "frames_over_22": int((observed_counts > 22).sum()),
        "frames_under_18": int((observed_counts < 18).sum()),
        "frames_more_than_11_on_either_team": int((team_counts > 11).any(1).sum()),
        "player_count_percentiles": np.percentile(observed_counts, [0, 10, 50, 90, 100]).tolist(),
        "pitch_coordinate_min": xy[observed].min(0).tolist() if observed.any() else [],
        "pitch_coordinate_max": xy[observed].max(0).tolist() if observed.any() else [],
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, xy=xy, observed=observed, segments=segments, teams=teams, roles=roles)
    tmp.replace(destination)
    meta = {
        "cache_version": CACHE_VERSION, "source": signature,
        "cache_sha256": file_digest(destination), "gsr_info": info,
        "frame_convention": "zero based decode_frame(image_id) == int(image filename stem)-1",
        "coordinate_source": "bbox_pitch.x/y_bottom_middle; native metric pitch coordinates",
        "identity_rule": "player_id matches identity; track_id/team/role changes split continuity",
        "source_release_commit": "UNKNOWN; local GSR info.version is not a dataset commit",
        "quality": quality, "runtime_seconds": time.monotonic() - started,
    }
    write_json(meta_path, meta)
    return meta


def validate_splits(fitting, validation):
    fitting, validation = set(fitting), set(validation)
    if fitting & validation or not fitting <= set(TRAIN_MATCHES) or validation != set(VALIDATION_MATCHES):
        raise ValueError("Frozen match split violation")


def causal_velocity(history, mask):
    """Least-squares velocity on up to 0.4 s, requiring >=6 real samples; cap at 12 m/s."""
    recent, valid = history[:, -11:], mask[:, -11:]
    times = np.arange(recent.shape[1], dtype=np.float32) / FPS
    weights = valid.astype(np.float32)
    count = weights.sum(1, keepdims=True)
    tbar = (weights * times).sum(1, keepdims=True) / np.maximum(count, 1)
    centered = (times - tbar) * weights
    denominator = (centered ** 2).sum(1, keepdims=True)
    velocity = (recent * centered[..., None]).sum(1) / np.maximum(denominator, 1e-8)
    velocity[count[:, 0] < 6] = 0
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    return velocity * np.minimum(1, 12 / np.maximum(speed, 1e-8))


def make_window(raw, anchor, history_frames=50, max_players=22):
    selected = np.flatnonzero(raw["observed"][anchor])
    if not 1 <= len(selected) <= max_players:
        raise ValueError("Invalid anchor player count")
    frames = np.arange(anchor - history_frames, anchor + 1)
    if frames[0] < 0 or anchor + max(HORIZONS) >= len(raw["xy"]):
        raise ValueError("Window outside half")
    segments = raw["segments"][anchor, selected]
    pos = raw["xy"][frames[:, None], selected].transpose(1, 0, 2).copy()
    mask = (
        raw["observed"][frames[:, None], selected]
        & (raw["segments"][frames[:, None], selected] == segments)
    ).T
    current = raw["xy"][anchor, selected].copy()
    pos[~mask] = 0
    velocity = causal_velocity(pos, mask)
    step_velocity = np.zeros_like(pos)
    step_valid = mask[:, 1:] & mask[:, :-1]
    delta = np.diff(pos, axis=1) * FPS
    speed = np.linalg.norm(delta, axis=-1, keepdims=True)
    delta *= np.minimum(1, 12 / np.maximum(speed, 1e-8))
    step_velocity[:, 1:] = np.where(step_valid[..., None], delta, 0)
    relative = np.where(mask[..., None], pos - current[:, None], 0)
    features = np.concatenate([
        relative / 10, step_velocity / 12, mask[..., None].astype(np.float32),
    ], axis=-1)
    target_frames = anchor + np.asarray(HORIZONS)
    target_mask = (
        raw["observed"][target_frames[:, None], selected]
        & (raw["segments"][target_frames[:, None], selected] == segments)
    ).T
    targets = raw["xy"][target_frames[:, None], selected].transpose(1, 0, 2) - current[:, None]
    targets[~target_mask] = 0  # Storage placeholder ONLY, excluded from all losses/metrics.
    team = raw["teams"][anchor, selected]
    role = raw["roles"][anchor, selected]

    def padded(a):
        return np.pad(a, [(0, max_players - len(selected))] + [(0, 0)] * (a.ndim - 1))

    return {
        "history": padded(features).astype(np.float32),
        "current": padded(current).astype(np.float32),
        "velocity": padded(velocity).astype(np.float32),
        "team": padded(team).astype(np.int64),
        "role": padded(role).astype(np.int64),
        "player_mask": np.arange(max_players) < len(selected),
        "target": padded(targets).astype(np.float32),
        "target_mask": padded(target_mask),
    }


class TrajectoryDataset(Dataset):
    def __init__(self, cache_dir, matches, stride=13, history_frames=50, max_windows=None):
        if stride < 1 or history_frames not in (0, 10, 50):
            raise ValueError("Invalid stride/history")
        self.raw, self.windows, self.match_names = [], [], []
        self.history_frames = history_frames
        for match in matches:
            for half in ("1st", "2nd"):
                path = Path(cache_dir) / f"{match}_{half}.npz"
                with np.load(path, allow_pickle=False) as blob:
                    raw = {key: blob[key] for key in blob.files}
                index = len(self.raw)
                counts = raw["observed"].sum(1)
                for frame in range(50, len(counts) - max(HORIZONS), stride):
                    if not 18 <= counts[frame] <= 22:
                        continue
                    team = raw["teams"][frame]
                    if any(np.sum(raw["observed"][frame] & (team == t)) > 11 for t in (1, 2)):
                        continue
                    selected = raw["observed"][frame]
                    seg = raw["segments"][frame, selected]
                    valid = raw["segments"][frame + max(HORIZONS), selected] == seg
                    if valid.sum() < 12:
                        continue
                    self.windows.append((index, frame))
                self.raw.append(raw)
                self.match_names.append((match, half))
        if max_windows is not None and len(self.windows) > max_windows:
            indices = np.linspace(0, len(self.windows) - 1, max_windows, dtype=int)
            self.windows = [self.windows[i] for i in indices]
        if not self.windows:
            raise ValueError(f"No valid trajectory windows for {matches}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        source, frame = self.windows[index]
        row = make_window(self.raw[source], frame, self.history_frames)
        row["source_index"] = source
        row["anchor_frame"] = frame
        return row


def feasibility(cache_path):
    with np.load(cache_path, allow_pickle=False) as blob:
        raw = {key: blob[key] for key in blob.files}
    counts = raw["observed"].sum(1)
    anchors = [f for f in range(50, len(counts) - 75, 13) if 18 <= counts[f] <= 22]
    coverages, moving = [], []
    for frame in anchors:
        row = make_window(raw, frame)
        coverages.append(row["target_mask"].sum(0) / max(1, row["player_mask"].sum()))
        valid = row["target_mask"][:, -1]
        moving.extend(np.linalg.norm(row["target"][valid, -1], axis=-1).tolist())
    result = {
        "windows": len(anchors),
        "target_coverage_by_horizon": np.mean(coverages, axis=0).tolist() if coverages else [],
        "three_second_displacement_median_m": float(np.median(moving)) if moving else 0,
        "three_second_moving_over_0_1m_fraction": float(np.mean(np.asarray(moving) > .1))
        if moving else 0,
    }
    result["suitable"] = bool(
        len(anchors) >= 20 and min(result["target_coverage_by_horizon"], default=0) > .5
        and result["three_second_moving_over_0_1m_fraction"] > .1
    )
    return result
