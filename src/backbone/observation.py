"""Sequential synthetic FOV; causal heading and causal occlusion intervals."""

from __future__ import annotations

from collections import deque

import numpy as np

from src.backbone.contracts import Observation, TrackingFrame
from src.player_centric_dataset import stable_seed
from src.player_centric_visibility import estimate_heading, visible_players


class SyntheticObservationProvider:
    def __init__(self, actor_track: str, seed: int = 42, fov: int = 120, dropout: bool = True):
        if fov not in (90, 120, 180, 360):
            raise ValueError("Unsupported synthetic FOV")
        self.actor_track, self.seed, self.fov, self.dropout = actor_track, seed, fov, dropout
        self.reset()

    def reset(self):
        self.history = deque(maxlen=6)
        self.previous = None
        self.actor_segment = None
        self.occluded_until = {}

    def observe(self, frame: TrackingFrame) -> Observation:
        if frame.discontinuity:
            self.reset()
        key = (frame.source_id, frame.half)
        if self.previous is not None and (
                key != self.previous[:2] or frame.index <= self.previous[2]):
            self.reset()
        elif self.previous is not None and frame.index - self.previous[2] != 5:
            self.history.clear()
        self.previous = (*key, frame.index)
        actor = next((m for m in frame.measurements if m.track == self.actor_track), None)
        if actor is None:
            self.history.clear()
            return Observation(*key, frame.index, frame.timestamp_s, self.actor_track, (),
                               (*frame.warnings, "ACTOR_MISSING: no viewpoint, no lane assessment"),
                               frame.discontinuity)
        if actor.segment != self.actor_segment:
            self.history.clear()
            self.actor_segment = actor.segment
        self.history.append(actor.xy)
        actor_history = np.asarray(self.history, np.float32)[:, None]
        seed = stable_seed(f"{frame.source_id}:{self.actor_track}:{frame.index}", self.seed)
        heading, reliable, _ = estimate_heading(actor_history, np.ones(actor_history.shape[:2], bool),
                                                0, seed=seed, mode="motion_random")
        measurements = frame.measurements
        xy = np.asarray([m.xy for m in measurements], np.float32)
        actor_slot = next(i for i, m in enumerate(measurements) if m.track == self.actor_track)
        visible = visible_players(xy, np.ones(len(xy), bool), actor_slot, heading,
                                  self.fov, 45 if self.fov != 360 else 1000)
        for i, measurement in enumerate(measurements):
            if i == actor_slot or not self.dropout:
                continue
            token = (measurement.track, measurement.segment)
            rng = np.random.default_rng(stable_seed(f"{token}:{frame.index}", seed))
            # Streaming counterpart of training's artificial hidden intervals:
            # starts are decided NOW, never from the future length of the replay.
            if self.occluded_until.get(token, -1) <= frame.timestamp_s and rng.random() < .04:
                self.occluded_until[token] = frame.timestamp_s + float(rng.uniform(.6, 2.2))
            if self.occluded_until.get(token, -1) > frame.timestamp_s:
                visible[i] = False
        warnings = [*frame.warnings, "Synthetic player visibility; tracking-based prototype."]
        if not reliable:
            warnings.append("Motion heading unreliable: seeded random direction, NOT gaze")
        if frame.missing_count:
            warnings.append(f"Source has {frame.missing_count} {frame.missing_count_semantics}; no imputation")
        return Observation(*key, frame.index, frame.timestamp_s, self.actor_track,
                           tuple(m for m, allowed in zip(measurements, visible) if allowed), tuple(warnings),
                           frame.discontinuity)
