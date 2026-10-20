"""First-sight memory with exact 11-sample world inputs and explicit age fallback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.backbone.contracts import Measurement, Observation, WorldInput
from src.backbone.models import ModelSession
from src.belief_observation import encode_observations


@dataclass
class MemoryEntry:
    identity: str
    measurement: Measurement
    last_seen: float
    velocity: np.ndarray


class CausalMemory:
    def __init__(self, models: ModelSession, expiry_s: float = 6., fallback: str = "last_seen"):
        if expiry_s < 2 or fallback not in ("last_seen", "constant_velocity"):
            raise ValueError("Memory requires expiry>=2s and an explicit supported fallback")
        self.models, self.expiry_s, self.fallback = models, expiry_s, fallback
        self.reset()

    def reset(self):
        self.entries = {}
        self.history = deque(maxlen=11)
        self.counter = 0
        self.previous = None
        self.actor_segment = None
        self.current_input = None

    def encode(self) -> WorldInput | None:
        if len(self.history) != 11 or any(obs.actor is None for obs in self.history):
            return None
        observations = list(self.history)
        actor = observations[-1].actor
        if actor is None:
            return None
        if any(obs.actor.segment != actor.segment for obs in observations):
            return None
        if any(m.team not in (1, 2) or m.goalkeeper is None
               for obs in observations for m in obs.measurements):
            return None
        tokens = []
        for obs in observations:
            for m in obs.measurements:
                token = (m.track, m.segment)
                if token not in tokens:
                    tokens.append(token)
        if len(tokens) > 22:
            return None
        actor_index = tokens.index((actor.track, actor.segment))
        index = {token: slot for slot, token in enumerate(tokens)}
        shape = (11, len(tokens))
        xy, observed = np.zeros((*shape, 2), np.float32), np.zeros(shape, bool)
        segments, teams, roles = (np.zeros(shape, np.int32) for _ in range(3))
        for time, obs in enumerate(observations):
            for m in obs.measurements:
                slot = index[(m.track, m.segment)]
                xy[time, slot], observed[time, slot] = m.xy, True
                segments[time, slot], teams[time, slot], roles[time, slot] = m.segment, m.team, m.goalkeeper
        encoded = encode_observations(xy, observed, segments, teams, roles, actor_index, observed.copy())
        identities = {int(encoded["mapping"][i]): self.entries[token].identity
                      for i, token in enumerate(tokens) if token in self.entries}
        return WorldInput(encoded["x"], encoded["mask"], np.asarray(actor.xy, np.float32),
                          encoded["last_xy"], identities)

    def step(self, observation: Observation):
        warnings = list(observation.warnings)
        if observation.discontinuity:
            self.reset()
            warnings.append("EXPLICIT_SOURCE_DISCONTINUITY: all viewpoint identities/history reset")
        key = (observation.source_id, observation.half)
        if self.previous is not None:
            if key != self.previous[:2]:
                self.reset()
                warnings.append("SOURCE_OR_HALF_RESET")
            elif observation.timestamp_s <= self.previous[2]:
                raise ValueError("Backwards seeking requires reconstructing the observation prefix; use replay_prefix")
            elif abs(observation.timestamp_s - self.previous[2] - .2) > 1e-4:
                self.history.clear()
                warnings.append("TIME_GAP: learned history reset; memory falls back until 11 new samples")
        self.previous = (*key, observation.timestamp_s)
        actor = observation.actor
        if actor is not None and self.actor_segment is not None and actor.segment != self.actor_segment:
            self.entries.clear()
            self.history.clear()
            warnings.append("ACTOR_IDENTITY_DISCONTINUITY: viewpoint history reset")
        if actor is not None:
            self.actor_segment = actor.segment
        expired = [token for token, entry in self.entries.items()
                   if observation.timestamp_s - entry.last_seen > self.expiry_s + 1e-6]
        for token in expired:
            del self.entries[token]
        if expired:
            warnings.append(f"EXPIRED_MEMORY: {len(expired)} entities beyond {self.expiry_s:g}s policy")
        for m in observation.measurements:
            for old in [token for token in self.entries if token[0] == m.track and token[1] != m.segment]:
                del self.entries[old]
                self.history.clear()
                warnings.append("OBSERVED_IDENTITY_DISCONTINUITY: old anonymous memory retired")
            token = (m.track, m.segment)
            entry = self.entries.get(token)
            if entry is None:
                identity = f"M{self.counter:03d}"
                self.counter += 1
                velocity = np.zeros(2, np.float32)
            else:
                identity = entry.identity
                dt = observation.timestamp_s - entry.last_seen
                velocity = np.clip((np.asarray(m.xy) - entry.measurement.xy) / max(dt, .2), -12, 12)
            self.entries[token] = MemoryEntry(identity, m, observation.timestamp_s, velocity)
        self.history.append(observation)
        self.current_input = self.encode()
        learned = self.models.predict(self.current_input) if self.current_input is not None and self.models.world is not None else None
        if self.models.world is not None and learned is None:
            warnings.append("WORLD_CONTRACT_UNAVAILABLE: warm-up, missing actor, unknown team/role or >22 first-sight tracks; explicit fallback")
        if actor is None:
            warnings.append("ACTOR_MISSING: candidate geometry unavailable")
        elif actor.team not in (1, 2):
            warnings.append("ACTOR_TEAM_UNKNOWN: geometric ordering unavailable; source adapter must supply known affiliation")
        by_identity = ({identity: slot for slot, identity in self.current_input.identities.items()}
                       if self.current_input is not None else {})
        visible = {(m.track, m.segment) for m in observation.measurements}
        entities = []
        for token, entry in self.entries.items():
            age = max(0., observation.timestamp_s - entry.last_seen)
            now = token in visible
            center = np.asarray(entry.measurement.xy, np.float32).copy()
            sigma = np.zeros(2, np.float32) if now else np.full(2, 1.5 + 1.1 * age, np.float32)
            backend = "observed_tracking" if now else self.fallback
            reason = None if now else ("older_than_2s_model_context" if age > 2 + 1e-5 else "insufficient_model_history")
            slot = by_identity.get(entry.identity)
            if not now:
                if learned is not None and slot is not None and age <= 2 + 1e-5:
                    center, sigma = learned["mean"][slot], learned["std"][slot]
                    backend, reason = self.models.backend, None
                elif self.fallback == "constant_velocity" or self.models.backend == "constant_velocity":
                    center += entry.velocity * age
                    backend = "constant_velocity"
            entities.append({
                "id": entry.identity, "xy": center.tolist(), "std_m": sigma.tolist(),
                "visible": now, "age_s": age, "team": entry.measurement.team,
                "goalkeeper": entry.measurement.goalkeeper,
                "actor": entry.measurement.track == observation.actor_track,
                "backend": backend, "fallback_reason": reason,
            })
        if any(not e["visible"] and max(e["std_m"]) > 5 for e in entities):
            warnings.append("LOW_CONFIDENCE_MEMORY: at least one remembered sigma exceeds 5m; not calibrated real perception")
        if any(abs(e["xy"][0]) > 52.5 or abs(e["xy"][1]) > 34 for e in entities):
            warnings.append("OUTSIDE_PITCH_ESTIMATE: inferred/fallback state exceeds nominal pitch; inspect uncertainty")
        return {"entities": entities, "actor_available": actor is not None,
                "actor_xy": list(actor.xy) if actor is not None else None,
                "actor_team": actor.team if actor is not None else None,
                "anonymous_occupancy": {
                    "status": "AVAILABLE" if learned is not None else "UNAVAILABLE",
                    "all_hidden": learned["occupancy"].tolist() if learned is not None else None,
                    "never_seen_defender": learned["anonymous_defender"].tolist() if learned is not None else None,
                    "label": "Unobserved within the model's 2s window, NOT necessarily never seen in this replay. "
                             "May overlap older retained memories; displayed separately, no invented identities or risk fusion.",
                }, "model_contract_met": self.current_input is not None,
                "warnings": warnings}
