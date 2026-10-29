"""Explicit source, student and evaluation boundaries in pitch-centred metres."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BallState:
    """Ball position and possession are independent, explicitly qualified estimates."""

    position_status: str = "unknown"
    xy: tuple[float, float] | None = None
    possession_status: str = "unknown"
    possessor_track: str | None = None
    possessor_team: int | None = None

    def __post_init__(self):
        if self.position_status not in ("observed", "estimated", "unknown"):
            raise ValueError("Ball position status must be observed, estimated or unknown")
        if (self.xy is None) != (self.position_status == "unknown"):
            raise ValueError("Ball coordinates must be present exactly when position status is known")
        if self.xy is not None and (len(self.xy) != 2 or not all(math.isfinite(v) for v in self.xy)):
            raise ValueError("Known ball position requires finite pitch coordinates")
        if self.possession_status not in ("observed", "estimated", "unknown"):
            raise ValueError("Possession status must be observed, estimated or unknown")
        if self.possession_status == "unknown" and (self.possessor_track is not None or self.possessor_team is not None):
            raise ValueError("Unknown possession cannot assert a possessor")
        if self.possessor_team is not None and self.possessor_team not in (1, 2):
            raise ValueError("Possessor team must be 1/2 or unknown (None)")


@dataclass(frozen=True)
class ActionOutput:
    """Action-general prediction contract; unsupported is an honest first-class result."""

    action_type: str | None = None
    teammate_target: str | None = None
    opponent_target: str | None = None
    location_xy: tuple[float, float] | None = None
    direction_xy: tuple[float, float] | None = None
    timing_start_s: float | None = None
    timing_end_s: float | None = None
    confidence: float | None = None
    support_status: str = "unsupported"
    reason: str = "No action model is configured"

    def __post_init__(self):
        if self.support_status not in ("supported", "partial", "unsupported"):
            raise ValueError("Action support status must be supported, partial or unsupported")
        if self.teammate_target is not None and self.opponent_target is not None:
            raise ValueError("An action cannot target both a teammate and opponent")
        for name, vector in (("location", self.location_xy), ("direction", self.direction_xy)):
            if vector is not None and (len(vector) != 2 or not all(math.isfinite(v) for v in vector)):
                raise ValueError(f"Action {name} must contain two finite coordinates")
        if self.timing_start_s is not None and not math.isfinite(self.timing_start_s):
            raise ValueError("Action start time must be finite")
        if self.timing_end_s is not None and not math.isfinite(self.timing_end_s):
            raise ValueError("Action end time must be finite")
        if (self.timing_start_s is not None and self.timing_end_s is not None
                and self.timing_end_s < self.timing_start_s):
            raise ValueError("Action timing end cannot precede start")
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("Action confidence must be in [0, 1]")
        if self.support_status == "unsupported" and not self.reason:
            raise ValueError("Unsupported actions require an explicit reason")


@dataclass(frozen=True)
class RestartContext:
    restart_type: str | None = None
    status: str = "unknown"
    reason: str | None = None

    def __post_init__(self):
        if self.status not in ("observed", "estimated", "unknown"):
            raise ValueError("Restart context status must be observed, estimated or unknown")
        if self.status == "unknown" and self.restart_type is not None:
            raise ValueError("Unknown restart context cannot assert a restart type")


@dataclass(frozen=True)
class ActionOutcome:
    outcome_type: str | None = None
    status: str = "unknown"
    reason: str | None = None

    def __post_init__(self):
        if self.status not in ("observed", "estimated", "unknown"):
            raise ValueError("Outcome status must be observed, estimated or unknown")
        if self.status == "unknown" and self.outcome_type is not None:
            raise ValueError("Unknown outcome cannot assert an outcome type")


@dataclass(frozen=True)
class Measurement:
    track: str
    segment: int
    xy: tuple[float, float]
    team: int
    goalkeeper: bool | None
    quality: dict | None = None

    def __post_init__(self):
        if not self.track or len(self.xy) != 2 or not all(math.isfinite(v) for v in self.xy):
            raise ValueError("Measurements require an internal track association and finite metre coordinates")
        if self.team not in (0, 1, 2):
            raise ValueError("Team must be 1/2 or explicitly unknown (0)")
        if self.goalkeeper is not None and not isinstance(self.goalkeeper, bool):
            raise ValueError("Goalkeeper role must be true, false or explicitly unknown (null)")


@dataclass(frozen=True)
class TrackingFrame:
    source_id: str
    half: str
    index: int
    timestamp_s: float
    measurements: tuple[Measurement, ...]
    missing_count: int
    discontinuity: bool = False
    warnings: tuple[str, ...] = ()
    missing_count_semantics: str = "absent cache slots (not a missing-player count)"


@dataclass(frozen=True)
class Observation:
    source_id: str
    half: str
    index: int
    timestamp_s: float
    actor_track: str
    measurements: tuple[Measurement, ...]
    warnings: tuple[str, ...] = ()
    discontinuity: bool = False

    def __post_init__(self):
        if not self.source_id or not self.half or not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("Observation requires source/half and finite nonnegative timestamp seconds")
        if len({m.track for m in self.measurements}) != len(self.measurements):
            raise ValueError("Duplicate source tracks in one observation")

    @property
    def actor(self) -> Measurement | None:
        return next((m for m in self.measurements if m.track == self.actor_track), None)


@dataclass
class WorldInput:
    x: np.ndarray
    mask: np.ndarray
    actor_xy: np.ndarray
    last_xy: np.ndarray
    identities: dict[int, str]

    def tensors(self):
        import torch
        return {key: torch.from_numpy(getattr(self, key)[None]) for key in (
            "x", "mask", "actor_xy", "last_xy")}
