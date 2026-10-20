"""Player-facing sequential inference accepts observations, never full-state truth."""

from __future__ import annotations

import time
import dataclasses

from src.backbone import SCHEMA_VERSION
from src.backbone.assessment import GaussianAssessor
from src.backbone.actions import ActionModelSession
from src.backbone.contracts import ActionOutput, ActionOutcome, BallState, Observation, RestartContext
from src.backbone.memory import CausalMemory
from src.backbone.models import ModelSession


class ReplayEngine:
    def __init__(self, models: ModelSession, expiry_s=6., fallback="last_seen", samples=128,
                 action_model: ActionModelSession | None = None):
        self.memory = CausalMemory(models, expiry_s, fallback)
        self.assessor = GaussianAssessor(samples)
        self.action_model = action_model

    def reset(self):
        self.memory.reset()

    def step(self, observation: Observation, *, include_passing_assessment: bool = True,
             ball_state: BallState | None = None,
             action: ActionOutput | None = None,
             restart_context: RestartContext | None = None,
             outcome: ActionOutcome | None = None):
        if not isinstance(observation, Observation):
            raise TypeError("ReplayEngine only accepts an Observation, never a privileged source frame")
        start = time.perf_counter()
        belief = self.memory.step(observation)
        sample_id = f"{observation.source_id}:{observation.actor_track}:{observation.index}"
        assessment = self.assessor.assess(belief, sample_id) if include_passing_assessment else None
        action_inference = None
        if action is not None:
            action_output = action
        elif self.action_model is not None:
            action_inference = self.action_model.predict(self.memory.current_input)
            action_output = action_inference.output
        else:
            action_output = ActionOutput()
        current_ball = ball_state or BallState()
        restart = restart_context or RestartContext()
        current_outcome = outcome or ActionOutcome()
        result = {"schema_version": SCHEMA_VERSION, "sample_id": sample_id,
                "source_id": observation.source_id, "half": observation.half,
                "frame_index": observation.index, "timestamp_s": observation.timestamp_s,
                "actor": observation.actor_track, "units": "metres",
                "belief": belief,
                "action": {"schema_version": "football.action-output.0.1",
                           **dataclasses.asdict(action_output)},
                "ball_state": dataclasses.asdict(current_ball),
                "restart_context": dataclasses.asdict(restart),
                "outcome": dataclasses.asdict(current_outcome),
                "modules": {"passing": ({"status": assessment["status"], "lane_assessment": assessment,
                                         **({"reason": assessment["unavailable_reason"]}
                                            if assessment["status"] == "UNAVAILABLE" else {})}
                                         if assessment is not None else
                                         {"status": "UNSUPPORTED", "reason": "Passing assessment disabled or not configured"})},
                "assessment": assessment,
                "latency_ms": (time.perf_counter() - start) * 1000}
        if action_inference is not None:
            result["action_inference"] = action_inference.to_dict()
        return result


def replay_prefix(engine: ReplayEngine, observations, through: int):
    """Seeking replays preceding observations; it never initializes from full state."""
    if through < 0 or through >= len(observations):
        raise IndexError("Seek target outside observation sequence")
    engine.reset()
    result = None
    for observation in observations[:through + 1]:
        result = engine.step(observation)
    return result
