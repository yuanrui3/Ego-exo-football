"""Checkpoint-backed event-conditioned action inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from src.backbone.contracts import ActionOutput, WorldInput
from src.models.partial_observation_student import PartialObservationStudent
from src.player_centric_dataset import ACTION_LABELS
from src.trajectory_dataset import file_digest


@dataclass(frozen=True)
class ActionInference:
    output: ActionOutput
    scores: dict[str, float] | None
    status: str
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "output": asdict(self.output),
            "scores": self.scores,
            "score_semantics": "Uncalibrated seven-way classifier softmax; not tactical value or a best-action score",
            "status": self.status,
            "reason": self.reason,
        }


class ActionModelSession:
    """Loads the selected partial-observation student once and accepts v0.2 WorldInput."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu"):
        self.checkpoint = checkpoint.resolve()
        self.device = torch.device(device)
        self.sha256 = file_digest(self.checkpoint)
        self.config_path = self.checkpoint.parent / "config.json"
        config_record = json.loads(self.config_path.read_text(encoding="utf-8"))
        if config_record.get("variant") != "S4" or config_record.get("fov_index") != 1:
            raise ValueError("Selected action checkpoint must be the S4 120-degree student")
        fitting_matches = config_record.get("fitting_matches")
        allowed_fitting_splits = {
            ("117092", "117093", "118576", "118578", "128057", "132831"),
            ("117092", "117093", "118576", "118578", "128057", "132831", "132877"),
        }
        if tuple(fitting_matches or ()) not in allowed_fitting_splits:
            raise ValueError("Selected action checkpoint fitting provenance is not a verified S4 split")
        config = config_record["config"]
        self.model = PartialObservationStudent(
            width=int(config["width"]), layers=int(config["layers"]),
            heads=int(config["heads"]), dropout=float(config["dropout"]), belief=True,
        ).to(self.device)
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.load_count = 1
        self.provenance = {
            "checkpoint": str(self.checkpoint),
            "sha256": self.sha256,
            "config_path": str(self.config_path),
            "config_sha256": file_digest(self.config_path),
            "variant": config_record["variant"],
            "seed": int(config["seed"]),
            "fitting_matches": list(fitting_matches),
            "classes": list(ACTION_LABELS),
        }

    @staticmethod
    def _unsupported(reason: str) -> ActionInference:
        output = ActionOutput(
            support_status="unsupported",
            reason=reason,
        )
        return ActionInference(output, None, "unsupported", reason)

    @torch.inference_mode()
    def predict(self, inputs: WorldInput | None) -> ActionInference:
        if inputs is None:
            return self._unsupported("The causal 11-sample player observation is not ready")
        x = np.asarray(inputs.x, dtype=np.float32)
        mask = np.asarray(inputs.mask, dtype=bool)
        if x.shape != (11, 22, 10) or mask.shape != (11, 22):
            return self._unsupported(
                f"Incompatible input shape: expected x (11, 22, 10), mask (11, 22); got {x.shape}, {mask.shape}"
            )
        if not np.isfinite(x).all():
            return self._unsupported("Action input contains non-finite features")
        visible_features = x[..., 4] > 0.5
        if np.any(visible_features & ~mask):
            return self._unsupported("Visibility mask includes an entity absent from causal memory")
        if not np.array_equal(mask, x[..., 5] > 0.5):
            return self._unsupported("Entity mask does not match the ever-seen causal-memory feature")
        actor_flags = (x[..., 9] > 0.5) & mask
        if not np.all(actor_flags.sum(axis=1) == 1):
            return self._unsupported("Each history sample must contain exactly one selected actor")
        actor_slots = actor_flags.argmax(axis=1)
        if not np.all(actor_slots == actor_slots[-1]) or not np.all(x[:, actor_slots[-1], 4] > 0.5):
            return self._unsupported("Selected actor is not continuously visible in the supplied history")
        if np.any(x[..., 6] < 0) or np.any(x[..., 6] > 0.25001):
            return self._unsupported("Input age normalization does not match v0.2's 8-second scale")
        model_x = x.copy()
        model_x[..., 6] = np.clip(model_x[..., 6] * (8.0 / 5.0), 0.0, 1.0)
        visible = visible_features
        torch_x = torch.from_numpy(model_x[None]).to(self.device)
        torch_mask = torch.from_numpy(mask[None]).to(self.device)
        torch_visible = torch.from_numpy(visible[None]).to(self.device)
        result = self.model(torch_x, torch_mask, torch_visible, memory_mode="memory")
        logits = result["action"].float()[0]
        if logits.shape != (len(ACTION_LABELS),) or not torch.isfinite(logits).all():
            raise FloatingPointError("Action model returned invalid logits")
        probabilities = logits.softmax(-1).cpu().numpy()
        selected = int(probabilities.argmax())
        label = ACTION_LABELS[selected]
        reason = (
            "Event-conditioned demonstration classifier only; not a continuous detector or best-action recommender. "
            "Scores are uncalibrated."
        )
        output = ActionOutput(
            action_type=label,
            confidence=None,
            support_status="partial",
            reason=reason,
        )
        return ActionInference(
            output,
            {name: float(probability) for name, probability in zip(ACTION_LABELS, probabilities)},
            "supported",
        )
