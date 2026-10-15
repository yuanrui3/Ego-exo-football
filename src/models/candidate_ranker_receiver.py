"""Football-specific candidate features and a shared contextual ranking network."""

import torch
from torch import nn

from src.models.entity_transformer_receiver import EntityTransformerReceiver

FEATURE_NAMES = tuple(
    f"{name}_{time}" for time in ("minus_0.4", "minus_0.2", "current")
    for name in ("relative_dx", "relative_dy", "distance")
) + tuple(
    f"{entity}_{name}_{interval}"
    for entity in ("candidate", "actor")
    for interval in ("early", "late", "full")
    for name in ("displacement_x", "displacement_y", "velocity_x", "velocity_y")
) + (
    "distance_change_early", "distance_change_late", "distance_change_full",
    "candidate_nearest_opponent", "actor_nearest_opponent",
    "opponents_within_3m", "opponents_within_5m", "opponents_within_10m",
    "distance_rank", "candidate_goalkeeper",
) + tuple(
    f"{entity}_observed_{time}" for entity in ("candidate", "actor")
    for time in ("minus_0.4", "minus_0.2", "current")
)
CONTINUOUS_COUNT = FEATURE_NAMES.index("candidate_goalkeeper")


def candidate_features(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,10,F] values and validity; coordinates/displacements are metres."""
    if states.ndim != 4 or tuple(states.shape[1:]) != (3, 22, 8):
        raise ValueError("Expected [batch,3,22,8]")
    xy = states[..., :2] * states.new_tensor([52.5, 34.0])
    seen = states[..., 2].bool()
    actor = xy[:, :, :1].expand(-1, -1, 10, -1)
    candidates = xy[:, :, 1:11]
    actor_seen = seen[:, :, :1].expand(-1, -1, 10)
    candidate_seen = seen[:, :, 1:11]
    pair_seen = actor_seen & candidate_seen
    relative = candidates - actor
    distance = torch.linalg.vector_norm(relative, dim=-1)
    values, valid = [], []

    def add(value, mask):
        values.append(value)
        valid.append(mask)

    for t in range(3):
        for value in (relative[:, t, :, 0], relative[:, t, :, 1], distance[:, t]):
            add(value, pair_seen[:, t])
    for positions, observed in ((candidates, candidate_seen), (actor, actor_seen)):
        for start, end in ((0, 1), (1, 2), (0, 2)):
            delta = positions[:, end] - positions[:, start]
            mask = observed[:, end] & observed[:, start]
            for value in (delta[..., 0], delta[..., 1],
                          delta[..., 0] / ((end - start) * 0.2),
                          delta[..., 1] / ((end - start) * 0.2)):
                add(value, mask)
    for start, end in ((0, 1), (1, 2), (0, 2)):
        add(distance[:, end] - distance[:, start], pair_seen[:, end] & pair_seen[:, start])
    opponent_seen = seen[:, -1, 11:]
    candidate_opponent = torch.cdist(candidates[:, -1], xy[:, -1, 11:])
    actor_opponent = torch.cdist(actor[:, -1, :1], xy[:, -1, 11:])
    has_opponent = opponent_seen.any(-1, keepdim=True).expand(-1, 10)
    add(candidate_opponent.masked_fill(~opponent_seen[:, None], float("inf")).amin(-1),
        candidate_seen[:, -1] & has_opponent)
    add(actor_opponent.masked_fill(~opponent_seen[:, None], float("inf")).amin(-1).expand(-1, 10),
        actor_seen[:, -1] & has_opponent)
    for radius in (3, 5, 10):
        add(((candidate_opponent <= radius) & opponent_seen[:, None]).sum(-1).float(),
            candidate_seen[:, -1] & has_opponent)
    distances = distance[:, -1].masked_fill(~pair_seen[:, -1], float("inf"))
    order = distances.argsort(dim=-1, stable=True)
    ranks = order.argsort(dim=-1).float() + 1
    add(ranks, pair_seen[:, -1])
    add(states[:, -1, 1:11, 6], candidate_seen[:, -1])
    for observed in (candidate_seen, actor_seen):
        for t in range(3):
            add(observed[:, t].float(), torch.ones_like(observed[:, t]))
    validity = torch.stack(valid, -1)
    result = torch.stack(values, -1).masked_fill(~validity, 0)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Non-finite candidate features")
    return result, validity


def fit_statistics(states: torch.Tensor) -> dict[str, torch.Tensor]:
    """Caller passes only the fitting partition, never development/validation."""
    values, valid = candidate_features(states)
    count = valid.sum((0, 1)).clamp_min(1)
    mean = values.sum((0, 1)) / count
    variance = ((values - mean).square() * valid).sum((0, 1)) / count
    std = variance.sqrt().clamp_min(1e-6)
    mean[CONTINUOUS_COUNT:] = 0
    std[CONTINUOUS_COUNT:] = 1
    xy = states[..., :2]
    observed = states[..., 2:3]
    n = observed.sum().clamp_min(1)
    scene_mean = (xy * observed).sum((0, 1, 2)) / n
    scene_std = (((xy - scene_mean).square() * observed).sum((0, 1, 2)) / n).sqrt().clamp_min(1e-6)
    return {"feature_mean": mean, "feature_std": std,
            "scene_mean": scene_mean, "scene_std": scene_std}


def prior_adjustment(logits: torch.Tensor, frequencies: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha not in (0.0, 0.25, 0.5):
        raise ValueError("Alpha must be 0, 0.25, or 0.5")
    if frequencies.shape != (10,) or not bool((frequencies > 0).all()):
        raise ValueError("All ten training target frequencies must be positive")
    return logits - alpha * frequencies.log()


class CandidateRankerReceiver(nn.Module):
    def __init__(self, statistics: dict[str, torch.Tensor]):
        super().__init__()
        for key, value in statistics.items():
            self.register_buffer(key, value.clone())
        self.scene = EntityTransformerReceiver()
        del self.scene.scoring_head
        self.candidate_mlp = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES) * 2, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.scoring_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features, validity = candidate_features(states)
        normalized = ((features - self.feature_mean) / self.feature_std).masked_fill(~validity, 0)
        engineered = self.candidate_mlp(torch.cat((normalized, validity.float()), -1))
        inputs = states.clone()
        inputs[..., :2] = ((states[..., :2] - self.scene_mean) / self.scene_std).masked_fill(
            states[..., 2:3] == 0, 0
        )
        scene = self.scene
        projected = (
            scene.feature_projection(inputs)
            + scene.time_embedding.weight.view(1, 3, 1, 128)
            + scene.entity_embedding.weight.view(1, 1, 22, 128)
            + scene.relation_embedding(scene.relations).view(1, 1, 22, 128)
        )
        observed = states[..., 2].bool().flatten(1, 2)
        context = scene.encoder(projected.flatten(1, 2), src_key_padding_mask=~observed)
        pooled = (context * observed[..., None]).sum(1) / observed.sum(1, keepdim=True)
        current = context.reshape(-1, 3, 22, 128)[:, -1]
        logits = self.scoring_head(torch.cat((
            engineered, current[:, 1:11], current[:, :1].expand(-1, 10, -1),
            pooled[:, None].expand(-1, 10, -1),
        ), -1)).squeeze(-1)
        return logits.masked_fill(states[:, -1, 1:11, 2] == 0, -10000.0)
