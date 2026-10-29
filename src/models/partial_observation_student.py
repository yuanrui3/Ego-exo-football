"""Permutation-equivariant global teacher and partial-observation student."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RelativeSocialLayer(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.heads, self.head_width = width, heads, width // heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.geometry = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, heads))
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 2, width), nn.Dropout(dropout),
        )
        self.dropout = dropout

    def forward(self, tokens: torch.Tensor, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, players, _ = tokens.shape
        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized).view(batch, players, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        relative = x[:, :, None, :2] - x[:, None, :, :2]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        same_team = (x[:, :, None, 7:8] == x[:, None, :, 7:8]).to(x.dtype)
        geometry = torch.cat((relative, distance, same_team), dim=-1)
        bias = self.geometry(geometry).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        tokens = tokens + self.out(attended.transpose(1, 2).reshape(batch, players, self.width))
        tokens = tokens + self.ff(self.norm2(tokens))
        return tokens.masked_fill(~valid[..., None], 0)


class PartialObservationStudent(nn.Module):
    """Shared-entity temporal/social network; it has no player-slot embeddings."""

    def __init__(self, width: int = 96, layers: int = 2, heads: int = 4,
                 dropout: float = .1, occupancy_cells: int = 294, belief: bool = True):
        super().__init__()
        self.width = width
        self.temporal = nn.GRU(10, width, batch_first=True)
        self.current = nn.Linear(10, width)
        self.norm = nn.LayerNorm(width)
        self.social = nn.ModuleList(
            [RelativeSocialLayer(width, heads, dropout) for _ in range(layers)]
        )
        self.action_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 7))
        self.receiver_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.unobserved_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.belief = belief
        self.occupancy_head = (
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                          nn.Linear(width, occupancy_cells)) if belief else None
        )

    def forward(self, x: torch.Tensor, entity_mask: torch.Tensor,
                visible: torch.Tensor, *, memory_mode: str = "memory"):
        if x.ndim != 4 or x.shape[-2:] != (22, 10):
            raise ValueError("Expected features [batch,time,22,10]")
        if entity_mask.shape != x.shape[:-1] or visible.shape != entity_mask.shape:
            raise ValueError("Entity and visibility masks must match [batch,time,22]")
        if memory_mode not in ("current", "visible_history", "memory"):
            raise ValueError(f"Unknown memory mode {memory_mode}")
        decision_visible = visible[:, -1]
        if memory_mode == "current":
            usable = decision_visible[:, None, :]
            sequence = x[:, -1:]
        elif memory_mode == "visible_history":
            usable = visible & decision_visible[:, None, :]
            sequence = x.masked_fill(~usable[..., None], 0)
        else:
            usable = entity_mask
            sequence = x.masked_fill(~usable[..., None], 0)
        b, time, players, dims = sequence.shape
        _, hidden = self.temporal(sequence.permute(0, 2, 1, 3).reshape(b * players, time, dims))
        tokens = hidden[-1].reshape(b, players, self.width)
        current_features = sequence[:, -1]
        valid = usable[:, -1]
        tokens = self.norm(tokens + self.current(current_features))
        tokens = tokens.masked_fill(~valid[..., None], 0)
        for layer in self.social:
            tokens = layer(tokens, current_features, valid)
        actor_weights = current_features[..., 9:10] * valid[..., None]
        actor = (tokens * actor_weights).sum(1) / actor_weights.sum(1).clamp_min(1)
        action_logits = self.action_head(actor)
        receiver_scores = self.receiver_head(tokens).squeeze(-1)
        unobserved = self.unobserved_head(actor)
        teammate = (current_features[..., 7] > .5) & (current_features[..., 9] < .5)
        receiver_scores = receiver_scores.masked_fill(~(teammate & valid), -1e4)
        receiver_logits = torch.cat((receiver_scores, unobserved), dim=-1)
        occupancy_logits = None
        if self.occupancy_head is not None:
            pooled = (tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            occupancy_logits = self.occupancy_head(pooled)
        return {
            "action": action_logits,
            "receiver": receiver_logits,
            "occupancy": occupancy_logits,
            "tokens": tokens,
            "valid": valid,
        }
