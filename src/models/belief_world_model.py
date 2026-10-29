"""Masked spatiotemporal world model and explicit belief-conditioned decision heads."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.partial_observation_student import PartialObservationStudent, RelativeSocialLayer


class WorldBlock(nn.Module):
    def __init__(self, width, heads, dropout, social=True):
        super().__init__()
        self.social = RelativeSocialLayer(width, heads, dropout) if social else None
        self.temporal = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, z, features, valid):
        b, t, p, w = z.shape
        if self.social is not None:
            z = self.social(z.reshape(b*t, p, w), features.reshape(b*t, p, 10),
                            valid.reshape(b*t, p)).reshape(b, t, p, w)
        sequence = z.transpose(1, 2).reshape(b*p, t, w)
        normalized = self.norm(sequence)
        causal = torch.ones(t, t, device=z.device, dtype=torch.bool).triu(1)
        known = valid.transpose(1, 2).reshape(b*p, t)
        blocked = causal[None] | ~known[:, None, :]
        empty_queries = blocked.all(-1)
        diagonal = torch.eye(t, device=z.device, dtype=torch.bool)[None]
        blocked &= ~(empty_queries[..., None] & diagonal)
        blocked = blocked.repeat_interleave(self.temporal.num_heads, dim=0)
        attended = self.temporal(normalized, normalized, normalized, attn_mask=blocked,
                                 need_weights=False)[0]
        sequence = sequence + attended
        sequence = sequence + self.ff(sequence)
        return sequence.reshape(b, p, t, w).transpose(1, 2).masked_fill(~valid[..., None], 0)


class BeliefWorldModel(nn.Module):
    def __init__(self, width=192, layers=4, heads=6, dropout=.1, social=True):
        super().__init__()
        self.embed = nn.Linear(10, width)
        self.blocks = nn.ModuleList([WorldBlock(width, heads, dropout, social) for _ in range(layers)])
        self.occupancy_head = nn.Sequential(nn.LayerNorm(width + 2), nn.Linear(width + 2, width),
                                            nn.GELU(), nn.Linear(width, 294))
        self.position_head = nn.Linear(width, 4)
        self.count_head = nn.Linear(width + 2, 1)
        self.future_head = nn.Linear(width, 16)
        self.register_buffer("coordinate_scale", torch.tensor([52.5, 34.]))

    def forward(self, x, mask, actor_xy, last_xy):
        b, t, p, _ = x.shape
        time = torch.arange(t, device=x.device, dtype=x.dtype) * .2
        frequencies = torch.exp(torch.arange(0, self.embed.out_features, 2, device=x.device)
                                * (-math.log(10000) / self.embed.out_features))
        phase = time[:, None] * frequencies
        positional = torch.stack((phase.sin(), phase.cos()), -1).flatten(1)
        z = self.embed(x) + positional[None, :, None]
        z = z.masked_fill(~mask[..., None], 0)
        for block in self.blocks:
            z = block(z, x, mask)
        valid = mask[:, -1]
        tokens = z[:, -1]
        pooled = (tokens * valid[..., None]).sum(1) / valid.sum(-1, keepdim=True).clamp_min(1)
        scene = torch.cat((pooled, actor_xy / self.coordinate_scale), -1)
        position = self.position_head(tokens).float()
        future = self.future_head(tokens).float().reshape(b, p, 4, 4)
        return {
            "occupancy": self.occupancy_head(scene).float(),
            "hidden_count": F.softplus(self.count_head(scene).float()).squeeze(-1),
            "position_mean": last_xy + position[..., :2] * 10,
            "position_log_std": position[..., 2:].clamp(-1.5, 3),
            "future_mean": last_xy[:, :, None] + future[..., :2] * 10,
            "future_log_std": future[..., 2:].clamp(-1.5, 3.5),
        }


class BeliefDecisionModel(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.encoder = PartialObservationStudent(width=width, layers=2, heads=4, belief=False)
        self.belief_encoder = nn.Sequential(nn.Linear(294 + 2, width), nn.GELU(),
                                             nn.LayerNorm(width))
        self.action = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 7))
        self.receiver = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
        self.unobserved = nn.Linear(width * 2, 1)
        self.register_buffer("coordinate_scale", torch.tensor([52.5, 34.]))

    def forward(self, x, mask, visible, actor_xy, belief):
        base = self.encoder(x, mask, visible)
        tokens = base["tokens"]
        weights = x[:, -1, :, 9:10] * mask[:, -1, :, None]
        actor = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
        context = self.belief_encoder(torch.cat((belief, actor_xy / self.coordinate_scale), -1))
        actor_context = torch.cat((actor, context), -1)
        receiver_context = torch.cat((tokens, context[:, None].expand_as(tokens)), -1)
        scores = self.receiver(receiver_context).squeeze(-1)
        # Only currently visible candidates can be named; memory is context, not an invisible receiver label.
        candidates = visible[:, -1] & (x[:, -1, :, 7] > .5) & (x[:, -1, :, 9] < .5)
        scores = scores.masked_fill(~candidates, -1e4)
        return {
            "action": self.action(actor_context),
            "receiver": torch.cat((scores, self.unobserved(actor_context)), -1),
        }


def gaussian_nll(mean, log_std, target, mask):
    terms = .5 * ((target - mean) * (-log_std).exp()) ** 2 + log_std + .5 * math.log(2 * math.pi)
    return (terms.sum(-1) * mask).sum() / mask.sum().clamp_min(1)


def world_loss(output, batch):
    occupancy = F.binary_cross_entropy_with_logits(output["occupancy"], batch["occupancy"])
    position = gaussian_nll(output["position_mean"], output["position_log_std"],
                            batch["position"], batch["position_mask"])
    future = gaussian_nll(output["future_mean"], output["future_log_std"],
                          batch["future"], batch["future_mask"])
    count = F.smooth_l1_loss(output["hidden_count"], batch["hidden_count"])
    return occupancy + .02 * count + .04 * position + .04 * future
