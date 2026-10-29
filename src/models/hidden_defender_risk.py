"""Capacity-matched candidate risk heads with explicit replaceable hidden belief."""

import torch
from torch import nn
from torch.nn import functional as F

from src.models.partial_observation_student import RelativeSocialLayer


class HiddenDefenderRisk(nn.Module):
    def __init__(self, width=96, layers=2, dropout=.1):
        super().__init__()
        self.width = width
        self.temporal = nn.GRU(10, width, batch_first=True)
        self.social = nn.ModuleList(RelativeSocialLayer(width, 4, dropout) for _ in range(layers))
        self.hidden = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.LayerNorm(width),
                                    nn.Linear(width, width), nn.GELU())
        self.occupancy = nn.Sequential(nn.Linear(294, width), nn.GELU(), nn.LayerNorm(width))
        self.query = nn.Sequential(nn.Linear(6, width), nn.GELU())
        self.head = nn.Sequential(nn.Linear(width*5, width*2), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width*2, 7))

    def forward(self, x, mask, candidate_slot, geometry, hidden, hidden_mask, occupancy):
        b, t, p, _ = x.shape
        clean = x.masked_fill(~mask[..., None], 0)
        _, encoded = self.temporal(clean.transpose(1, 2).reshape(b*p, t, 10))
        tokens = encoded[-1].reshape(b, p, self.width).masked_fill(~mask[:, -1, :, None], 0)
        for layer in self.social:
            tokens = layer(tokens, clean[:, -1], mask[:, -1])
        actor_mask = clean[:, -1, :, 9:10]*mask[:, -1, :, None]
        actor = (tokens*actor_mask).sum(1)/actor_mask.sum(1).clamp_min(1)
        candidate = tokens[torch.arange(b, device=x.device), candidate_slot]
        h = self.hidden(hidden).masked_fill(~hidden_mask[..., None], 0)
        h = h.sum(1)/hidden_mask.sum(1, keepdim=True).clamp_min(1)
        scene = self.occupancy(occupancy)
        z = self.head(torch.cat((actor, candidate, self.query(geometry), h, scene), -1)).float()
        return {"logits": z[:, :6], "pressure": F.softplus(z[:, 6])*10}
