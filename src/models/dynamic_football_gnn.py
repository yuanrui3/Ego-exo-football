"""Anonymous edge-conditioned GATv2 and matched temporal football encoders."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.belief_world_model import WorldBlock


def graph_geometry(x, mask, std=None, distance_only=False, static_edges=None):
    """All geometry comes from measured/remembered/inferred inputs, never labels."""
    xy = x[..., :2] * x.new_tensor([52.5, 34.])
    delta = xy[:, :, None] - xy[:, None, :]
    velocity = (x[:, :, None, 2:4] - x[:, None, :, 2:4]) * 12
    distance = delta.norm(dim=-1)
    same = (x[:, :, None, 7] == x[:, None, :, 7]).float()
    actor_link = (x[:, :, None, 9] > .5) | (x[:, None, :, 9] > .5)
    actor = (xy * x[..., 9:10]).sum(1) / x[..., 9].sum(1).clamp_min(1)[:, None]
    lane = xy - actor[:, None]
    denominator = lane.square().sum(-1).clamp_min(1e-6)
    # Query i is the candidate; key j is a possible defender.
    u = ((xy[:, None] - actor[:, None, None]) * lane[:, :, None]).sum(-1) / denominator[:, :, None]
    feet = actor[:, None, None] + u.clamp(0, 1)[..., None] * lane[:, :, None]
    lane_distance = (xy[:, None] - feet).norm(dim=-1)
    interior = (u * denominator.sqrt()[:, :, None] >= 2) & (
        (1 - u) * denominator.sqrt()[:, :, None] >= 2)
    opponent_lane = (x[:, :, None, 7] > .5) & (x[:, None, :, 7] < -.5) & interior
    if std is None:
        uncertainty = (1.5 + x[..., 6] * 8 * 1.1)[..., None].expand(-1, -1, 2)
    else:
        uncertainty = std
    variance = uncertainty.square().sum(-1)
    relational_std = (variance[:, :, None] + variance[:, None, :]).sqrt()
    edges = ((distance <= 20) | actor_link | (opponent_lane & (lane_distance <= 5)))
    identity = torch.eye(x.shape[1], device=x.device, dtype=torch.bool)[None]
    edges |= identity
    if static_edges is not None:
        edges = static_edges | identity
    edges &= mask[:, :, None] & mask[:, None, :]
    values = torch.cat((
        delta / 52.5, velocity / 12, distance[..., None] / 52.5, same[..., None],
        actor_link[..., None].float(), x[:, :, None, 4:5].expand(-1, -1, x.shape[1], -1),
        x[:, None, :, 4:5].expand(-1, x.shape[1], -1, -1),
        (x[:, :, None, 6:7] + x[:, None, :, 6:7]),
        relational_std[..., None] / 10,
        lane_distance[..., None] / 52.5, opponent_lane[..., None].float(),
    ), -1)
    if distance_only:
        values = values * 0
        values[..., 4] = distance / 52.5
        edges = (distance <= 20) | identity
        edges &= mask[:, :, None] & mask[:, None, :]
    return values, edges


class EdgeGATv2(nn.Module):
    """Dynamic attention: score(a^T LeakyReLU(Wq hi + Wk hj + We eij))."""

    def __init__(self, width, heads=4, dropout=.1):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.query, self.key, self.value = (nn.Linear(width, width) for _ in range(3))
        self.edge = nn.Linear(13, width)
        self.attention = nn.Parameter(torch.empty(heads, self.dim))
        nn.init.xavier_uniform_(self.attention)
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width))
        self.dropout = dropout

    def forward(self, z, edge_features, edges, valid):
        b, n, _ = z.shape
        normalized = self.norm(z)
        query = self.query(normalized).view(b, n, self.heads, self.dim)
        key = self.key(normalized).view(b, n, self.heads, self.dim)
        value = self.value(normalized).view(b, n, self.heads, self.dim)
        edge = self.edge(edge_features).view(b, n, n, self.heads, self.dim)
        score = (F.leaky_relu(query[:, :, None] + key[:, None] + edge, .2) * self.attention).sum(-1)
        # An invalid query gets a harmless self edge; its output is explicitly zeroed.
        diagonal = torch.eye(n, dtype=torch.bool, device=z.device)[None]
        safe = edges | (~valid[:, :, None] & diagonal)
        score = score.masked_fill(~safe[..., None], -1e4)
        weights = F.dropout(score.softmax(2), p=self.dropout, training=self.training)
        message = torch.einsum("bijh,bjhd->bihd", weights, value).flatten(-2)
        z = z + self.output(message)
        return (z + self.ff(z)).masked_fill(~valid[..., None], 0)


class FootballEncoder(nn.Module):
    def __init__(self, family="G3", width=128, layers=3, dropout=.1, static=False, distance_only=False):
        super().__init__()
        self.family, self.width = family, width
        self.static, self.distance_only = static, distance_only
        self.embed = nn.Linear(10, width)
        if family in ("G1", "independent"):
            self.blocks = nn.ModuleList([WorldBlock(width, 4, dropout, family != "independent")
                                         for _ in range(layers)])
            self.temporal = None
        else:
            self.blocks = nn.ModuleList([EdgeGATv2(width, 4, dropout) for _ in range(layers)])
            self.temporal = (nn.Sequential(nn.Linear(width, width * 3), nn.GELU(), nn.Linear(width * 3, width))
                             if family == "G2" else nn.GRU(width, width, batch_first=True))
        self.final = nn.LayerNorm(width)

    def forward(self, x, mask, std=None):
        x = x.masked_fill(~mask[..., None], 0)
        b, t, n, _ = x.shape
        z = self.embed(x).masked_fill(~mask[..., None], 0)
        if self.family in ("G1", "independent"):
            frequencies = torch.exp(torch.arange(0, self.width, 2, device=x.device) * (-math.log(10000) / self.width))
            phase = (torch.arange(t, device=x.device) * .2)[:, None] * frequencies
            z = z + torch.stack((phase.sin(), phase.cos()), -1).flatten(1)[None, :, None]
            z = z.masked_fill(~mask[..., None], 0)
            for block in self.blocks:
                z = block(z, x, mask)
            tokens = z[:, -1]
        else:
            if self.family == "G2":
                # Spatial-only control consumes the same history through a fixed summary.
                z = (z * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[..., None]
                features, edges = graph_geometry(x[:, -1], mask[:, -1], std, self.distance_only)
                for block in self.blocks:
                    z = block(z, features, edges, mask[:, -1])
                tokens = self.temporal(z)
            else:
                fixed = None
                if self.static:
                    # First observed state; no future-time edges in earlier graph steps.
                    _, fixed = graph_geometry(x[:, 0], mask[:, 0], distance_only=self.distance_only)
                flat = x.reshape(b * t, n, 10)
                flat_mask = mask.reshape(b * t, n)
                all_std = std[:, None].expand(-1, t, -1, -1).reshape(b * t, n, 2) if std is not None else None
                if self.family != "G4" and all_std is None:
                    all_std = flat.new_full((b * t, n, 2), 1.5)
                fixed_batch = fixed[:, None].expand(-1, t, -1, -1).reshape(b * t, n, n) if fixed is not None else None
                features, edges = graph_geometry(flat, flat_mask, all_std, self.distance_only, fixed_batch)
                spatial = z.reshape(b * t, n, self.width)
                for block in self.blocks:
                    spatial = block(spatial, features, edges, flat_mask)
                temporal = spatial.reshape(b, t, n, self.width).transpose(1, 2).reshape(b * n, t, self.width)
                tokens = self.temporal(temporal)[0][:, -1].reshape(b, n, self.width)
        return self.final(tokens).masked_fill(~mask[:, -1, :, None], 0)


class GraphWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config["width"]
        self.encoder = FootballEncoder(config["family"], width, config["layers"],
                                       static=config.get("static", False), distance_only=config.get("distance_only", False))
        self.position = nn.Linear(width, 4)
        self.occupancy = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, 294 * 2))
        nn.init.zeros_(self.position.weight)
        nn.init.zeros_(self.position.bias)

    def forward(self, x, mask, actor_xy, last_xy):
        tokens = self.encoder(x, mask)
        valid = mask[:, -1]
        pooled = (tokens * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        scene = torch.cat((pooled, actor_xy / actor_xy.new_tensor([52.5, 34.])), -1)
        position = self.position(tokens).float()
        age = x[:, -1, :, 6] * 8
        # Age-dependent causal uncertainty prior; learned residual, no oracle ages.
        prior_log_std = (1.5 + 1.1 * age).log()
        occ = self.occupancy(scene).float()
        return {"mean": last_xy + position[..., :2] * 5,
                "log_std": (position[..., 2:] + prior_log_std[..., None]).clamp(-1.5, 3),
                "occupancy_logits": occ[:, :294], "anonymous_defender_logits": occ[:, 294:]}


class GraphDecisionModel(nn.Module):
    def __init__(self, config, task="risk"):
        super().__init__()
        self.config, self.task = config, task
        family = config["family"]
        encoder_family = "G3" if family in ("G4", "G5", "G6") else family
        width = config["width"]
        self.encoder = FootballEncoder(encoder_family, width, config["layers"],
                                       static=config.get("static", False), distance_only=config.get("distance_only", False))
        self.belief = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.LayerNorm(width))
        self.occupancy = nn.Sequential(nn.Linear(294, width), nn.GELU(), nn.LayerNorm(width))
        self.head = nn.Sequential(nn.Linear(width * 4 + 3, width), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(width, 7 if task == "risk" else 1))

    def forward(self, x, mask, candidate_slot, belief, belief_mask, occupancy):
        # Hidden memory is replaceable: no second path through the visible encoder.
        visible_now = x[:, -1, :, 4] > .5
        allowed = mask & visible_now[:, None]
        tokens = self.encoder(x, allowed)
        b, n, w = tokens.shape
        actor = (tokens * x[:, -1, :, 9:10]).sum(1) / x[:, -1, :, 9].sum(1).clamp_min(1)[:, None]
        candidate = tokens[torch.arange(b, device=x.device), candidate_slot]
        memory = self.belief(belief).masked_fill(~belief_mask[..., None], 0)
        relative = x[:, -1, :, :2] * x.new_tensor([52.5, 34.])
        q = relative[torch.arange(b, device=x.device), candidate_slot]
        centers = belief[..., :2] * x.new_tensor([52.5, 34.])
        u = (centers * q[:, None]).sum(-1) / q.square().sum(-1).clamp_min(1e-6)[:, None]
        nearest = u.clamp(0, 1)[..., None] * q[:, None]
        d2 = (centers - nearest).square().sum(-1)
        uncertainty = (belief[..., 2:4].square().sum(-1) * 100).clamp_min(.01)
        weights = torch.exp(-d2 / (2 * uncertainty)).masked_fill(~belief_mask, 0)
        context = (memory * weights[..., None]).sum(1) / weights.sum(1).clamp_min(1e-5)[:, None]
        geometric = torch.cat((q / q.new_tensor([52.5, 34.]), q.norm(dim=-1, keepdim=True) / 105), -1)
        output = self.head(torch.cat((actor, candidate, context, self.occupancy(occupancy), geometric), -1)).float()
        if self.task == "risk":
            output = torch.cat((output[:, :6], F.softplus(output[:, 6:])), -1)
        return output
