"""Permutation-equivariant player forecasting with optional social attention."""

import torch
from torch import nn
from torch.nn import functional as F

HORIZON_SECONDS = (0.4, 1.0, 2.0, 3.0)


def persistence(batch):
    return torch.zeros_like(batch["target"])


def constant_velocity(batch):
    times = batch["velocity"].new_tensor(HORIZON_SECONDS)
    return batch["velocity"][:, :, None] * times[None, None, :, None]


def masked_loss(prediction, target, mask, log_scale=None):
    if not mask.any():
        raise ValueError("No valid regression targets")
    if log_scale is None:
        error = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none").mean(-1)
    else:
        # Isotropic Gaussian NLL in metric coordinates.
        scale = log_scale.float().clamp(-2.3, 3.0)
        error = .5 * ((prediction.float() - target.float()) ** 2).sum(-1) * torch.exp(-2 * scale)
        error = error + 2 * scale
    return error[mask].mean()


class SocialBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.geometry = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, heads))
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 2, width), nn.Dropout(dropout),
        )
        self.dropout = dropout

    def forward(self, x, current, team, valid):
        b, players, width = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, players, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        relative = current[:, :, None] - current[:, None, :]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        same_team = (team[:, :, None] == team[:, None, :]).unsqueeze(-1)
        geometry = torch.cat([relative / 30, distance / 30, same_team.to(relative.dtype)], -1)
        bias = self.geometry(geometry).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x + self.output(attended.transpose(1, 2).reshape(b, players, width))
        return (x + self.ff(self.norm2(x))).masked_fill(~valid[..., None], 0)


class TrajectoryForecaster(nn.Module):
    def __init__(self, width=128, layers=3, heads=4, dropout=.1, social=True,
                 history_frames=50, probabilistic=False):
        super().__init__()
        if width % heads:
            raise ValueError("Embedding width must divide heads")
        self.history_frames, self.probabilistic = history_frames, probabilistic
        self.temporal = nn.GRU(5, width, batch_first=True)
        self.position = nn.Linear(2, width)
        self.team = nn.Embedding(3, width, padding_idx=0)
        self.role = nn.Embedding(2, width)
        self.velocity = nn.Linear(2, width)
        self.norm = nn.LayerNorm(width)
        self.social = social
        if social:
            self.blocks = nn.ModuleList([SocialBlock(width, heads, dropout) for _ in range(layers)])
        else:
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                              nn.Dropout(dropout), nn.Linear(width * 2, width))
                for _ in range(layers)
            ])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                  nn.GELU(), nn.Linear(width, 12 if probabilistic else 8))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch):
        history = batch["history"][:, :, -(self.history_frames + 1):].clone()
        # The first retained step must not encode a velocity from outside the chosen history.
        history[:, :, 0, 2:4] = 0
        b, n, t, features = history.shape
        _, hidden = self.temporal(history.reshape(b * n, t, features))
        velocity = batch["velocity"] if self.history_frames else torch.zeros_like(batch["velocity"])
        x = hidden[-1].reshape(b, n, -1)
        x = self.norm(x + self.position(batch["current"] / 60)
                      + self.team(batch["team"]) + self.role(batch["role"])
                      + self.velocity(velocity / 12))
        valid = batch["player_mask"]
        x = x.masked_fill(~valid[..., None], 0)
        for block in self.blocks:
            x = block(x, batch["current"], batch["team"], valid) if self.social else (
                x + block(x)
            ).masked_fill(~valid[..., None], 0)
        output = self.head(x).float()
        times = velocity.new_tensor(HORIZON_SECONDS)
        mean = output[..., :8].view(b, n, 4, 2) + velocity[:, :, None] * times[None, None, :, None]
        scale = output[..., 8:12].clamp(-2.3, 3.0) if self.probabilistic else None
        return mean, scale
