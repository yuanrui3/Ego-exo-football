"""Shared metric-feature ranker with fitting-only masked normalization."""

import torch
from torch import nn


def feature_statistics(values, valid):
    count = valid.sum((0, 1)).clamp_min(1)
    mean = (values * valid).sum((0, 1)) / count
    variance = ((values - mean).square() * valid).sum((0, 1)) / count
    return mean, variance.sqrt().clamp_min(1e-6)


class ReceiverForecastRanker(nn.Module):
    def __init__(self, mean, std, width=128):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
        self.network = nn.Sequential(
            nn.Linear(2 * len(mean), width * 2), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(width * 2, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(.1), nn.Linear(width, 1),
        )

    def forward(self, values, valid, candidates):
        x = ((values - self.mean) / self.std).masked_fill(~valid, 0)
        logits = self.network(torch.cat((x, valid.float()), -1)).squeeze(-1)
        return logits.masked_fill(~candidates, -10000)
