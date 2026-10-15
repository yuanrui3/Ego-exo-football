"""Fixed compact Transformer with one shared actor-teammate scoring head."""

import math

import torch
from torch import nn

ARCHITECTURE = {
    "features": 8,
    "time_steps": 3,
    "entities": 22,
    "d_model": 128,
    "layers": 3,
    "attention_heads": 4,
    "feedforward_dim": 256,
    "dropout": 0.1,
    "scoring_head": "259 -> 128 -> 1, ReLU, dropout 0.1, shared over candidates",
    "relative_features": "dx/52.5, dy/34, Euclidean distance / pitch half-diagonal",
    "missing_candidates": "finite logit -10000; absent tokens masked as attention keys",
}


class EntityTransformerReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_projection = nn.Linear(8, 128)
        self.time_embedding = nn.Embedding(3, 128)
        self.entity_embedding = nn.Embedding(22, 128)
        self.relation_embedding = nn.Embedding(3, 128)
        self.register_buffer(
            "relations", torch.tensor([0] + [1] * 10 + [2] * 11), persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=3, enable_nested_tensor=False
        )
        # Encoder clones start identically unless their matrix weights are reset.
        for encoder_layer in self.encoder.layers:
            for parameter in encoder_layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.scoring_head = nn.Sequential(
            nn.Linear(259, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or tuple(features.shape[1:]) != (3, 22, 8):
            raise ValueError("Expected [batch, 3, 22, 8] features")
        time = self.time_embedding.weight.view(1, 3, 1, 128)
        slots = self.entity_embedding.weight.view(1, 1, 22, 128)
        relations = self.relation_embedding(self.relations).view(1, 1, 22, 128)
        projected = self.feature_projection(features) + time + slots + relations
        observed = features[..., 2].reshape(-1, 66).bool()
        contextual = self.encoder(
            projected.flatten(1, 2), src_key_padding_mask=~observed
        ).reshape(-1, 3, 22, 128)
        actor = contextual[:, -1, 0:1].expand(-1, 10, -1)
        teammates = contextual[:, -1, 1:11]
        relative = features[:, -1, 1:11, :2] - features[:, -1, 0:1, :2]
        scale = relative.new_tensor([52.5, 34.0])
        distance = torch.linalg.vector_norm(
            relative * scale, dim=-1, keepdim=True
        ) / math.hypot(52.5, 34.0)
        logits = self.scoring_head(
            torch.cat((actor, teammates, relative, distance), dim=-1)
        ).squeeze(-1)
        return logits.masked_fill(features[:, -1, 1:11, 2] == 0, -10000.0)
