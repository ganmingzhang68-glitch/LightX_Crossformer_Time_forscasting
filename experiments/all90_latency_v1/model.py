"""130 dynamic histories + every added field's absolute history -> old128."""
from __future__ import annotations

import torch
from torch import nn
from vendor.crossformer import Crossformer, CrossformerConfig


class ForecastModel(nn.Module):
    def __init__(self, c, group, pre, params):
        super().__init__()
        g = c["groups"][group]
        self.group, self.horizon, self.indices = group, g["horizon"], list(range(128))
        config = CrossformerConfig.from_dict(dict(params, enc_in=128 + len(c["dynamic_added"]),
            seq_len=g["context"], pred_len=g["horizon"], use_revin=True, residual_forecast=True))
        self.backbone = Crossformer(config)
        self.category_positions = [c["added"].index(f) for f in c["categorical"]]
        self.scalar_positions = [i for i in range(90) if i not in self.category_positions]
        self.embeddings = nn.ModuleList([nn.Embedding(len(pre["categorical_vocabulary"][f]) + 2, 4, padding_idx=0)
                                         for f in c["categorical"]])
        with torch.no_grad():
            for embedding in self.embeddings:
                embedding.weight[1].zero_()  # unknown categories have a neutral, non-random fallback
        # All 90 added values have a known/missing mask, plus 3 historical source-quality channels.
        width = len(self.scalar_positions) + 4 * len(self.embeddings) + 93
        hidden = params["condition_hidden"]
        self.gru = nn.GRU(width, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, g["horizon"] * 128))
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, dynamic, conditions, aux):
        if conditions.shape[-1] != 90 or aux.shape[-1] != 93:
            raise ValueError("All 90 added fields and their masks are mandatory")
        base = self.backbone(dynamic)[..., :128]
        parts = [conditions[..., self.scalar_positions], aux]
        parts.extend(embedding(conditions[..., pos].long()) for pos, embedding in zip(self.category_positions, self.embeddings))
        _, hidden = self.gru(torch.cat(parts, dim=-1))
        correction = self.head(hidden[-1]).reshape(-1, self.horizon, 128)
        return base + torch.tanh(self.gate) * correction


def grouped_loss(prediction, target, c):
    continuous = [c["targets"].index(f) for f in c["continuous"]]
    boolean = [c["targets"].index(f) for f in c["boolean"]]
    error = nn.functional.huber_loss(prediction, target, reduction="none", delta=1.0)
    return 0.6 * error[..., continuous].mean() + 0.4 * error[..., boolean].mean()
