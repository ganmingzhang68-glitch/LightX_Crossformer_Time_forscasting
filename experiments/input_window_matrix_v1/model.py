"""Generalized Crossformer with optional historical condition/quality/event GRU."""
import torch
from torch import nn
from vendor.crossformer import Crossformer, CrossformerConfig


class ForecastModel(nn.Module):
    def __init__(self, c, group, pre, params):
        super().__init__()
        p, g = c["profile"], c["groups"][group]
        self.profile, self.horizon, self.outputs = p, g["horizon"], len(c["targets"])
        self.indices = [p["dynamic"].index(f) for f in c["targets"]]
        self.backbone = Crossformer(CrossformerConfig.from_dict(dict(params,
            enc_in=len(p["dynamic"]), seq_len=g["context"], pred_len=g["horizon"],
            use_revin=True, residual_forecast=True)))
        self.categories = [f for f in p["conditions"] if f in c["categorical"]]
        self.scalars = [i for i, f in enumerate(p["conditions"]) if f not in self.categories]
        self.embeddings = nn.ModuleList([nn.Embedding(len(pre["categorical_vocabulary"][f]) + 2, 4, padding_idx=0)
                                        for f in self.categories])
        with torch.no_grad():
            for e in self.embeddings:
                e.weight[1].zero_()
        masks = len(set(p["dynamic"] + p["conditions"]) & set(c["added"])) if p["missing_masks"] else 0
        self.aux_width = masks + (3 if p["quality"] else 0) + (4 * (len(c["storage_boolean"]) + len(p["conditions"])) if p["events"] else 0)
        width = len(self.scalars) + 4 * len(self.categories) + self.aux_width
        self.has_correction = width > 0
        if self.has_correction:
            hidden = params["condition_hidden"]
            self.gru = nn.GRU(width, hidden, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, self.horizon * self.outputs))
            self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, dynamic, conditions, aux):
        if dynamic.shape[-1] != len(self.profile["dynamic"]) or conditions.shape[-1] != len(self.profile["conditions"]) or aux.shape[-1] != self.aux_width:
            raise ValueError("Input shape differs from selected profile")
        base = self.backbone(dynamic)[..., self.indices]
        if not self.has_correction:
            return base
        parts = [aux, conditions[..., self.scalars]]
        parts.extend(e(conditions[..., self.profile["conditions"].index(f)].long()) for f, e in zip(self.categories, self.embeddings))
        _, h = self.gru(torch.cat(parts, dim=-1))
        return base + torch.tanh(self.gate) * self.head(h[-1]).reshape(-1, self.horizon, self.outputs)


def grouped_loss(prediction, target, c):
    error = nn.functional.huber_loss(prediction, target, reduction="none", delta=1.0)
    ci = [c["targets"].index(f) for f in c["continuous"] if f in c["score_targets"]]
    bi = [c["targets"].index(f) for f in c["boolean"] if f in c["score_targets"]]
    primary = 0.6 * error[..., ci].mean() + 0.4 * error[..., bi].mean()
    extra = [i for i, f in enumerate(c["targets"]) if f not in c["score_targets"]]
    # Preserve the old133 experiment's auxiliary-target design; don't rank on new settings.
    return 0.9 * primary + 0.1 * error[..., extra].mean() if extra else primary
