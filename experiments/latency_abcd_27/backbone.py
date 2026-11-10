"""Repository-owned Crossformer adapter for the historical four-argument API."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crossformer import Crossformer, CrossformerConfig


class HistoricalCrossformer(Crossformer):
    def forward(self, x, x_mark=None, decoder=None, decoder_mark=None, mask=None):
        return super().forward(x)


def build_backbone(params, enc_in):
    config = dict(params, enc_in=enc_in, seq_len=150, pred_len=60,
                  crossformer_win_size=2, use_revin=True, residual_forecast=True)
    if int(config["d_model"]) % int(config["n_heads"]):
        raise ValueError("d_model must be divisible by n_heads")
    return HistoricalCrossformer(CrossformerConfig.from_dict(config)).float()
