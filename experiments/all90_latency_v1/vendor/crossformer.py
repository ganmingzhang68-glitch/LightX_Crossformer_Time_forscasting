# Independent snapshot; einops-only tensor layout operations replaced with native Torch.
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, sqrt

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class CrossformerConfig:
    enc_in: int = 128
    seq_len: int = 150
    pred_len: int = 5
    d_model: int = 128
    n_heads: int = 2
    e_layers: int = 3
    d_ff: int = 512
    factor: int = 1
    dropout: float = 0.0
    crossformer_seg_len: int = 10
    crossformer_win_size: int = 2
    use_revin: bool = True
    revin_affine: bool = True
    residual_forecast: bool = True

    @classmethod
    def from_dict(cls, values: dict) -> "CrossformerConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: values[key] for key in allowed if key in values})


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False,
                 subtract_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            dimensions = tuple(range(1, x.ndim - 1))
            if self.subtract_last:
                self.last = x[:, -1, :].unsqueeze(1)
            else:
                self.mean = torch.mean(x, dim=dimensions, keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(x, dim=dimensions, keepdim=True, unbiased=False) + self.eps
            ).detach()
            if self.subtract_last:
                x = x - self.last
            else:
                x = x - self.mean
            x = x / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev
            return x + (self.last if self.subtract_last else self.mean)
        raise ValueError(f"Unsupported normalization mode: {mode}")


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int,
                 dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class FullAttention(nn.Module):
    def __init__(self, mask_flag: bool, factor: int, scale=None,
                 attention_dropout: float = 0.1, output_attention: bool = False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        _, _, _, embedding_size = queries.shape
        scale = self.scale or 1.0 / sqrt(embedding_size)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            raise RuntimeError("This deployment model does not use causal self-attention")
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attention, values)
        return values.contiguous(), attention if self.output_attention else None


class AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int,
                 d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        batch, query_length, _ = queries.shape
        _, key_length, _ = keys.shape
        heads = self.n_heads
        queries = self.query_projection(queries).view(batch, query_length, heads, -1)
        keys = self.key_projection(keys).view(batch, key_length, heads, -1)
        values = self.value_projection(values).view(batch, key_length, heads, -1)
        output, attention = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        return self.out_projection(output.reshape(batch, query_length, -1)), attention


class TwoStageAttentionLayer(nn.Module):
    def __init__(self, config: CrossformerConfig, seg_num: int, factor: int,
                 d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        attention = lambda: AttentionLayer(
            FullAttention(False, config.factor, attention_dropout=config.dropout),
            d_model,
            n_heads,
        )
        self.time_attention = attention()
        self.dim_sender = attention()
        self.dim_receiver = attention()
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        time_input = x.flatten(0, 1)
        time_encoded, _ = self.time_attention(time_input, time_input, time_input)
        dimension_input = self.norm1(time_input + self.dropout(time_encoded))
        dimension_input = self.norm2(dimension_input + self.dropout(self.MLP1(dimension_input)))
        dimension_send = dimension_input.reshape(batch, -1, dimension_input.shape[1], dimension_input.shape[2]).permute(0, 2, 1, 3).flatten(0, 1)
        routers = self.router.unsqueeze(0).expand(batch, -1, -1, -1).flatten(0, 1)
        buffer, _ = self.dim_sender(routers, dimension_send, dimension_send)
        received, _ = self.dim_receiver(dimension_send, buffer, buffer)
        dimension_encoded = self.norm3(dimension_send + self.dropout(received))
        dimension_encoded = self.norm4(
            dimension_encoded + self.dropout(self.MLP2(dimension_encoded))
        )
        return dimension_encoded.reshape(batch, -1, dimension_encoded.shape[1], dimension_encoded.shape[2]).permute(0, 2, 1, 3)


class SegMerging(nn.Module):
    def __init__(self, d_model: int, win_size: int):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = nn.LayerNorm(win_size * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_num = x.shape[2] % self.win_size
        if pad_num:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
        merged = torch.cat([x[:, :, index::self.win_size, :] for index in range(self.win_size)], -1)
        return self.linear_trans(self.norm(merged))


class ScaleBlock(nn.Module):
    def __init__(self, config: CrossformerConfig, win_size: int, seg_num: int):
        super().__init__()
        self.merge_layer = SegMerging(config.d_model, win_size) if win_size > 1 else None
        self.encode_layers = nn.ModuleList([
            TwoStageAttentionLayer(
                config, seg_num, config.factor, config.d_model, config.n_heads,
                config.d_ff, config.dropout
            )
        ])

    def forward(self, x: torch.Tensor):
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        for layer in self.encode_layers:
            x = layer(x)
        return x, None


class Encoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.encode_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor):
        encoded = [x]
        for block in self.encode_blocks:
            x, _ = block(x)
            encoded.append(x)
        return encoded, None


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, seg_len: int,
                 d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x: torch.Tensor, cross: torch.Tensor):
        batch = x.shape[0]
        x = self.self_attention(x)
        x = x.flatten(0, 1)
        cross = cross.flatten(0, 1)
        temporary, _ = self.cross_attention(x, cross, cross)
        x = x + self.dropout(temporary)
        prediction_input = x = self.norm1(x)
        prediction_input = self.MLP1(prediction_input)
        decoded = self.norm2(x + prediction_input)
        decoded = decoded.reshape(batch, -1, decoded.shape[1], decoded.shape[2])
        layer_prediction = self.linear_pred(decoded).flatten(1, 2)
        return decoded, layer_prediction


class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.decode_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, cross):
        final_prediction = None
        dimensions = x.shape[1]
        for index, layer in enumerate(self.decode_layers):
            x, layer_prediction = layer(x, cross[index])
            final_prediction = (
                layer_prediction if final_prediction is None
                else final_prediction + layer_prediction
            )
        return final_prediction.reshape(final_prediction.shape[0], dimensions, -1, final_prediction.shape[-1]).permute(0, 2, 3, 1).flatten(1, 2)


class Crossformer(nn.Module):
    def __init__(self, config: CrossformerConfig):
        super().__init__()
        self.config = config
        self.pred_len = config.pred_len
        self.seg_len = config.crossformer_seg_len
        self.win_size = config.crossformer_win_size
        self.use_revin = config.use_revin
        self.residual_forecast = config.residual_forecast
        if self.use_revin:
            self.revin = Normalize(
                config.enc_in,
                affine=config.revin_affine,
                subtract_last=config.residual_forecast,
            )
        self.pad_in_len = ceil(config.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(config.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (config.e_layers - 1)))
        self.head_nf = config.d_model * self.out_seg_num
        self.enc_value_embedding = PatchEmbedding(
            config.d_model, self.seg_len, self.seg_len,
            self.pad_in_len - config.seq_len, 0
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, config.enc_in, self.in_seg_num, config.d_model)
        )
        self.pre_norm = nn.LayerNorm(config.d_model)
        self.encoder = Encoder([
            ScaleBlock(
                config,
                1 if level == 0 else self.win_size,
                self.in_seg_num if level == 0 else ceil(self.in_seg_num / self.win_size ** level),
            )
            for level in range(config.e_layers)
        ])
        output_segments = self.pad_out_len // self.seg_len
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, config.enc_in, output_segments, config.d_model)
        )
        self.decoder = Decoder([
            DecoderLayer(
                TwoStageAttentionLayer(
                    config, output_segments, config.factor, config.d_model,
                    config.n_heads, config.d_ff, config.dropout
                ),
                AttentionLayer(
                    FullAttention(
                        False, config.factor, attention_dropout=config.dropout,
                        output_attention=False
                    ),
                    config.d_model,
                    config.n_heads,
                ),
                self.seg_len,
                config.d_model,
                config.d_ff,
                config.dropout,
            )
            for _ in range(config.e_layers + 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.config.seq_len, self.config.enc_in):
            raise ValueError(
                f"Expected [batch,{self.config.seq_len},{self.config.enc_in}], got {list(x.shape)}"
            )
        residual_base = x[:, -1:, :] if self.residual_forecast else None
        if self.use_revin:
            x = self.revin(x, "norm")
        elif self.residual_forecast:
            x = x - residual_base
        x, dimensions = self.enc_value_embedding(x.permute(0, 2, 1))
        x = x.reshape(-1, dimensions, x.shape[1], x.shape[2])
        x = self.pre_norm(x + self.enc_pos_embedding)
        encoded, _ = self.encoder(x)
        decoder_input = self.dec_pos_embedding.repeat(x.shape[0], 1, 1, 1)
        output = self.decoder(decoder_input, encoded)
        if self.use_revin:
            output = self.revin(output, "denorm")
        elif self.residual_forecast:
            output = output + residual_base
        return output[:, -self.pred_len:, :]
