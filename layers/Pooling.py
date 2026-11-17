# 通道注意力池化与有效时长池化。
from __future__ import annotations
import torch
import torch.nn as nn


class ChannelAttentionPool(nn.Module):
    """Trainable channel pooling for frozen per-channel Mantis tokens."""

    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels,
        hidden_dim=64,
    ):
        super().__init__()
        if min(input_dim, output_dim, num_channels, hidden_dim) <= 0:
            raise ValueError("channel-attention dimensions must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_channels = int(num_channels)
        self.value_projection = nn.Linear(input_dim, output_dim)
        self.value_norm = nn.LayerNorm(output_dim)
        self.channel_embeddings = nn.Parameter(
            torch.empty(num_channels, output_dim)
        )
        self.scorer = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.channel_embeddings, mean=0.0, std=0.02)

    def forward(self, tokens, patch_mask=None):
        # tokens=[B,N,C,Dt] -> values=[B,N,C,F]；F=fusion_dim，当前 TDBRAIN 脚本设为 128。
        if tokens.shape[2] != self.num_channels or tokens.shape[3] != self.input_dim:
            raise ValueError(
                "Mantis channel-token shape does not match the configured "
                f"[C={self.num_channels}, D={self.input_dim}]: {tuple(tokens.shape)}"
            )

        # 步骤：每个通道向量线性投影到 F 维（values）-> LayerNorm 后加上可学习的通道嵌入 ->
        # 小 MLP 给每个通道打一个分 -> 沿通道 softmax 得权重 -> 按权重求和（断点 D22 在 if patch_mask 这一行）。
        values = self.value_projection(tokens)
        score_inputs = self.value_norm(values) + self.channel_embeddings.view(
            1, 1, self.num_channels, self.output_dim
        )
        scores = self.scorer(score_inputs).squeeze(-1)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=values.dtype)
        pooled = torch.sum(weights.unsqueeze(-1) * values, dim=2)
        # softmax 沿 C 轴，weights=[B,N,C]，通道加权和 pooled=[B,N,F]，不是通道拼接。

        if patch_mask is not None:
            valid = patch_mask.to(dtype=torch.bool, device=tokens.device)
            weights = weights * valid.unsqueeze(-1)
            pooled = pooled * valid.unsqueeze(-1)
        return pooled, weights


def valid_fraction_weighted_pool(
    patch_features,
    patch_mask,
    valid_fraction,
):
    """Pool valid patches while giving a partial tail its true time weight."""
    # 输入 [B,N,D] 与两项 [B,N] 权重，沿 N 求加权均值得 [B,D]；完整 TDBRAIN 块等权 1/4。
    pooling_weights = (
        valid_fraction.to(dtype=patch_features.dtype, device=patch_features.device)
        * patch_mask.to(dtype=patch_features.dtype, device=patch_features.device)
    )
    denominators = pooling_weights.sum(dim=1, keepdim=True)
    if torch.any(denominators <= 0):
        raise ValueError("every sample must contain at least one valid patch")
    return torch.sum(
        patch_features * pooling_weights.unsqueeze(-1), dim=1
    ) / denominators

