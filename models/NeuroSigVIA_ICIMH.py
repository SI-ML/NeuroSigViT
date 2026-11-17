from torch import nn
from torch.nn import functional as F
from layers.Fusion import FusionModule, _MLPHead
from layers.Pooling import ChannelAttentionPool, valid_fraction_weighted_pool


class NeuroSigVIA_ICIMH(nn.Module):
    def __init__(self, visual_dim, temporal_dim, num_channels, num_classes, *,
                 branches="both", fusion="concat_attn", fusion_dim=128, fusion_heads=2,
                 dropout=0.1, classifier_hidden_dim=128, classifier_num_layers=2,
                 channel_hidden_dim=64, mask_prob=0.3, numeric_reduce="pool"):
        super().__init__()
        self.branches = branches
        self.mask_prob = mask_prob
        head_dim = visual_dim
        self.numeric_reduce = numeric_reduce
        # 单数值分支（branches=numeric）的数值表示：pool 与主方法相同，通道注意力池化到 fusion_dim（默认）；
        # concat 逐通道特征直接拼接 [C*Dt]；mean 每通道同一个线性投影 Dt -> fusion_dim 后沿通道取平均。
        if branches == "both" or numeric_reduce == "pool" and branches == "numeric":
            self.channel_pool = ChannelAttentionPool(
                temporal_dim, fusion_dim, num_channels, channel_hidden_dim)
            head_dim = fusion_dim
        elif branches == "numeric" and numeric_reduce == "mean":
            self.numeric_projection = nn.Linear(temporal_dim, fusion_dim)
            head_dim = fusion_dim
        elif branches == "numeric":
            head_dim = num_channels * temporal_dim
        if branches == "both":
            self.fusion = FusionModule(
                branch_dims=[visual_dim, fusion_dim], modal_interaction=fusion,
                fusion_dim=fusion_dim, fusion_heads=fusion_heads,
                branch_names=["visual", "temporal"])
            head_dim = self.fusion.output_dim
        self.classifier = _MLPHead(head_dim, classifier_hidden_dim,
                                   classifier_num_layers, dropout, num_classes)

    def forward(self, graph_tokens, mantis_tokens, patch_mask, valid_fraction, *, pretrain=False):
        # 第 1 步：缓存 [B,N,C,Dt] 经通道注意力得到 [B,N,F]。
        temporal = None
        reconstruction_loss = None
        if self.branches == "both" or self.branches == "numeric" and self.numeric_reduce == "pool":
            temporal, _ = self.channel_pool(mantis_tokens, patch_mask)
        elif self.branches == "numeric":
            if self.numeric_reduce == "mean":
                temporal = self.numeric_projection(mantis_tokens).mean(dim=2)  # [B,N,C,Dt] -> [B,N,F]
            else:
                temporal = mantis_tokens.flatten(2)  # [B,N,C,Dt] -> [B,N,C*Dt]
        if self.branches == "both":
            # 第 2 步：只融合有效 patch 的两路特征 [M,Dv]、[M,F]。
            b, n = patch_mask.shape
            indices = patch_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            valid_branches = [graph_tokens.reshape(b*n, -1).index_select(0, indices),
                              temporal.reshape(b*n, -1).index_select(0, indices)]
            fused = self.fusion(valid_branches)
            if pretrain:
                reconstruction = self.fusion.reconstruct_masked(valid_branches, self.mask_prob)
                if reconstruction is not None:
                    prediction, target, _ = reconstruction
                    reconstruction_loss = F.mse_loss(prediction, target)
            # 第 3 步：回填 [B,N,D]，无效块保持零。
            patch_features = fused.new_zeros(b*n, self.fusion.output_dim)
            patch_features = patch_features.index_copy(0, indices, fused).reshape(b, n, -1)
        else:
            patch_features = graph_tokens if self.branches == "visual" else temporal
        # 第 4 步：按有效时长池化 [B,D]，输出 logits [B,K]。
        pooled = valid_fraction_weighted_pool(patch_features, patch_mask, valid_fraction)
        logits = self.classifier(pooled)
        return logits, {"temporal_tokens": temporal, "patch_features": patch_features,
                        "sample_features": pooled, "reconstruction_loss": reconstruction_loss}
