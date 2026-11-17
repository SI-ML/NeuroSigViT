# 视觉和数值分支的融合与分类头。
import torch
import torch.nn as nn


class _MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super().__init__()
        if num_layers < 1:
            raise ValueError("MLP must contain at least one linear layer")
        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        # 当前自适应主路径：池化输入 [B,2F] -> 隐层 [B,H] -> 未归一化 logits=[B,K]。
        return self.network(inputs)


class FusionModule(nn.Module):
    def __init__(
        self,
        branch_dims,
        modal_interaction="concat",
        fusion_dim=512,
        fusion_heads=4,
        cross_attn_query="ts",
        branch_names=None,
    ):
        super().__init__()

        if not branch_dims:
            raise ValueError("FusionModule requires at least one enabled branch.")
        if modal_interaction not in {
            "concat",
            "concat_attn",
            "cross_attn_gate",
            "masked_pretrain",
        }:
            raise ValueError(f"Unsupported modal_interaction: {modal_interaction}")

        self.modal_interaction = modal_interaction
        self.fusion_dim = fusion_dim
        self.fusion_heads = fusion_heads
        self.cross_attn_query = cross_attn_query
        self.input_branch_dims = [int(dim) for dim in branch_dims]
        branch_count = len(self.input_branch_dims)
        self.branch_names = list(branch_names or [
            f"branch_{index}" for index in range(branch_count)
        ])
        if len(self.branch_names) != branch_count:
            raise ValueError(
                f"branch_names must contain {branch_count} values, got {len(self.branch_names)}."
            )
        if len(set(self.branch_names)) != branch_count:
            raise ValueError(f"Branch names must be unique, got {self.branch_names}.")

        self.branch_dims = self.input_branch_dims

        self.projections = nn.ModuleList()
        if self.modal_interaction == "concat":
            self.output_dim = int(sum(self.branch_dims))
        else:
            for dim in self.branch_dims:
                self.projections.append(nn.Linear(dim, fusion_dim))

            if self.modal_interaction == "concat_attn":
                self.output_dim = fusion_dim * len(self.branch_dims)
                if fusion_dim % fusion_heads != 0:
                    raise ValueError(
                        f"fusion_dim ({fusion_dim}) must be divisible by fusion_heads ({fusion_heads})."
                    )
                self.attention = nn.MultiheadAttention(
                    embed_dim=fusion_dim,
                    num_heads=fusion_heads,
                    batch_first=True,
                )
            elif self.modal_interaction == "cross_attn_gate":
                if len(self.branch_dims) < 2:
                    raise ValueError(
                        "cross_attn_gate requires at least two enabled branches."
                    )
                self.output_dim = fusion_dim
                if fusion_dim % fusion_heads != 0:
                    raise ValueError(
                        f"fusion_dim ({fusion_dim}) must be divisible by fusion_heads ({fusion_heads})."
                    )
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=fusion_dim,
                    num_heads=fusion_heads,
                    batch_first=True,
                )
                self.gate = nn.Linear(2 * fusion_dim, fusion_dim)
            elif self.modal_interaction == "masked_pretrain":
                self.output_dim = fusion_dim
                self.fusion_encoder = nn.Sequential(
                    nn.Linear(fusion_dim * len(self.branch_dims), fusion_dim),
                    nn.ReLU(),
                    nn.Linear(fusion_dim, fusion_dim),
                )
                self.reconstruction_heads = nn.ModuleList(
                    [nn.Linear(fusion_dim, fusion_dim) for _ in self.branch_dims]
                )
            else:
                self.attention = None

    def _project_branches(self, branch_embeddings):
        if len(branch_embeddings) != len(self.branch_dims):
            raise ValueError(
                f"Expected {len(self.branch_dims)} branch embeddings, got {len(branch_embeddings)}."
            )

        selected_embeddings = []
        for branch_index, embedding in enumerate(branch_embeddings):
            expected_dim = self.input_branch_dims[branch_index]
            if embedding.ndim != 2 or embedding.shape[1] != expected_dim:
                raise ValueError(
                    f"Branch {self.branch_names[branch_index]} expects "
                    f"[batch, {expected_dim}], got {tuple(embedding.shape)}."
                )
            selected_embeddings.append(embedding)

        if self.modal_interaction == "concat":
            return selected_embeddings

        return [
            projection(embedding)
            for projection, embedding in zip(self.projections, selected_embeddings)
        ]

    # 输入是几路 [M,D] 向量（主方法：视觉、时序两路），按 modal_interaction 合成一个向量。
    def forward(self, branch_embeddings):
        if self.modal_interaction == "concat":
            return torch.cat(self._project_branches(branch_embeddings), dim=1)

        projected = self._project_branches(branch_embeddings)

        # 主方法 concat_attn：两路各自线性投影到 F 维 -> 叠成 2 个 token [M,2,F] ->
        # 2 个 token 之间做多头自注意力（断点 D31 在 attention 这一行，D32 在 return 这一行）-> 展平成 [M,2F]。
        if self.modal_interaction == "concat_attn":
            tokens = torch.stack(projected, dim=1)
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            return attended.reshape(attended.shape[0], -1)

        if self.modal_interaction == "cross_attn_gate":
            # For two branches, use them directly as visual and time-series streams.
            # If more than two branches are enabled, we keep the first projected branch
            # as the visual stream and collapse all remaining projected branches into
            # a single auxiliary time-series-side embedding by mean pooling.
            if len(projected) > 2:
                e_img = projected[0]
                e_ts = torch.stack(projected[1:], dim=0).mean(dim=0)
            else:
                e_img, e_ts = projected[0], projected[1]

            if self.cross_attn_query == "ts":
                query, context = e_ts, e_img
                query_is_ts = True
            else:
                query, context = e_img, e_ts
                query_is_ts = False

            e_cross, _ = self.cross_attention(
                query=query.unsqueeze(1),
                key=context.unsqueeze(1),
                value=context.unsqueeze(1),
                need_weights=False,
            )
            e_cross = e_cross.squeeze(1)

            gate_input = torch.cat([e_ts, e_img], dim=1)
            gate = torch.sigmoid(self.gate(gate_input))
            self.last_gate_mean = gate.detach().mean()

            if query_is_ts:
                return gate * e_ts + (1.0 - gate) * e_cross

            return gate * e_img + (1.0 - gate) * e_cross

        if self.modal_interaction == "masked_pretrain":
            return self.fusion_encoder(torch.cat(projected, dim=1))

        # Placeholder for future fusion strategies.
        return torch.cat(projected, dim=1)

    def reconstruct_masked(self, branch_embeddings, mask_prob=0.3, mask_index=None):
        if self.modal_interaction != "masked_pretrain":
            raise ValueError("reconstruct_masked is only available for masked_pretrain.")

        projected = self._project_branches(branch_embeddings)

        if mask_index is None:
            mask_index = torch.randint(len(projected), (1,), device=projected[0].device).item()
        apply_mask = bool((torch.rand((), device=projected[0].device) < mask_prob).item())
        if not apply_mask:
            return None

        masked = []
        for idx, embedding in enumerate(projected):
            if idx == mask_index:
                masked.append(torch.zeros_like(embedding))
            else:
                masked.append(embedding)

        fused = self.fusion_encoder(torch.cat(masked, dim=1))
        recon = self.reconstruction_heads[mask_index](fused)
        target = projected[mask_index]

        return recon, target, mask_index

