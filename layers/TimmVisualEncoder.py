from pathlib import Path
import timm
import torch
from torch import nn
from torchvision.transforms import Normalize


class TimmVisualEncoder(nn.Module):
    def __init__(self, backbone, model_dir, layer_idx):
        super().__init__()
        names = {"siglip2": "vit_so400m_patch14_siglip_224.v2_webli",
                 "dinov3": "vit_large_patch16_dinov3.lvd1689m",
                 "swin": "swin_base_patch4_window7_224.ms_in22k"}
        self.backbone = backbone
        if backbone == "swin":
            # ViTST 的视觉骨干：Swin-B（ImageNet-22k），4 个阶段 [2,2,18,2] 块，输出 128/256/512/1024 维；layer_idx 为阶段号 1–4，
            # 4 = 完整 Swin（同 ViTST）。本地权重含 22k 分类头，num_classes=0 时由 timm 预训练加载流程丢弃分类头。
            if layer_idx not in (1, 2, 3, 4):
                raise ValueError("swin: layer is the stage index 1-4")
            self.vit = timm.create_model(names[backbone], pretrained=True, num_classes=0,
                pretrained_cfg_overlay=dict(file=str(Path(model_dir) / "model.safetensors")))
        else:
            # 只读取本地权重；224×224 输入沿用上游活动图，不执行裁剪或缩放。
            self.vit = timm.create_model(names[backbone], pretrained=False, num_classes=0,
                img_size=224, checkpoint_path=str(Path(model_dir) / "model.safetensors"))
            # 加载完整权重后仅保留前 L 块，避免把不执行的后续块搬到 GPU。
            self.vit.blocks = self.vit.blocks[:layer_idx]
        self.processor = Normalize(self.vit.pretrained_cfg["mean"], self.vit.pretrained_cfg["std"])
        self.layer_idx = layer_idx
        self.aggregation = "mean"

    def forward_vit(self, inputs):
        if self.backbone == "swin":
            # 第 L 阶段未归一化输出 [M,C,H,W] -> [M,H*W,C]（第 4 阶段为 [M,49,1024]）
            hidden = self.vit.forward_intermediates(self.processor(inputs), indices=[self.layer_idx - 1], norm=False, stop_early=True,
                output_fmt="NCHW", intermediates_only=True)[0]
            return hidden.flatten(2).transpose(1, 2)
        # timm 的索引从第一个块起为 0；L-1 对应前 L 块，自动排除全部 0/5 个前缀 token。
        # [M,3,224,224] -> [M,256,1152] 或 [M,196,1024]，保留未归一化的空间 token。
        return self.vit.forward_intermediates(self.processor(inputs), indices=[self.layer_idx - 1],
            norm=False, stop_early=True, output_fmt="NLC", intermediates_only=True)[0]

    def aggregate_hidden_representations(self, hidden_states, aggregation):
        # [M,P,Dv] 先空间均值再 final norm；共享编码入口随后执行 L2，不使用投影或原模型分类头。
        if self.backbone == "swin":
            # 第 4 阶段：Swin 原生池化特征（ViTST 分类头之前的表示），先 final norm 再空间均值，等于 forward_head(pre_logits=True)；
            # 中间阶段维度与 final norm 不符，只取空间均值。
            if self.layer_idx == 4:
                return self.vit.norm(hidden_states).mean(dim=1)
            return hidden_states.mean(dim=1)
        return self.vit.norm(hidden_states.mean(dim=1))
