# 冻结视觉编码器；静态图像经空间均值和投影得到单个向量。
from __future__ import annotations
import os
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
OPENCLIP_LAION_MODELS = {
    "clip-vit-h-14-laion2b-s32b-b79k": ("ViT-H-14", "laion2b_s32b_b79k"),
}


def get_openclip_config(model_name):
    model_key = model_name.lower()
    model_key = os.path.basename(os.path.normpath(model_key))

    if model_key in OPENCLIP_LAION_MODELS:
        return OPENCLIP_LAION_MODELS[model_key]

    raise ValueError(f"Unsupported OpenCLIP model {model_name}.")


def find_openclip_checkpoint(model_dir):
    preferred_names = [
        "open_clip_pytorch_model.bin",
        "open_clip_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
        "model.bin",
        "model.pt",
        "model.pth",
    ]

    for name in preferred_names:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return path

    for name in os.listdir(model_dir):
        if name.endswith((".safetensors", ".bin", ".pt", ".pth")):
            return os.path.join(model_dir, name)

    raise FileNotFoundError(
        f"No OpenCLIP checkpoint file found in {model_dir}. "
        "Expected a .bin, .pt, .pth, or .safetensors file."
    )


def get_processor_vit(model_name):
    openclip_model_name, openclip_pretrained = get_openclip_config(model_name)
    if os.path.isdir(model_name):
        openclip_pretrained = find_openclip_checkpoint(model_name)

    model, _, processor = open_clip.create_model_and_transforms(
        model_name=openclip_model_name,
        pretrained=openclip_pretrained,
    )
    vit = model.visual

    return processor, vit


def get_neurosigvia(
    model_name,
    model_layer,
    aggregation,
    stride,
    patch_size,
    image_mode="line_plot",
):
    processor, vit = get_processor_vit(model_name)

    if hasattr(vit, "transformer") and hasattr(vit.transformer, "resblocks"):
        NeuroSigVIAClass = NeuroSigVIA_OpenCLIP
    else:
        raise ValueError("Unsupported model structure.")

    neurosigvia = NeuroSigVIAClass(
        processor=processor,
        vit=vit,
        layer_idx=model_layer,
        aggregation=aggregation,
        patch_size=patch_size,
        stride=stride,
        image_mode=image_mode,
    )
    return neurosigvia


class NeuroSigVIA_OpenCLIP(nn.Module):
    # 保留预处理流水线最后的归一化步骤，图像尺寸由上游成像函数处理。
    def __init__(
        self, processor, vit, layer_idx, aggregation, patch_size, stride, image_mode
    ):
        self.hidden_representations = {}
        super().__init__()
        self.processor = processor
        self.vit = vit
        self.layer_idx = layer_idx
        self.aggregation = aggregation
        self.patch_size = patch_size
        self.stride = stride
        self.image_mode = image_mode
        self.truncate_layers()
        self.processor.transforms = [self.processor.transforms[-1]]

    # layer_idx 指保留的 Transformer 块数；None 或 -1 不截断。
    def truncate_layers(self):
        if self.layer_idx is not None and self.layer_idx != -1:
            self.vit.transformer.resblocks = self.vit.transformer.resblocks[
                : self.layer_idx
            ]

    def forward_vit(self, inputs):
        # 主脚本 ViT-H/14、layer_idx=14：RGB [V,3,224,224] -> [V,257,1280]。
        # _embeds 构造 CLS+16*16 空间 token；truncate_layers 已将 resblocks 截至前 14 层。
        # 本路径使用张量归一化，未 detach 或 no_grad；冻结权重时仍可对在线图像求导。
        # 步骤：图像归一化（只保留 CLIP 预处理的最后一步 Normalize）-> 切成 14×14 像素的 patch
        # 并加上 CLS 和位置编码 -> 依次经过保留的 14 个 Transformer 块 -> 返回最后一层的全部 token。
        hidden_states = []

        inputs = self.processor(inputs)

        x = self.vit._embeds(inputs)
        hidden_states.append(x)

        for blk in self.vit.transformer.resblocks:
            x = blk(x)
            hidden_states.append(x)

        if self.layer_idx is not None:
            return x
        else:
            return torch.stack(hidden_states, dim=-1)

    def aggregate_hidden_representations(self, hidden_states, aggregation):
        # 静态 line 路径调用此函数：[V,1+P,Dh] -> [V,Dh] -> [V,Dv]。
        # 在线活动图不调用此全局聚合，改保留 4*4 空间 token 作为交叉注意力 K/V。
        if aggregation == "mean":
            if hidden_states.ndim != 3 or hidden_states.shape[1] < 2:
                raise ValueError(
                    "Mean aggregation expects [batch, cls+patches, hidden] with "
                    f"at least one patch token, got {tuple(hidden_states.shape)}."
                )
            # NeuroSigVIA mean-pools spatial patch tokens only.  The leading
            # OpenCLIP token is the CLS token and is deliberately
            # excluded from the activity-graph representation.
            pooled = hidden_states[:, 1:, :].mean(dim=1)
        elif aggregation == "cls_token":
            pooled = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unsupported aggregation {aggregation}")

        return self.project_pooled_representation(pooled)


    def project_pooled_representation(self, pooled):
        # 对最后一维执行 1280 -> 1024 投影：兼容静态 [V,1280] 和在线 [V,16,1280]。
        # OpenCLIP ViT-H/14 has transformer width 1280 and a learned 1024-D
        # output projection.  Applying the frozen post norm/projection makes
        # the implementation match NeuroSigVIA Eq. (5), while keeping the
        # selected intermediate transformer depth fixed.
        if hasattr(self.vit, "ln_post"):
            pooled = self.vit.ln_post(pooled)
        projection = getattr(self.vit, "proj", None)
        if projection is not None:
            pooled = pooled @ projection
        return pooled


def _encode_visual_images(model, images, device):
    # 静态折线图 [V,3,224,224] -> forward_vit 的 [V,257,1280]（ViT-H/14）。
    # mean 聚合排除 CLS，接冻结 ln_post/proj 后为 [V,1024]，再沿特征维做 L2 归一化。
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"visual images must have shape [B, 3, H, W], got {tuple(images.shape)}")
    # 静态折线图编码：ViT 前向（断点 B5 在 embeddings 这一行）-> 去 CLS 均值池化 + ln_post/proj
    # （断点 B6 在下面的 if 这一行）-> L2 归一化。
    hidden = model.forward_vit(images.to(device))
    embeddings = model.aggregate_hidden_representations(
        hidden,
        aggregation=model.aggregation,
    )
    if embeddings.ndim != 2:
        raise ValueError(
            f"the visual encoder must return one vector per image, got {tuple(embeddings.shape)}"
        )
    return F.normalize(embeddings.float(), dim=-1, eps=1e-8)

