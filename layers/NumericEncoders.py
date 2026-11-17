# 冻结数值骨干 TSPulse / Zeus（替换 Mantis 的对照实验）：输入单通道 [B,1,512]，输出 [B,D]，调用方式与 Mantis8M 相同。
import importlib
import json
import os
import sys
import types

import torch
import torch.nn.functional as F
from torch import nn


def _import_tspulse(repo):
    # tsfm_public/__init__.py 依赖新版 transformers 的 pipeline 接口；这里只加载 TSPulse 模型文件，跳过包初始化。
    root = os.path.join(repo, "tsfm_public")
    for name, path in (("tsfm_public", root), ("tsfm_public.models", os.path.join(root, "models")),
                       ("tsfm_public.models.tspulse", os.path.join(root, "models", "tspulse"))):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [path]
            sys.modules[name] = module
    return importlib.import_module("tsfm_public.models.tspulse.modeling_tspulse")


class TSPulseEncoder(nn.Module):
    # TSPulse 骨干输出 [B,1,69,48]（32 个时域块 + 32 个频域块 + 5 个寄存 token），展平为 [B,3312]。
    # 推理时关闭掩码（官方分类流程同样传 enable_masking=False）。
    def __init__(self, path, repo):
        super().__init__()
        self.model = _import_tspulse(repo).TSPulseForReconstruction.from_pretrained(path).backbone

    def forward(self, x):
        output = self.model(past_values=x.transpose(1, 2), enable_masking=False, return_dict=True)
        return output.last_hidden_state.flatten(1)


def _import_zeus(repo, basicts_dir):
    # Zeus 只用到 BasicTS 1.1.0 的 basicts.modules；BasicTS 以 --no-deps 装在独立目录，不改动主环境。
    # basicts/__init__.py 会导入训练框架 easytorch，这里只注册包路径、跳过包初始化；zeus/ 同样注册为包后导入。
    for name, path in (("basicts", os.path.join(basicts_dir, "basicts")), ("zeus", os.path.join(repo, "zeus"))):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [path]
            sys.modules[name] = module
    return importlib.import_module("zeus.modeling_zeus")


def _zeus_sdpa_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                       use_cache=False, output_attentions=False, layer_idx=None):
    # ZeusFlashAttention.forward 的等价实现（服务器未装 flash-attn）：官方先去掉 padding token，再按各 token 原位置
    # 施加 RoPE，做非因果的 flash_attn_varlen_func（默认缩放 1/sqrt(d)），最后 padding 位置补 0。
    # 这里对全部 token 按原位置施加 RoPE，用 key padding mask 做 SDPA，再把 padding 位置置 0，结果相同。
    import torch
    import torch.nn.functional as F
    B, L, _ = hidden_states.shape
    q = self._shape(self.q_proj(hidden_states), B, L)
    k = self._shape(self.k_proj(hidden_states), B, L)
    v = self._shape(self.v_proj(hidden_states), B, L)
    valid = (torch.ones((B, L), dtype=torch.bool, device=hidden_states.device) if attention_mask is None
             else attention_mask.reshape(B, L).bool())
    if self.rope is not None:
        if position_ids is None:
            position_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        q, k = self.rope(q.reshape(B * L, self.n_heads, self.head_size),
                         k.reshape(B * L, self.n_heads, self.head_size), position_ids.reshape(-1))
        q, k = q.view(B, L, self.n_heads, self.head_size), k.view(B, L, self.n_heads, self.head_size)
    context = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             attn_mask=valid[:, None, None, :], dropout_p=0.0)
    context = context.transpose(1, 2).reshape(B, L, self.hidden_size) * valid[..., None]
    return self.out_proj(context), None, past_key_value


class ZeusEncoder(nn.Module):
    # 官方分类接口 ZeusForClassification.generate_one_sample：逐通道减均值、除以标准差后取 arcsinh，bf16 前向，
    # 返回 5 个阶段（尺度 1/8/32/8/1，维度 384/768/768/768/384）的隐状态。默认按论文附录 A.2 取第 4 阶段
    # （倒数第二个尺度，scale 8，768 维），沿时间做最大池化 -> [B,768]。512 点能被最大尺度 32 整除，不产生 padding 位置。
    # stages 给多个阶段时（仅供一次前向同时构造多份缓存），各阶段池化后分别 L2 归一化再拼接。
    def __init__(self, path, repo, basicts_dir, stages=(4,)):
        super().__init__()
        module = _import_zeus(repo, basicts_dir)
        # ZeusConfig 写死 self.dtype = torch.bfloat16，transformers 4.33 打印配置时无法序列化；
        # 模型代码不读取该字段，这里直接由 config.json 构造并删去它。
        # 官方权重对应 flash_attention_2 结构（out_proj 无 bias），按该结构搭建，前向换成等价的 SDPA。
        with open(os.path.join(path, "config.json")) as handle:
            values = json.load(handle)
        values["attn_implementation"] = "flash_attention_2"
        config = module.ZeusConfig(**values)
        del config.dtype
        # 该 safetensors 无元数据，transformers 4.33 的 from_pretrained 无法读取；直接实例化后严格载入全部权重。
        from safetensors.torch import load_file
        module.ZeusFlashAttention.forward = _zeus_sdpa_forward
        self.model = module.ZeusForClassification(config)
        self.model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=True)
        self.stages = tuple(stages)

    def forward(self, x):
        states = self.model.generate_one_sample(x.transpose(1, 2), use_norm=True)
        pooled = [states[stage - 1].float().amax(dim=1) for stage in self.stages]
        if len(pooled) == 1:
            return pooled[0]
        return torch.cat([F.normalize(p, dim=-1, eps=1e-8) for p in pooled], dim=-1)


def build_numeric_encoder(name, path, zeus_stages=(4,)):
    if name == "tspulse":
        return TSPulseEncoder(path, os.environ.get("TSPULSE_REPO", os.path.expanduser("~/granite-tsfm")))
    if name == "zeus":
        return ZeusEncoder(path, os.environ.get("ZEUS_REPO", os.path.expanduser("~/Zeus")),
                           os.environ.get("BASICTS_DIR", os.path.expanduser("~/pylibs/basicts110")), stages=zeus_stages)
    raise ValueError(f"Unknown numeric backbone: {name}")
