# 所有分支共用时间切块及尾部有效长度。
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TemporalPatchBatch:
    """One shared temporal partition consumed by all three feature branches."""

    patches: torch.Tensor
    time_mask: torch.Tensor
    patch_mask: torch.Tensor
    valid_lengths: torch.Tensor
    valid_fraction: torch.Tensor


def make_temporal_patches(
    inputs,
    window_size=64,
    stride=64,
    lengths=None,
):
    """Split ``[B, C, T]`` into aligned, right-padded temporal windows.

    ``lengths`` optionally describes the true prefix length of each sample
    when the input tensor was padded by a data loader.  Values outside those
    prefixes are zeroed in the returned patches and are never considered
    valid by downstream renderers.
    """
    # 第 1 步：检查块长、步长和输入形状 [B,C,T]。
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive integers")
    if stride > window_size:
        raise ValueError(
            "stride cannot exceed window_size because that would leave time points uncovered"
        )

    tensor = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs)
    if tensor.ndim != 3:
        raise ValueError(
            f"inputs must have shape [batch, channels, time], got {tuple(tensor.shape)}"
        )
    batch_size, channels, time_steps = tensor.shape
    # 输入维度 [B,C,T]：TDBRAIN 标准输入 [B,33,256]；B 轴对应有独立标签的分类窗口。
    if batch_size < 1 or channels < 1 or time_steps < 1:
        raise ValueError(f"input dimensions must be non-empty, got {tuple(tensor.shape)}")

    # 第 2 步：确定每个样本的真实长度；没有给 lengths 时（EEG 就是这样）视为整段 T 都有效。
    if lengths is None:
        true_lengths = torch.full(
            (batch_size,),
            time_steps,
            dtype=torch.long,
            device=tensor.device,
        )
    else:
        true_lengths = torch.as_tensor(
            lengths,
            dtype=torch.long,
            device=tensor.device,
        )
        if true_lengths.ndim != 1 or len(true_lengths) != batch_size:
            raise ValueError(
                f"lengths must have shape [{batch_size}], got {tuple(true_lengths.shape)}"
            )
        if torch.any(true_lengths <= 0):
            raise ValueError("every sample must contain at least one valid time point")
        if torch.any(true_lengths > time_steps):
            raise ValueError(
                f"sample lengths cannot exceed the input time dimension {time_steps}"
            )

    # The padded tensor width, rather than the largest length in this raw
    # mini-batch, defines N.  This keeps the structured cache shape stable
    # across loader batches while ``true_lengths`` only controls validity.
    if time_steps <= window_size:
        patch_count = 1
    else:
        patch_count = 1 + math.ceil((time_steps - window_size) / stride)
    # N=max(1,1+ceil((T-L)/S))；默认 T=256、L=S=64 得到 N=4，无重叠且无尾部补零。

    # 第 3 步：块数 N 已算出；若最后一块不够长，在右侧补零到 (N-1)*stride+L。
    required_time = (patch_count - 1) * stride + window_size
    if required_time > time_steps:
        padded = F.pad(tensor, (0, required_time - time_steps), value=0)
    else:
        padded = tensor

    # 第 4 步：沿时间轴滑窗切块。
    patches = padded.unfold(-1, window_size, stride)[..., :patch_count, :]
    patches = patches.permute(0, 2, 1, 3).contiguous()
    # unfold 得 [B,C,N,L]，调轴得到所有模态共用的 [B,N,C,L]，TDBRAIN 为 [B,4,33,64]。

    # 第 5 步：计算每块每个时间点是否落在真实长度内，得到 time_mask，再由它得到
    # 有效长度、块是否有效、有效比例（断点 B1 在这一行，此时 patches 已切好）。
    starts = torch.arange(
        patch_count,
        dtype=torch.long,
        device=tensor.device,
    ) * stride
    offsets = torch.arange(
        window_size,
        dtype=torch.long,
        device=tensor.device,
    )
    absolute_positions = starts[:, None] + offsets[None, :]
    time_mask = absolute_positions[None, :, :] < true_lengths[:, None, None]
    valid_lengths = time_mask.sum(dim=-1)
    patch_mask = valid_lengths > 0
    valid_fraction = valid_lengths.to(dtype=torch.float32) / float(window_size)
    # time_mask=[B,N,L]；lengths/mask/fraction 均为 [B,N]；完整块 fraction=1，空块 mask=False。

    # 第 6 步：无效时间点清零，打包返回。
    patches = patches.masked_fill(~time_mask[:, :, None, :], 0)
    return TemporalPatchBatch(
        patches=patches,
        time_mask=time_mask,
        patch_mask=patch_mask,
        valid_lengths=valid_lengths,
        valid_fraction=valid_fraction,
    )

