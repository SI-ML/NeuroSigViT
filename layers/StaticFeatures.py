# 冻结 Mantis 逐通道特征。
from __future__ import annotations
import torch
import torch.nn.functional as F
from utils.tools import resize_mantis_input


@torch.no_grad()
def _extract_mantis_channel_tokens(
    windows,
    valid_lengths,
    mantis_model,
    device,
    encode_batch_size,
):
    # 输入有效块 [V,C,L]；逐通道裁切为 [V_length,1,length] 后线性插值至 [V_length,1,512]。
    # Mantis 单次返回 [V_chunk,Dt]；按原位置重组并堆叠通道后返回 [V,C,Dt]。
    # 当前调用的是原始 Mantis8M 的 forward。
    # 逐通道处理（外层循环 33 次）：
    #   1. 按有效长度把块分组（EEG 全部 64 点，只有一组）；
    #   2. 取出该通道 [块数,1,64]，线性插值到 Mantis 要求的 512 点（断点 B7 在检查 output 形状的 if 这一行）；
    #   3. 分组送进 Mantis 得到 [组大小,256]，L2 归一化后放回各块原位置；
    # 最后把 33 个通道堆成 [块数,33,256]（断点 B8 在 return 这一行）。
    channels = windows.shape[1]
    channel_outputs = []
    for channel_index in range(channels):
        outputs_for_channel = None
        for length in torch.unique(valid_lengths, sorted=True).tolist():
            length = int(length)
            if length <= 0:
                continue
            indices = torch.nonzero(valid_lengths == length, as_tuple=False).flatten()
            cropped = windows.index_select(0, indices)[:, channel_index : channel_index + 1, :length]
            resized = resize_mantis_input(cropped).float()
            output_batches = []
            for start in range(0, len(resized), encode_batch_size):
                stop = min(start + encode_batch_size, len(resized))
                output = mantis_model(resized[start:stop].to(device))
                if not torch.is_tensor(output) or output.ndim != 2:
                    raise ValueError(
                        "Mantis must return [batch, feature_dim], got "
                        f"{type(output).__name__} {getattr(output, 'shape', None)}"
                    )
                output_batches.append(F.normalize(output.float(), dim=-1, eps=1e-8).cpu())
            selected_outputs = torch.cat(output_batches, dim=0)
            if outputs_for_channel is None:
                outputs_for_channel = torch.empty(
                    (len(windows), selected_outputs.shape[-1]),
                    dtype=selected_outputs.dtype,
                )
            outputs_for_channel.index_copy_(0, indices, selected_outputs)
        if outputs_for_channel is None:
            raise ValueError("Mantis received only empty temporal patches")
        channel_outputs.append(outputs_for_channel)
    return torch.stack(channel_outputs, dim=1)

