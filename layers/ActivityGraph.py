# 原始波形活动图；保留参考实现的栅格化数值。
from __future__ import annotations
from functools import lru_cache
import math
from typing import Optional
import torch
import torch.nn.functional as F
PAPER_ACTIVITY_GRAPH_CANVAS_SIZE = 360


def _edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


@lru_cache(maxsize=None)
def paper_signal_order(num_channels: int) -> tuple[int, ...]:
    """Return the exact deterministic order described by paper Algorithm 1.

    IDs in the paper are one-based; this implementation uses zero-based tensor
    indices. The initial sequence is ``0, 1, ..., C-1`` and the paper's scan
    rule appends IDs until every unordered pair occurs in adjacent positions.
    For the four-signal example this returns ``(0,1,2,3,0,2,3,1)``.
    """

    if isinstance(num_channels, bool) or not isinstance(num_channels, int):
        raise TypeError("num_channels must be an integer")
    if num_channels < 1:
        raise ValueError("an Activity Graph requires at least one signal")
    if num_channels == 1:
        return (0,)

    # 通道 ID 从 0 开始；不断扩展序列，直到每对不同通道都曾相邻出现。
    order = list(range(num_channels))
    covered = {
        _edge(left, right)
        for left, right in zip(order, order[1:])
        if left != right
    }
    target_count = num_channels * (num_channels - 1) // 2

    def pair_is_covered(left: int, right: int) -> bool:
        return left == right or _edge(left, right) in covered

    def append(signal_id: int) -> None:
        if not 0 <= signal_id < num_channels:
            raise RuntimeError(
                "paper Algorithm 1 produced an out-of-range signal index"
            )
        previous = order[-1]
        order.append(signal_id)
        if previous != signal_id:
            covered.add(_edge(previous, signal_id))

    # Literal zero-based transcription of Algorithm 1. Keeping this scan rule,
    # rather than substituting another Euler walk, is necessary to reproduce
    # the published four-signal sequence and its three-column layout.
    current = num_channels - 1
    candidate = 0
    while len(covered) < target_count:
        if current == candidate:
            candidate += 1
            if candidate >= num_channels:
                candidate -= 2
                append(candidate)
                current = candidate
                candidate = 0

        if pair_is_covered(current, candidate):
            candidate += 1
            if candidate >= num_channels:
                current += 1
                candidate = 0
                while pair_is_covered(current, candidate):
                    candidate += 1
                    if current == candidate:
                        candidate += 1
                    if candidate >= num_channels:
                        current += 1
                        break
                candidate = 0
                append(current)
        else:
            append(candidate)
            current = candidate
            candidate = 0

    return tuple(order)


def paper_multicolumn_indices(num_channels: int) -> tuple[tuple[int, int, int], ...]:
    """Return paper Algorithm 3 rows as ``(previous, current, next)`` IDs."""

    # 每行按前一、当前、后一通道排三列，首尾按循环序列连接。
    order = paper_signal_order(num_channels)
    return tuple(
        (order[row - 1], signal_id, order[(row + 1) % len(order)])
        for row, signal_id in enumerate(order)
    )


def _coerce_valid_lengths(
    valid_lengths: Optional[torch.Tensor],
    batch_size: int,
    time_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_lengths is None:
        return torch.full(
            (batch_size,), time_steps, dtype=torch.long, device=device
        )
    lengths = torch.as_tensor(valid_lengths, device=device)
    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ValueError(f"valid_lengths must have shape [{batch_size}]")
    if lengths.dtype == torch.bool or lengths.is_floating_point():
        raise ValueError("valid_lengths must use an integer dtype")
    lengths = lengths.to(dtype=torch.long)
    if bool(((lengths < 1) | (lengths > time_steps)).any()):
        raise ValueError(f"valid_lengths entries must lie in [1, {time_steps}]")
    return lengths


def _normalize_waveform_lanes(
    signals: torch.Tensor,
    valid_lengths: torch.Tensor,
    vertical_margin: float,
) -> torch.Tensor:
    """Map valid signal lanes to plot bounds while ignoring padded tails.

    The paper does not report plotting bounds. Per-signal min/max scaling with
    a small plotting margin is the explicit deterministic choice used here to
    emulate independent waveform axes; it is not presented as a paper formula.
    """

    time_steps = signals.shape[-1]
    sample_positions = torch.arange(time_steps, device=signals.device)
    valid = sample_positions[None, None, :] < valid_lengths[:, None, None]
    finite = torch.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    # 每个样本、每个通道独立缩放；常值通道置于中线，填充段不影响范围。
    minimum = finite.masked_fill(~valid, float("inf")).amin(dim=-1, keepdim=True)
    maximum = finite.masked_fill(~valid, float("-inf")).amax(dim=-1, keepdim=True)
    span = maximum - minimum
    unit = (finite - minimum) / span.clamp_min(1e-8)
    unit = torch.where(span <= 1e-8, torch.full_like(unit, 0.5), unit)
    unit = unit.clamp(0.0, 1.0)
    scaled = vertical_margin + (1.0 - 2.0 * vertical_margin) * unit
    return scaled.masked_fill(~valid, 0.5)


def render_paper_activity_graph(
    signals: torch.Tensor,
    *,
    valid_lengths: Optional[torch.Tensor] = None,
    output_size: int = 224,
    canvas_size: int = PAPER_ACTIVITY_GRAPH_CANVAS_SIZE,
    line_width: float = 1.0,
    vertical_margin: float = 0.05,
) -> torch.Tensor:
    """Differentiably draw paper Algorithm 1+3 waveform graphs.

    ``canvas_size=360`` reproduces the paper's reported square raster. The
    result is resized only at the visual-backbone boundary when ``output_size``
    differs from 360.
    """

    # 绘图步骤：
    # 第 1 步：每个通道的波形按自己的最小/最大值缩放到泳道高度内（断点 D10 在 layout 这一行）。
    output_dtype = signals.dtype
    # 半精度输入也用 float32 计算栅格距离，最终图像恢复输入 dtype。
    working_dtype = torch.float64 if signals.dtype == torch.float64 else torch.float32
    working = signals.to(dtype=working_dtype)
    batch_size, channels, time_steps = working.shape
    lengths = _coerce_valid_lengths(
        valid_lengths, batch_size, time_steps, working.device
    )
    normalized = _normalize_waveform_lanes(
        working, lengths, float(vertical_margin)
    )

    # 第 2 步：生成论文的行布局：33 个通道按算法 1 排成 543 行，每行三列（前一/当前/后一通道）。
    layout = torch.tensor(
        paper_multicolumn_indices(channels),
        dtype=torch.long,
        device=working.device,
    )
    row_count = layout.shape[0]
    # Rasterize every original sample-to-sample segment. Drawing only a
    # resampled value at each image column can erase narrow spikes and leaves
    # steep waveforms disconnected. At least eight vertical samples per row
    # retain all Algorithm 1 rows before reducing the reference canvas.
    # 第 3 步：确定每条泳道的像素尺寸：宽 = 画布宽/3 = 120，高 = max(8, 360/543) = 8；
    # 并把每个采样点换算成泳道内的像素坐标 (x, y)。
    panel_width = canvas_size // 3
    lane_height = max(8, math.ceil(canvas_size / row_count))
    x_grid = torch.arange(panel_width, device=working.device, dtype=working.dtype)
    y_grid = torch.arange(lane_height, device=working.device, dtype=working.dtype)
    grid_x = x_grid.view(1, 1, 1, panel_width, 1)
    grid_y = y_grid.view(1, 1, lane_height, 1, 1)
    samples_x = torch.arange(time_steps, device=working.device, dtype=working.dtype)
    samples_x = samples_x[None, :] * (panel_width - 1) / (lengths[:, None] - 1).clamp_min(1)
    samples_y = (1.0 - normalized) * (lane_height - 1)
    # A one-sample record is a horizontal constant trace across the lane.
    if time_steps == 1:
        samples_x = torch.stack((samples_x[:, 0], samples_x[:, 0] + panel_width - 1), -1)
        samples_y = samples_y.expand(-1, -1, 2)
    else:
        singleton = lengths == 1
        samples_x = samples_x.clone()
        samples_x[:, 1] = torch.where(singleton, panel_width - 1, samples_x[:, 1])
        samples_y = samples_y.clone()
        samples_y[:, :, 1] = torch.where(singleton[:, None], samples_y[:, :, 0], samples_y[:, :, 1])
    # 第 4 步：对每条泳道的每个像素，求它到折线（相邻采样点连成的线段）的最短距离（断点 D11 在 sigma 这一行）。
    distance_squared = None
    segment_count = samples_x.shape[-1] - 1
    # 每批处理 32 条原始采样线段，累积像素到有效线段的最短距离以控制内存。
    for start in range(0, segment_count, 32):
        stop = min(start + 32, segment_count)
        ax = samples_x[:, start:stop].view(batch_size, 1, 1, 1, -1)
        ay = samples_y[:, :, start:stop].unsqueeze(2).unsqueeze(2)
        dx = (samples_x[:, start + 1:stop + 1] - samples_x[:, start:stop]).view(batch_size, 1, 1, 1, -1)
        dy = (samples_y[:, :, start + 1:stop + 1] - samples_y[:, :, start:stop]).unsqueeze(2).unsqueeze(2)
        projection = ((grid_x - ax) * dx + (grid_y - ay) * dy) / (dx.square() + dy.square()).clamp_min(1e-12)
        projection = projection.clamp(0.0, 1.0)
        distances = (grid_x - ax - projection * dx).square() + (grid_y - ay - projection * dy).square()
        segment_ids = torch.arange(start, stop, device=working.device)
        valid_segments = segment_ids[None, :] < (lengths - 1).clamp_min(1)[:, None]
        distances = distances.masked_fill(~valid_segments[:, None, None, None, :], float("inf"))
        nearest = distances.amin(dim=-1)
        distance_squared = nearest if distance_squared is None else torch.minimum(distance_squared, nearest)
    # 距离经连续函数转为黑线白底像素；此处保留波形到图像的自动求导路径。
    # 第 5 步：距离越近像素越亮，用高斯函数把距离变成 0–1 的像素值，得到 33 条泳道图 [M,33,8,120]。
    sigma = max(float(line_width) / 2.354820045, 0.25)
    lanes = 1.0 - torch.exp(-0.5 * distance_squared / sigma**2)
    # 将通道波形按论文的行顺序和三列布局拼接，最后复制灰度通道得到 RGB。
    # 第 6 步：按布局把泳道拼成 543 行 × 3 列的灰度大图 [M,1,4344,360]（断点 D12 在 if gray.shape 这一行）。
    panels = lanes.index_select(1, layout.reshape(-1)).reshape(
        batch_size, row_count, 3, lane_height, panel_width
    )
    gray = panels.permute(0, 1, 3, 2, 4).reshape(
        batch_size, 1, row_count * lane_height, canvas_size
    )
    # 第 7 步：面积平均缩到 360×360，复制成 3 个颜色通道，再双线性缩到 224×224（断点 D13 在 return 这一行）。
    if gray.shape[-2] != canvas_size:
        gray = F.interpolate(gray, size=(canvas_size, canvas_size), mode="area")
    image = gray.expand(-1, 3, -1, -1)
    if output_size != canvas_size:
        image = F.interpolate(
            image, size=(output_size, output_size), mode="bilinear", align_corners=False
        )
    return image.clamp(0.0, 1.0).to(dtype=output_dtype)

