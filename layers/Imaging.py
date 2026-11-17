# 成像消融：每个原始 patch 渲染一幅图。
import math
import numpy as np
import torch
from PIL import Image, ImageDraw
from layers.ActivityGraph import paper_signal_order
REPRESENTATIONS = ('ordinary_line', 'adaptive_heatmap', 'gaf', 'tivit_grayscale', 'spectrogram')


def valid_window(window, valid_length=None):
    x = torch.as_tensor(window).detach().cpu().double().numpy()
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError('Expected nonempty [channels,time]')
    n = x.shape[1] if valid_length is None else int(valid_length)
    if not 1 <= n <= x.shape[1]:
        raise ValueError('Invalid valid_length')
    x = x[:, :n].copy()
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite valid samples')
    return x


def scale_channels(x):
    low=x.min(axis=1,keepdims=True); span=x.max(axis=1,keepdims=True)-low
    return np.divide(x-low,span,out=np.full_like(x,.5),where=span>0)


def dominant_period(x):
    x=np.asarray(x,dtype=np.float64)
    if x.shape[-1] < 2:
        return 1
    amplitude=np.abs(np.fft.rfft(x-x.mean(axis=-1,keepdims=True),axis=-1)).mean(axis=0)
    amplitude[0]=0
    if not np.any(amplitude[1:]>0):
        return x.shape[-1]
    index=int(np.argmax(amplitude[1:]))+1
    return max(1,x.shape[-1]//index)


def folded_channels(x):
    period=dominant_period(x)
    n=x.shape[-1]; tail=(-n)%period
    padded=np.pad(x,((0,0),(0,tail)))
    mask=np.pad(np.ones_like(x,dtype=bool),((0,0),(0,tail)),constant_values=False)
    return padded.reshape(x.shape[0],-1,period),mask.reshape(x.shape[0],-1,period),period


def gasf_channels(x):
    z=np.clip(2*scale_channels(x)-1,-1,1)
    sine=np.sqrt(np.maximum(0,1-z*z))
    return np.clip(z[:,:,None]*z[:,None,:]-sine[:,:,None]*sine[:,None,:],-1,1)


def spectrogram_channels(x, segment=16):
    """逐通道短时傅里叶幅度谱：Hann 窗、窗长 min(16, T)、步长为窗长的 1/4，每帧先减均值。

    x: [channels, T] 有效前缀。返回 [channels, 频率, 帧] 的 log(1+幅度)，低频在下方，按通道缩放到 [0, 1]。
    """
    n=x.shape[1]; length=min(segment,n); hop=max(1,length//4)
    starts=range(0,n-length+1,hop)
    frames=np.stack([x[:,s:s+length] for s in starts],axis=1)          # [C,帧,窗长]
    frames=(frames-frames.mean(axis=-1,keepdims=True))*np.hanning(length)
    power=np.log1p(np.abs(np.fft.rfft(frames,axis=-1)))                  # [C,帧,频率]
    matrices=power.transpose(0,2,1)[:,::-1]                              # [C,频率(高->低),帧]
    return scale_channels(matrices.reshape(len(x),-1)).reshape(matrices.shape)


def tivit_gray_channels(x, stride=0.1, patch_size=None, gamma=0.8):
    """TiViT (upstream 5faafcd, src/tivit.py ts2image_transformation) before its final resize.

    x: [channels, T] valid prefix. Returns [channels, windows, patch] in [0, 1].
    Each channel is its own univariate series, exactly as TiViT embeds channels separately.
    默认参数即 TiViT 原样；stride（窗长倍数）、patch_size（窗长，默认 √T）、gamma（灰度幂次）仅供灰度图参数分析。
    """
    import torch.nn.functional as F
    t = torch.as_tensor(np.asarray(x, dtype=np.float32).T[None])      # B x T x D, B = 1
    median = t.median(1, keepdim=True)[0]
    q = torch.tensor([0.75, 0.25], dtype=t.dtype)
    q75, q25 = torch.quantile(t, q, dim=1, keepdim=True)
    t = (t - median) / (q75 - q25 + 1e-5)
    t = t.permute(0, 2, 1)                                             # b d t
    T = t.shape[-1]
    patch_size = min(patch_size, T) if patch_size else int(math.sqrt(T))
    stride_len = int(patch_size * stride) or 1
    remainder = (T - patch_size) % stride_len
    pad_left = stride_len - remainder if remainder else 0
    x_pad = F.pad(t, (pad_left, 0), mode="replicate")
    x_2d = x_pad.unfold(dimension=2, size=patch_size, step=stride_len)  # b d windows patch
    lo = x_2d.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    hi = x_2d.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    x_2d = torch.pow((x_2d - lo) / (hi - lo + 1e-5), gamma)
    return x_2d[0].double().numpy()


# 面积压缩（area_compress）：与活动图同一条缩放链——高分辨率作图 -> 面积平均到 area_canvas（360）-> 双线性到 224。
# 矩阵类成像按矩阵原分辨率逐元素上色后直接面积平均到格子尺寸（等价于先放大作图再压缩）；
# 折线图没有离散矩阵，按 AREA_SUPERSAMPLE 倍画布作图，线宽与边距按最终 224 图的像素含义同比放大。
AREA_SUPERSAMPLE = 4


def _area_panel_box(ch, rows, columns, final_size, size):
    # 格子位置先按最终 224 图计算（含格间 1 像素白线），再等比放大到 size，布局与默认路径一致、只换缩放方式。
    row, col = divmod(ch, columns)
    s = size / final_size
    return (round(col*final_size//columns*s), round(row*final_size//rows*s),
            round(((col+1)*final_size//columns-1)*s), round(((row+1)*final_size//rows-1)*s))


def _area_matrix_panel(matrix, valid, height, width, colored):
    # 全程浮点，不经 8 位量化；viridis 取 matplotlib 同一 256 级查找表与同一取整规则。
    import torch.nn.functional as F
    from matplotlib import colormaps
    v = np.clip(matrix, 0, 1)
    if colored:
        lut = torch.from_numpy(colormaps['viridis'](np.arange(256))[:, :3].astype(np.float32))
        rgb = lut[torch.from_numpy(np.minimum((v * 256).astype(np.int64), 255))]
        rgb[torch.from_numpy(~valid)] = .5
        image = rgb.permute(2, 0, 1)
    else:
        image = torch.from_numpy(v.astype(np.float32))[None]
    panel = F.interpolate(image[None], size=(height, width), mode='area')[0]
    return panel.expand(3, -1, -1)


def _area_line_canvas(unit, rows, columns, size, final_size, line_width, panel_gap, overlay):
    # 与 render_image 的折线分支同一几何，只是画布放大到 size；各格单独作图再粘贴，粗线不会越界到相邻格。
    scale = size / final_size
    width_px = max(1, round(line_width * scale))
    canvas = Image.new('RGB', (size, size), 'white')
    n = unit.shape[1]
    for ch in range(len(unit)):
        left, top, right, bottom = _area_panel_box(ch, rows, columns, final_size, size)
        if overlay:
            left, top = 0, 0; right, bottom = size, size
        width, height = right-left, bottom-top
        panel = canvas if overlay else Image.new('RGB', (width, height), 'white')
        mx, my = round(panel_gap*width/2), round(panel_gap*height/2)
        ys = my + (1-unit[ch])*(height-3*scale-2*my) + scale
        xs = np.linspace(mx, width-1-mx, n) if n > 1 else np.array([mx, width-1-mx])
        ys = ys if n > 1 else np.repeat(ys, 2)
        ImageDraw.Draw(panel).line([(float(a), float(b)) for a, b in zip(xs, ys)], fill=(31, 119, 180),
                                   width=width_px, joint='curve')
        if not overlay:
            canvas.paste(panel, (left, top))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255).permute(2, 0, 1)


def _area_render(image_representation, unit, matrices, mask, rows, columns, final_size, area_canvas,
                 line_width, panel_gap, overlay, vertical_margin=0.0):
    import torch.nn.functional as F
    if image_representation == 'ordinary_line':
        big = _area_line_canvas(unit, rows, columns, area_canvas*AREA_SUPERSAMPLE, final_size,
                                line_width, panel_gap, overlay)
        canvas = F.interpolate(big[None], size=(area_canvas, area_canvas), mode='area')[0]
    else:
        canvas = torch.ones(3, area_canvas, area_canvas)
        for ch in range(len(matrices)):
            left, top, right, bottom = _area_panel_box(ch, rows, columns, final_size, area_canvas)
            inset = round(vertical_margin*(bottom-top))
            canvas[:, top+inset:bottom-inset, left:right] = _area_matrix_panel(
                matrices[ch], mask[ch], bottom-top-2*inset, right-left, image_representation != 'tivit_grayscale')
    # 与 render_paper_activity_graph 第 7 步相同：双线性（align_corners=False）缩到最终尺寸。
    image = F.interpolate(canvas[None], size=(final_size, final_size), mode='bilinear', align_corners=False)[0]
    return image.clamp(0.0, 1.0)


# 成像参数实验（area_exact，仅矩阵类）：格子、外框与子图按连续坐标排布，画布每个像素按各矩形的精确覆盖面积加权
# （可分离：A_行 · 颜色 · A_列ᵀ，不经超采样）。参数全默认时版式与 _area_render 相同；差别只在于官方
# F.interpolate(mode='area') 对跨像素边界的元素等权平均，这里按覆盖面积加权，所以实验另设"精确路径、参数全默认"的对照组。
# 间距、外框、拉伸只改排布，不改矩阵数值。
def _coverage(starts, stops, size):
    p = np.arange(size, dtype=np.float64)[:, None]
    return np.clip(np.minimum(p+1, np.asarray(stops)[None]) - np.maximum(p, np.asarray(starts)[None]), 0, None)


def _weighted(a_r, colors, a_c):
    # a_r · colors[k] · a_cᵀ（k 为颜色通道）。覆盖矩阵每行只有少数非零，用稀疏乘法；结果与稠密乘法只差浮点求和顺序。
    from scipy import sparse
    r, c = sparse.csr_matrix(a_r), sparse.csr_matrix(a_c)
    return np.stack([(c @ (r @ k).T).T for k in colors])


def _cell_edges(start, length, n, gap, frame):
    # 一维排布 n 个格子：gap 为相邻格子之间的空白占格子边长的比例（首尾不留）；frame 为矩阵外框粗细（画布像素），格子在框内均分。
    cell = (length-2*frame) / (n+(n-1)*gap)
    if cell <= 0:
        raise ValueError('matrix frame leaves no room for cells')
    starts = start + frame + np.arange(n)*cell*(1+gap)
    return starts, starts+cell


def _exact_render(image_representation, matrices, mask, rows, columns, final_size, area_canvas, vertical_margin,
                  row_gap, col_gap, frame, aspect, subplot_gap):
    import torch.nn.functional as F
    from matplotlib import colormaps
    lut = colormaps['viridis'](np.arange(256))[:, :3].astype(np.float32)
    s = area_canvas / final_size
    canvas = np.ones((3, area_canvas, area_canvas))
    for ch, (matrix, valid) in enumerate(zip(matrices, mask)):
        if subplot_gap is None:  # 原始子图间距：格间 1 像素白线（最终 224 图），与 _area_panel_box 相同
            left, top, right, bottom = _area_panel_box(ch, rows, columns, final_size, area_canvas)
            inset = round(vertical_margin*(bottom-top))
        else:  # 子图间距 = subplot_gap × 子图节距（横纵各按本方向节距），留在每格右侧与下方，与原 1 像素白线位置一致
            row, col = divmod(ch, columns)
            px, py = final_size/columns, final_size/rows
            left, top = col*px*s, row*py*s
            right, bottom = ((col+1)*px - subplot_gap*px)*s, ((row+1)*py - subplot_gap*py)*s
            inset = vertical_margin*(bottom-top)
        top, bottom = top+inset, bottom-inset
        width, height = right-left, bottom-top
        # 拉伸：矩阵区域宽高比 = aspect × 原宽高比，放进本格并居中，其余留白。
        if aspect > 1:
            top, height = top+(height-height/aspect)/2, height/aspect
        elif aspect < 1:
            left, width = left+(width-width*aspect)/2, width*aspect
        v = np.clip(matrix, 0, 1)
        if image_representation != 'tivit_grayscale':
            colors = lut.astype(np.float32)[np.minimum((v*256).astype(np.int64), 255)]  # 查找表本身是 float32，不损失精度
            colors[~valid] = .5
            colors = colors.transpose(2, 0, 1)
        else:
            colors = v[None]  # 灰度：三通道相同，算一次再广播
        t = frame * s
        a_r = _coverage(*_cell_edges(top, height, v.shape[0], row_gap, t), area_canvas)
        a_c = _coverage(*_cell_edges(left, width, v.shape[1], col_gap, t), area_canvas)
        cells = np.outer(a_r.sum(1), a_c.sum(1))
        canvas += _weighted(a_r, colors, a_c) - cells  # [3 或 1, 行, 列]，灰度广播到三通道
        if frame:  # 黑色外框：矩阵区域内不被格子覆盖的部分
            region = np.outer(_coverage([top], [top+height], area_canvas)[:, 0], _coverage([left], [left+width], area_canvas)[:, 0])
            canvas -= region - cells
    canvas = torch.from_numpy(canvas.astype(np.float32))
    image = F.interpolate(canvas[None], size=(final_size, final_size), mode='bilinear', align_corners=False)[0]
    return image.clamp(0.0, 1.0)


def render_image(window, *, valid_length=None, image_representation, image_size=224,
                 line_width=1, render_size=None, panel_gap=0.0, overlay=False, global_scale=False,
                 activity_lane=False, gray_stride=0.1, gray_patch_size=0, gray_gamma=0.8,
                 area_compress=False, area_canvas=360, vertical_margin=0.0, area_exact=False,
                 cell_row_gap=0.0, cell_col_gap=0.0, matrix_frame=0.0, matrix_aspect=1.0, subplot_gap=None):
    # 折线图参数分析：render_size 先按该分辨率作图再双线性缩放到 image_size；panel_gap 为每格四周留白占边长的比例。
    if area_compress and (render_size not in (None, image_size) or activity_lane):
        raise ValueError('area_compress replaces line_plot_render_size and line_plot_activity_lane')
    exact_params = (cell_row_gap, cell_col_gap, matrix_frame, matrix_aspect, subplot_gap) != (0.0, 0.0, 0.0, 1.0, None)
    if (area_exact or exact_params) and not (area_exact and area_compress and image_representation != 'ordinary_line'):
        raise ValueError('cell gaps / frame / aspect / subplot gap need area_exact + area_compress on a matrix imaging')
    final_size, image_size = image_size, render_size or image_size
    from matplotlib import colormaps
    if image_representation not in REPRESENTATIONS:
        raise ValueError(image_representation)
    x=valid_window(window,valid_length)
    channels,n=x.shape
    columns=math.ceil(math.sqrt(channels)); rows=math.ceil(channels/columns)
    if not isinstance(image_size,int) or image_size < max(rows,columns)*3:
        raise ValueError('Image too small for channel grid')
    # 与活动图行高对齐（activity_lane）：活动图每行在最终 224 图里高 band = 224 / 活动图行数 像素，作图时每行 8 像素高、
    # 再面积平均缩小。折线图照做：画布只在纵向拉伸 stretch 倍，使波形带高 8 像素，最后面积平均缩回 final_size。
    canvas_h=image_size
    if activity_lane and image_representation=='ordinary_line':
        band=final_size/len(paper_signal_order(channels))
        canvas_h=round(image_size*max(1.0,8/band))
    canvas=Image.new('RGB',(image_size,canvas_h),'white')
    unit=scale_channels(x)
    # 折线图全局缩放：块内所有通道共用一个最小值和最大值，保留通道间的幅值差异。
    if global_scale and image_representation=='ordinary_line':
        unit=scale_channels(x.reshape(1,-1)).reshape(x.shape)
    # 上下边距（vertical_margin）：与活动图 _normalize_waveform_lanes 相同，波形只占每格中间 1-2m 的高度、左右不留；
    # 矩阵类成像则把矩阵放进每格中间 1-2m 的高度，上下各留白 m。默认 0 与原行为逐像素一致。
    if vertical_margin and image_representation=='ordinary_line':
        unit=vertical_margin+(1-2*vertical_margin)*unit
    if image_representation=='adaptive_heatmap':
        # Determine period on input amplitudes, render channel-normalized values.
        raw,mask,period=folded_channels(x)
        tail=raw.shape[1]*period-n
        matrices=np.pad(unit,((0,0),(0,tail))).reshape(raw.shape)
    elif image_representation=='gaf':
        matrices=(gasf_channels(x)+1)/2
        mask=np.ones_like(matrices,dtype=bool)
    elif image_representation=='tivit_grayscale':
        matrices=tivit_gray_channels(x,stride=gray_stride,patch_size=gray_patch_size or None,gamma=gray_gamma)
        mask=np.ones_like(matrices,dtype=bool)
    elif image_representation=='spectrogram':
        matrices=spectrogram_channels(x)
        mask=np.ones_like(matrices,dtype=bool)
    if area_exact:
        return _exact_render(image_representation, matrices, mask, rows, columns, final_size, area_canvas,
                             vertical_margin, cell_row_gap, cell_col_gap, matrix_frame, matrix_aspect, subplot_gap)
    if area_compress:
        line = image_representation=='ordinary_line'
        return _area_render(image_representation, unit, None if line else matrices, None if line else mask,
                            rows, columns, final_size, area_canvas, line_width, panel_gap, overlay, vertical_margin)
    for ch in range(channels):
        row,col=divmod(ch,columns)
        left=col*image_size//columns; right=(col+1)*image_size//columns-1
        top=row*canvas_h//rows; bottom=(row+1)*canvas_h//rows-1
        # 叠加时各通道共享整幅画布，仍各自按最小值和最大值缩放。
        if overlay and image_representation=='ordinary_line':
            left,top=0,0; right,bottom=image_size,canvas_h
        width,height=right-left,bottom-top
        inset=0 if image_representation=='ordinary_line' else round(vertical_margin*height)
        if image_representation=='ordinary_line':
            panel=canvas if overlay else Image.new('RGB',(width,height),'white')
            draw=ImageDraw.Draw(panel)
            mx, my = round(panel_gap*width/2), round(panel_gap*height/2)
            ys=my+(1-unit[ch])*(height-3-2*my)+1
            # 与活动图行高对齐：波形纵向只占 band 像素（拉伸后的画布上为 band*canvas_h/final_size），居中放在本格。
            if activity_lane:
                lane=band*canvas_h/final_size
                ys=(height-1)/2-lane/2+(1-unit[ch])*lane
            points=[(float(t),float(y)) for t,y in zip(np.linspace(mx,width-1-mx,n),ys)]
            if n==1:
                draw.line([(mx,points[0][1]),(width-1-mx,points[0][1])],fill=(31,119,180),width=line_width)
            else:
                draw.line(points,fill=(31,119,180),width=line_width)
        elif image_representation=='tivit_grayscale':
            rgb=np.repeat(np.clip(matrices[ch],0,1)[...,None],3,axis=-1)
            panel=Image.fromarray(np.rint(rgb*255).astype(np.uint8)).resize((width,height-2*inset),Image.Resampling.NEAREST)
        else:
            rgb=colormaps['viridis'](np.clip(matrices[ch],0,1))[...,:3]
            rgb[~mask[ch]]=.5
            panel=Image.fromarray(np.rint(rgb*255).astype(np.uint8)).resize((width,height-2*inset),Image.Resampling.NEAREST)
        canvas.paste(panel,(left,top+inset))
    if canvas_h != image_size:
        canvas=canvas.resize((final_size,final_size),Image.Resampling.BOX)
    elif image_size != final_size:
        canvas=canvas.resize((final_size,final_size),Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(canvas,copy=True)).permute(2,0,1).float()/255
