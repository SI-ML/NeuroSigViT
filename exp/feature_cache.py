import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
import torch
from mantis.architecture import Mantis8M
from layers.ActivityGraph import render_paper_activity_graph
from layers.Imaging import render_image
from layers.Patch import make_temporal_patches
from layers.StaticFeatures import _extract_mantis_channel_tokens
from layers.VisualEncoder import get_neurosigvia, _encode_visual_images


def cache_path(args, dataset):
    # 仅包含改变特征或缓存内容的选项；训练 seed 不参与键。
    fields = ("datasets", "data_dir", "wearable_label_mode", "eeg_protocol", "eeg_normalization",
              "icimh_branches", "image_representation", "outer_patch_size", "outer_patch_stride",
              "vit_1_name", "vit_1_layer", "mantis_name", "activity_graph_canvas_size",
              "activity_graph_line_width", "activity_graph_vertical_margin", "visual_encode_batch_size", "batch_size")
    settings = {key: getattr(args, key) for key in fields}
    settings["dataset"] = dataset
    # 折线图参数只在非默认时进入缓存键，保持既有缓存路径不变。
    for name, default in (("line_plot_width", 1), ("line_plot_render_size", 224), ("line_plot_gap", 0.0),
                          ("line_plot_overlay", False), ("line_plot_global_scale", False),
                          ("line_plot_activity_lane", False), ("gray_stride", 0.1), ("gray_patch_size", 0),
                          ("gray_gamma", 0.8), ("image_area_compress", False),
                          ("image_vertical_margin", 0.0), ("image_area_exact", False), ("image_cell_row_gap", 0.0),
                          ("image_cell_col_gap", 0.0), ("image_matrix_frame", 0.0), ("image_matrix_aspect", 1.0),
                          ("image_subplot_gap", None)):
        if getattr(args, name, default) != default:
            settings[name] = getattr(args, name)
    if getattr(args, "vision_backbone", "clip") != "clip":
        settings["vision_backbone"] = args.vision_backbone
    if getattr(args, "numeric_backbone", "mantis") != "mantis":
        settings["numeric_backbone"] = args.numeric_backbone
    if getattr(args, "zeus_stage", 4) != 4:
        settings["zeus_stage"] = args.zeus_stage
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    experiment = "main-patch64" if args.outer_patch_size == 64 else "main-nopatch"
    date = os.environ.get("CACHE_DATE", datetime.now().strftime("%Y%m%d"))
    root = args.feature_cache_dir or f"main_runs/feature_cache/{experiment}_icimh_{date}"
    return Path(root) / dataset / key


def load_encoders(args, device):
    vision, mantis = None, None
    if args.icimh_branches in ("both", "visual"):
        if getattr(args, "vision_backbone", "clip") == "clip":
            vision = get_neurosigvia(args.vit_1_name, args.vit_1_layer, "mean", 1, 1,
                                    image_mode=args.image_representation).to(device)
        else:
            from layers.TimmVisualEncoder import TimmVisualEncoder
            vision = TimmVisualEncoder(args.vision_backbone, args.vit_1_name, args.vit_1_layer).to(device)
        vision.requires_grad_(False).eval()
    if args.icimh_branches in ("both", "numeric"):
        if getattr(args, "numeric_backbone", "mantis") == "mantis":
            mantis = Mantis8M(device=device).from_pretrained(args.mantis_name).to(device)
        else:
            from layers.NumericEncoders import build_numeric_encoder
            mantis = build_numeric_encoder(args.numeric_backbone, args.mantis_name,
                                           zeus_stages=getattr(args, "zeus_stages", (getattr(args, "zeus_stage", 4),))).to(device)
        mantis.requires_grad_(False).eval()
    return vision, mantis


def render_graph_images(windows, lengths, args, device):
    if args.image_representation == "raw_activity_graph":
        return render_paper_activity_graph(
            windows.to(device), valid_lengths=lengths.to(device),
            canvas_size=args.activity_graph_canvas_size, line_width=args.activity_graph_line_width,
            vertical_margin=args.activity_graph_vertical_margin)
    return torch.stack([render_image(window, valid_length=length,
                                    image_representation=args.image_representation,
                                    line_width=args.line_plot_width, render_size=args.line_plot_render_size,
                                    panel_gap=args.line_plot_gap, overlay=args.line_plot_overlay,
                                    global_scale=args.line_plot_global_scale,
                                    activity_lane=args.line_plot_activity_lane, gray_stride=args.gray_stride,
                                    gray_patch_size=args.gray_patch_size, gray_gamma=args.gray_gamma,
                                    area_compress=getattr(args, "image_area_compress", False),
                                    area_canvas=args.activity_graph_canvas_size,
                                    vertical_margin=getattr(args, "image_vertical_margin", 0.0),
                                    area_exact=getattr(args, "image_area_exact", False),
                                    cell_row_gap=getattr(args, "image_cell_row_gap", 0.0),
                                    cell_col_gap=getattr(args, "image_cell_col_gap", 0.0),
                                    matrix_frame=getattr(args, "image_matrix_frame", 0.0),
                                    matrix_aspect=getattr(args, "image_matrix_aspect", 1.0),
                                    subplot_gap=getattr(args, "image_subplot_gap", None))
                        for window, length in zip(windows, lengths)])


@torch.no_grad()
def extract_split(loader, args, vision, mantis, device):
    pieces = {key: [] for key in ("graph_tokens", "mantis_tokens", "patch_mask", "valid_fraction")}
    # 第 1 步：按原始窗口顺序切块；每个原始窗口仍只有一个标签。
    for (inputs,) in loader:
        patches = make_temporal_patches(inputs, args.outer_patch_size, args.outer_patch_stride)
        b, n, c, length = patches.patches.shape
        mask = patches.patch_mask.reshape(-1)
        windows = patches.patches.reshape(b*n, c, length)[mask]
        lengths = patches.valid_lengths.reshape(-1)[mask]
        graph = torch.empty(b, n, 0)
        numeric = torch.empty(b, n, c, 0)
        # 第 2 步：原始块 [M,C,L] 成像，冻结 ViT 输出 [M,Dv]；CLIP/DINOv3=1024，SigLIP2=1152。
        if vision is not None:
            tokens = []
            for start in range(0, len(windows), args.visual_encode_batch_size):
                stop = start + args.visual_encode_batch_size
                images = render_graph_images(windows[start:stop], lengths[start:stop], args, device)
                tokens.append(_encode_visual_images(vision, images, device).cpu())
            encoded = torch.cat(tokens)
            graph = encoded.new_zeros(b*n, encoded.shape[-1])
            graph[mask] = encoded
            graph = graph.reshape(b, n, -1)
        # 第 3 步：逐通道 Mantis 特征 [B,N,C,256]；缓存保持 float32。
        if mantis is not None:
            encoded = _extract_mantis_channel_tokens(
                windows, lengths, mantis, device, args.visual_encode_batch_size)
            numeric = encoded.new_zeros(b*n, c, encoded.shape[-1])
            numeric[mask] = encoded
            numeric = numeric.reshape(b, n, c, -1)
        for key, value in zip(pieces, (graph, numeric, patches.patch_mask, patches.valid_fraction)):
            pieces[key].append(value)
    return {key: torch.cat(value) for key, value in pieces.items()}


def get_features(bundle, args, dataset, device):
    directory = cache_path(args, dataset)
    if directory.exists():
        print(f"Feature cache hit: {directory}", flush=True)
        return {split: torch.load(directory / f"{split}.pt", weights_only=True)
                for split in ("train", "vali", "test")}
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=directory.name + ".tmp-", dir=directory.parent))
    vision, mantis = load_encoders(args, device)
    result = {}
    for split in ("train", "vali", "test"):
        result[split] = extract_split(getattr(bundle, split + "_loader"), args, vision, mantis, device)
        torch.save(result[split], temporary / f"{split}.pt")
        print(f"Cached {dataset}/{split}: " + str({k: list(v.shape) for k, v in result[split].items()}), flush=True)
    temporary.rename(directory)
    del vision, mantis
    torch.cuda.empty_cache()
    return result
