import argparse
from pathlib import Path
import runpy
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="NeuroSigVIA ICIMH")
    parser.add_argument("--icimh_branches", choices=["both", "visual", "numeric"], default="both")
    # 单数值分支（--icimh_branches numeric）时 Mantis 逐通道特征 [C,256] 如何进分类头：pool 与主方法相同（通道注意力池化，默认）；
    # concat 直接拼接 C*256；mean 每通道同一个线性投影到 fusion_dim 后沿通道取平均。
    parser.add_argument("--numeric_only_reduce", choices=["pool", "concat", "mean"], default="pool")
    parser.add_argument("--vision_backbone", choices=["clip", "siglip2", "dinov3", "swin"], default="clip")
    parser.add_argument("--vit_1_name", default="./pretrained/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--vit_1_layer", type=int, default=14)
    parser.add_argument("--mantis_name", default="./pretrained/Mantis-8M")
    # 数值骨干替换：mantis（默认）/ tspulse / zeus；权重路径仍用 --mantis_name 传入。
    parser.add_argument("--numeric_backbone", choices=["mantis", "tspulse", "zeus"], default="mantis")
    # Zeus 取哪个阶段的隐状态（1–5，对应尺度 1/8/32/8/1）；默认 4 = 论文附录 A.2 的倒数第二个尺度。
    parser.add_argument("--zeus_stage", type=int, choices=[1, 2, 3, 4, 5], default=4)
    parser.add_argument("--outer_patch_size", type=int, default=64)
    parser.add_argument("--outer_patch_stride", type=int, default=64)
    parser.add_argument("--visual_encode_batch_size", type=int, default=32)
    parser.add_argument("--image_representation", default="raw_activity_graph",
                        choices=["raw_activity_graph", "ordinary_line", "gaf", "adaptive_heatmap", "tivit_grayscale",
                                 "spectrogram"])
    parser.add_argument("--activity_graph_canvas_size", type=int, default=360)
    parser.add_argument("--activity_graph_line_width", type=float, default=1.0)
    parser.add_argument("--line_plot_width", type=int, default=1)
    parser.add_argument("--line_plot_render_size", type=int, default=224)
    parser.add_argument("--line_plot_gap", type=float, default=0.0)
    parser.add_argument("--line_plot_overlay", action="store_true")
    parser.add_argument("--line_plot_global_scale", action="store_true")
    parser.add_argument("--line_plot_activity_lane", action="store_true")
    # TiViT 灰度图参数分析：滑窗步长（窗长倍数）、窗长（0 = √T）、灰度幂次；默认即 TiViT 原样。
    parser.add_argument("--gray_stride", type=float, default=0.1)
    parser.add_argument("--gray_patch_size", type=int, default=0)
    parser.add_argument("--gray_gamma", type=float, default=0.8)
    # 面积压缩：非活动图成像也走"高分辨率作图 -> 面积平均到 activity_graph_canvas_size -> 双线性到 224"（见 layers/Imaging.py）。
    parser.add_argument("--image_area_compress", action="store_true")
    # 非活动图成像的每格上下边距（占格高比例，与 --activity_graph_vertical_margin 含义相同）；默认 0 即原行为。
    parser.add_argument("--image_vertical_margin", type=float, default=0.0)
    # 成像参数实验（矩阵类，需 --image_area_compress）：精确面积覆盖路径及其排布参数，默认值即关闭（见 layers/Imaging.py 的 _exact_render）。
    parser.add_argument("--image_area_exact", action="store_true")
    parser.add_argument("--image_cell_row_gap", type=float, default=0.0)  # 相邻格子行间空白 / 格子高
    parser.add_argument("--image_cell_col_gap", type=float, default=0.0)  # 相邻格子列间空白 / 格子宽
    parser.add_argument("--image_matrix_frame", type=float, default=0.0)  # 每个矩阵的黑色外框粗细（最终 224 图像素）
    parser.add_argument("--image_matrix_aspect", type=float, default=1.0)  # 矩阵区域宽高比相对原值的倍数
    parser.add_argument("--image_subplot_gap", type=float, default=None)  # 子图间距 / 子图节距；不设 = 原 1 像素白线
    parser.add_argument("--activity_graph_vertical_margin", type=float, default=0.05)
    parser.add_argument("--fusion_dim", type=int, default=128)
    parser.add_argument("--fusion_heads", type=int, default=2)
    parser.add_argument("--activity_graph_granularity_hidden_dim", type=int, default=64)
    parser.add_argument("--temporal_visual_fusion", default="concat_attn",
                        choices=["concat_attn", "concat", "cross_attn_gate", "masked_pretrain"])
    parser.add_argument("--mask_prob", type=float, default=0.3)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_num_layers", type=int, default=2)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--mlp_lr", type=float, default=1e-4)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-3)
    parser.add_argument("--mlp_class_weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--mlp_epochs", type=int, default=100)
    parser.add_argument("--mlp_early_stop_patience", type=int, default=12)
    parser.add_argument("--mlp_early_stop_strategy", default="raw_primary",
                        choices=["raw_selection_key", "raw_primary", "ema_primary"])
    parser.add_argument("--mlp_early_stop_warmup_epochs", type=int, default=10)
    parser.add_argument("--mlp_early_stop_min_epochs", type=int, default=0)
    parser.add_argument("--mlp_early_stop_ema_decay", type=float, default=0.6)
    parser.add_argument("--mlp_early_stop_min_delta", type=float, default=0.002)
    parser.add_argument("--mlp_lr_scheduler", choices=["none", "reduce_on_plateau"], default="reduce_on_plateau")
    parser.add_argument("--mlp_lr_warmup_epochs", type=int, default=0)
    parser.add_argument("--mlp_lr_scheduler_patience", type=int, default=4)
    parser.add_argument("--mlp_lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--mlp_lr_scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--patch_checkpoint_metric", default="window_macro_f1",
                        choices=["auto", "subject_macro_f1", "window_macro_f1"])
    parser.add_argument("--datasets", choices=["eeg", "wearable"])
    parser.add_argument("--dataset_names", nargs="+",
                        choices=["APAVA", "TDBRAIN", "Shimmer_10_session10_AFC", "PADS_11_task08_TouchIndex"])
    parser.add_argument("--wearable_label_mode", default="original",
                        choices=["original", "shimmer_hc_vs_pd", "pads_pd_vs_hc"])
    parser.add_argument("--eeg_protocol", default="medformer_code_exact", choices=["medformer_code_exact"])
    parser.add_argument("--eeg_normalization", default="per_window_per_channel_standard_scaler_ddof0",
                        choices=["per_window_per_channel_standard_scaler_ddof0"])
    parser.add_argument("--feature_cache_dir")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args(argv)
    # CLIP 不增加参数字段，保持既有命令的 args.json 和缓存键逐字节不变。
    if args.vision_backbone == "clip":
        del args.vision_backbone
    if args.numeric_backbone != "zeus" or args.zeus_stage == 4:
        del args.zeus_stage
    if args.numeric_backbone == "mantis":
        del args.numeric_backbone
    # 一个命令对应一个结果目录，禁止多个数据集静默覆盖同一输出。
    if len(args.dataset_names) != 1:
        parser.error("--dataset_names must select one dataset per run")
    return args


if __name__ == "__main__":
    dispatcher = argparse.ArgumentParser(add_help=False)
    dispatcher.add_argument("--model", default="NeuroSigVIA_ICIMH")
    dispatcher.add_argument("--result_dir", "--output", dest="output", type=Path)
    selected, remaining = dispatcher.parse_known_args()
    if selected.model == "NeuroSigVIA_ICIMH":
        from exp.exp_classification import main
        main(parse_args(remaining + ["--result_dir", str(selected.output)]))
    else:
        module = "exp.exp_tech" if selected.model == "TeCh" else "exp.exp_baselines"
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        # 保留 harness 原输出入口，把最终权重归入与 ICIMH 相同的目录树。
        if "--smoke" not in sys.argv:
            result = selected.output.resolve()
            checkpoint = result.parent.parent.parent / "checkpoints" / result.parent.name / result.name
            checkpoint.mkdir(parents=True, exist_ok=True)
            weight = result / "best_checkpoint.pt"
            target = checkpoint / weight.name
            weight.replace(target)
            weight.symlink_to(target)
