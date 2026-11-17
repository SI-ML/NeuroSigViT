"""ViTST（NeurIPS 2023）基线：各通道折线子图按网格拼成一张图，预训练 Swin-B（in22k）换二分类头后全量微调。

成像与训练按官方仓库 Leezekun/ViTST：
  - 成像（dataset/*/process_scripts/ConstructImage.py）：网格 ceil(sqrt(C)) 行，默认每通道 64x64 子图；
    官方允许另设图像尺寸（P12/P19 用 384），这里每边取 max(224, 格数x64)，保证 Swin 第 4 阶段不小于 7x7 窗口；线型 "-"、线宽 1、
    标记 "*"、标记大小 2，各通道不同颜色（官方颜色表）；x 轴 [1,T]，y 轴为训练集该通道的 [min,max]，超出范围的点不画；
    无坐标刻度，子图间距 0。数据是稠密序列，不沿用官方把 0 视为缺失值的规则。
  - 输入（code/Vision/run_ImgCLS.py）：原尺寸图像 ToTensor 到 [0,1]，不缩放、不做 ImageNet 归一化；
    Swin 用 ViTST 修改版（code/models/swin），image_size 设为图像尺寸。
  - 训练：AdamW（lr 2e-5，wd 0），warmup 10% 后线性衰减，梯度裁剪 1.0，最多 20 轮，早停 patience 5；
    二分类训练集少数类上采样到与多数类等量、正负样本交替（官方 upsample）。
    EEG 有效批量 192（48x4，同官方）；可穿戴数据训练集只有 69 / 212 个样本，有效批量改为 8，否则 20 轮只更新约 20 次。
划分与评估与主方法相同：固定受试者划分；每轮在验证集上算宏平均 F1，取最高的一轮（相同取更早）；
测试集只在恢复最佳权重后评估一次。缺少验证集时直接报错，不回退到测试集。
"""
import argparse
import importlib
import json
import math
import os
import random
import sys
import time
import types
from itertools import chain
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_provider.data_factory import load_data  # noqa: E402
from exp.exp_baselines import aggregate_subjects, metrics, seed_training  # noqa: E402

# 官方仓库 Leezekun/ViTST 与 Swin-B（in22k）权重的位置，可用环境变量覆盖。
VITST = Path(os.environ.get("VITST_REPO", Path.home() / "ViTST"))
WEIGHTS = Path(os.environ.get("VITST_SWIN_WEIGHTS", Path.home() / "pretrained_baselines" / "swin-base-patch4-window7-224-in22k"))
# 官方颜色表（ConstructImage.py 的 color_detailed_description，38 种）
COLORS = ["green", "black", "blue", "brown", "chartreuse", "chocolate", "coral", "crimson", "blueviolet", "darkblue",
          "darkgreen", "firebrick", "gold", "teal", "grey", "indigo", "steelblue", "indianred", "goldenrod", "darkred",
          "darkorange", "magenta", "maroon", "navy", "olive", "orange", "orchid", "pink", "plum", "purple", "red",
          "cornflowerblue", "sienna", "darkkhaki", "tan", "dodgerblue", "darkseagreen", "cadetblue"]
TRAIN = {"apava": (48, 4), "tdbrain": (48, 4), "shimmer10": (8, 1), "pads11": (8, 1)}  # (每步批量, 梯度累积)


def grid(channels):
    rows = math.ceil(math.sqrt(channels))
    return rows, math.ceil(channels / rows)


def image_size(channels):
    rows, cols = grid(channels)
    return max(224, rows * 64), max(224, cols * 64)


def _render(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, scales = job
    channels, length = x.shape
    rows, cols = grid(channels)
    height, width = image_size(channels)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, frameon=False)
    t = np.arange(1, length + 1)
    for c in range(channels):
        low, high = scales[c]
        if low == high:
            low, high = low - 0.5, high + 0.5
        keep = (x[c] >= low) & (x[c] <= high)
        ax = fig.add_subplot(rows, cols, c + 1)
        ax.plot(t[keep], x[c][keep], linestyle="-", linewidth=1, marker="*", markersize=2, color=COLORS[c])
        ax.set_xlim(1, length)
        ax.set_ylim(low, high)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def render_dataset(key, bundle, cache_dir, workers):
    path = cache_dir / f"{key}.npz"
    if path.exists():
        data = np.load(path)
        return {s: data[s] for s in ("train", "vali", "test")}
    x = {s: getattr(bundle, s + "_loader").dataset.tensors[0].numpy() for s in ("train", "vali", "test")}
    train = x["train"]
    scales = np.stack([train.min(axis=(0, 2)), train.max(axis=(0, 2))], axis=1)  # [C,2]，仅用训练集
    images = {}
    with Pool(workers) as pool:
        for split in ("train", "vali", "test"):
            images[split] = np.stack(pool.map(_render, [(w, scales) for w in x[split]], chunksize=8))
            print(key, split, images[split].shape, flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / f"{key}.tmp.npz"
    np.savez(temporary, **images)
    temporary.rename(path)
    return images


def load_swin_class():
    # ViTST 的 swin 目录用相对导入，且与本仓库的 models 包同名；这里注册为独立包名再导入。
    root = VITST / "code" / "models"
    for name, path in (("vitst_models", root), ("vitst_models.swin", root / "swin")):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    return importlib.import_module("vitst_models.swin.modeling_swin").SwinForImageClassification


def upsample(labels, rng):
    # 官方 get_data_split(upsample=True)：少数类有放回抽到与多数类等量，两类各自打乱后交替排列。
    idx0, idx1 = list(np.flatnonzero(labels == 0)), list(np.flatnonzero(labels == 1))
    if len(idx0) > len(idx1):
        idx1 = rng.choices(idx1, k=len(idx0))
    else:
        idx0 = rng.choices(idx0, k=len(idx1))
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    return np.asarray(list(chain.from_iterable(zip(idx0, idx1))))


def to_pixels(images, device):
    return torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float().div_(255)


@torch.no_grad()
def predict(model, images, device, batch=128):
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        out.append(torch.softmax(model(pixel_values=to_pixels(images[i:i + batch], device)).logits.float(), dim=-1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def evaluate(probabilities, labels, subjects):
    result = metrics(labels, probabilities)
    agg = aggregate_subjects(labels, probabilities, subjects)
    result.update({"subject_" + k: v for k, v in metrics(agg["subject_y_true"], agg["subject_y_score"]).items()})
    return result


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", choices=sorted(TRAIN), required=True)
    cli.add_argument("--seed", type=int, required=True)
    cli.add_argument("--result_root", default="main_runs/results")
    cli.add_argument("--image_cache", default="work/baselines_20261010/vitst_images")
    cli.add_argument("--run_name", default="vitst-official_20261010")
    cli.add_argument("--epochs", type=int, default=20)
    cli.add_argument("--patience", type=int, default=5)
    cli.add_argument("--lr", type=float, default=2e-5)
    cli.add_argument("--micro_batch", type=int, default=24)
    cli.add_argument("--workers", type=int, default=24)
    cli.add_argument("--checkpoint_root", default="main_runs/checkpoints")
    cli.add_argument("--render_only", action="store_true", help="只生成并缓存图像，不训练")
    opt = cli.parse_args()
    start = time.time()
    loaded = load_data(opt.dataset)
    bundle = loaded[0] if isinstance(loaded, tuple) else loaded
    for split in ("train", "vali", "test"):
        if getattr(bundle, split + "_loader", None) is None:
            raise ValueError(f"missing {split} split")
    raw = {s: np.asarray(getattr(bundle, s + "_labels")) for s in ("train", "vali", "test")}
    classes = np.unique(raw["train"])
    assert len(classes) == 2, classes
    labels = {s: np.searchsorted(classes, raw[s]).astype(np.int64) for s in raw}
    subjects = {s: np.asarray(getattr(bundle, s + "_loader").dataset.sample_subject_ids) for s in raw}
    channels = getattr(bundle, "train_loader").dataset.tensors[0].shape[1]
    images = render_dataset(opt.dataset, bundle, Path(opt.image_cache), opt.workers)
    if opt.render_only:
        return
    height, width = images["train"].shape[1:3]

    seed_training(opt.seed)
    rng = random.Random(opt.seed)
    device = torch.device("cuda")
    model = load_swin_class().from_pretrained(str(WEIGHTS), num_labels=2, ignore_mismatched_sizes=True,
                                              image_size=(height, width)).to(device)
    batch, accum = TRAIN[opt.dataset]
    micro = min(opt.micro_batch, batch)
    micro_steps = accum * batch // micro  # 显存限制下拆成更小的微批，有效批量不变
    order = upsample(labels["train"], rng)
    per_step = micro * micro_steps
    steps_per_epoch = math.ceil(len(order) / per_step)
    total = steps_per_epoch * opt.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(0.1 * total), total)
    best, best_epoch, best_state, history, bad = -1.0, -1, None, [], 0
    for epoch in range(1, opt.epochs + 1):
        model.train()
        perm = order[torch.randperm(len(order)).numpy()]
        losses = []
        for step in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            chunk = perm[step * per_step:(step + 1) * per_step]
            for i in range(0, len(chunk), micro):
                idx = chunk[i:i + micro]
                y = torch.from_numpy(labels["train"][idx]).to(device)
                loss = model(pixel_values=to_pixels(images["train"][idx], device), labels=y).loss
                (loss * len(idx) / len(chunk)).backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        val = evaluate(predict(model, images["vali"], device), labels["vali"], subjects["vali"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val["macro_f1"],
                        "val_macro_auroc": val["macro_auroc"]})
        print(f"Epoch {epoch}/{opt.epochs} | CE={np.mean(losses):.6f} | val_macro_f1={val['macro_f1']:.6f}", flush=True)
        if val["macro_f1"] > best:
            best, best_epoch, bad = val["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= opt.patience:
                break
    model.load_state_dict(best_state)
    result_dir = ROOT / opt.result_root / f"icimh4{opt.dataset}_{opt.run_name}" / f"seed{opt.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    # 最佳验证轮的全部权重（Swin-B + 二分类头），目录结构与主方法相同：checkpoints/<实验>/seed<k>/
    checkpoint_dir = ROOT / opt.checkpoint_root / f"icimh4{opt.dataset}_{opt.run_name}" / f"seed{opt.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "best_epoch": best_epoch, "image_size": [int(height), int(width)],
                "classes": classes.tolist(), "args": vars(opt)}, checkpoint_dir / "best.pt")
    final = {"best_epoch": best_epoch, "image_size": [int(height), int(width)], "grid": list(grid(channels)),
             "effective_batch": batch * accum, "train_after_upsample": int(len(order)),
             "steps_per_epoch": steps_per_epoch, "history": history, "args": vars(opt)}
    for split, key in (("vali", "validation_metrics"), ("test", "test_metrics")):
        p = predict(model, images[split], device)
        final[key] = evaluate(p, labels[split], subjects[split])
        np.savez(result_dir / f"{split}_predictions.npz", y_true=labels[split], y_score=p, subject=subjects[split])
    final["seconds"] = round(time.time() - start, 1)
    (result_dir / "metrics.json").write_text(json.dumps(final, indent=1, default=str))
    shown = {k: round(v, 4) for k, v in final["test_metrics"].items() if v is not None and not k.startswith("subject")}
    print("TEST", json.dumps(shown), flush=True)


if __name__ == "__main__":
    main()
