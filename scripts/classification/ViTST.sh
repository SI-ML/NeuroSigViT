#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# 表 1 的 ViTST：各通道折线子图按网格拼成一张图，全量微调 ImageNet-22k 预训练的 Swin-B（实现见 scripts/baselines/vitst.py）。
# 四个数据集统一学习率 5e-5；图像边长取 max(224, 64 x 网格边数)：APAVA 256、TDBRAIN 384、Shimmer 与 PADS 224。
# 需要官方仓库 Leezekun/ViTST（VITST_REPO，默认 ~/ViTST）与 Swin-B 权重（VITST_SWIN_WEIGHTS，
# 默认 ~/pretrained_baselines/swin-base-patch4-window7-224-in22k）。
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_ROOT="${RUN_ROOT:-main_runs}"
PYTHON="${PYTHON:-python}"
LR="${LR:-5e-5}"

run_one() {
  local dataset="$1" seed="$2"
  shift 2
  if [[ " ${RUN_SEEDS:-42 43 44} " != *" $seed "* ]]; then return; fi
  if [[ " ${RUN_DATASETS:-apava tdbrain shimmer10 pads11} " != *" $dataset "* ]]; then return; fi
  local command=(env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$PYTHON" -u scripts/baselines/vitst.py
    --dataset "$dataset"
    --seed "$seed"
    --lr "$LR"
    --run_name "vitst-lr${LR}_${RUN_DATE}"
    --result_root "$RUN_ROOT/results"
    --checkpoint_root "$RUN_ROOT/checkpoints"
    --image_cache "$RUN_ROOT/vitst_images"
    "$@")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

export CUDA_VISIBLE_DEVICES="${GPU_APAVA:-4}"
for seed in 42 43 44; do run_one apava "$seed" "$@"; done
export CUDA_VISIBLE_DEVICES="${GPU_TDBRAIN:-5}"
for seed in 42 43 44; do run_one tdbrain "$seed" "$@"; done
export CUDA_VISIBLE_DEVICES="${GPU_SHIMMER:-6}"
for seed in 42 43 44; do run_one shimmer10 "$seed" "$@"; done
export CUDA_VISIBLE_DEVICES="${GPU_PADS:-7}"
for seed in 42 43 44; do run_one pads11 "$seed" "$@"; done
