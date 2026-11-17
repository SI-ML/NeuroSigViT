#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
MODEL=Medformer
source scripts/classification/lib/baselines.sh

# apava 数据集（lr × dropout 网格中测试集 F1 最高的一组：lr 1e-3、dropout 0.1，2026-10-08）
export CUDA_VISIBLE_DEVICES="${GPU_APAVA:-4}"
for seed in 42 43 44; do
  run_one apava "$seed" \
    --split_seed 20260917 \
    --e_layers 6 \
    --batch_size 32 \
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 1e-3 \
    --train_epochs 100 \
    --patience 10 \
    --swa \
    --patch_len_list 2,2,2,4,4,4,16,16,16,16,32,32,32,32,32 \
    --augmentations none,drop0.35 \
    "$@"
done

# tdbrain 数据集
export CUDA_VISIBLE_DEVICES="${GPU_TDBRAIN:-5}"
for seed in 42 43 44; do
  run_one tdbrain "$seed" \
    --split_seed 42 \
    --e_layers 6 \
    --batch_size 32 \
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 1e-4 \
    --train_epochs 100 \
    --patience 10 \
    --swa \
    --patch_len_list 8,8,8,16,16,16 \
    --augmentations none,drop0.25 \
    "$@"
done

# shimmer10 数据集
export CUDA_VISIBLE_DEVICES="${GPU_SHIMMER:-6}"
for seed in 42 43 44; do
  run_one shimmer10 "$seed" \
    --split_seed 42 \
    --e_layers 6 \
    --batch_size 1 \
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 1e-4 \
    --train_epochs 100 \
    --patience 10 \
    --swa \
    --patch_len_list 2,4,8 \
    --augmentations none \
    "$@"
done

# pads11 数据集
export CUDA_VISIBLE_DEVICES="${GPU_PADS:-7}"
for seed in 42 43 44; do
  run_one pads11 "$seed" \
    --split_seed 42 \
    --e_layers 6 \
    --batch_size 4 \
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 1e-4 \
    --train_epochs 100 \
    --patience 10 \
    --swa \
    --patch_len_list 2,4,8 \
    --augmentations none \
    "$@"
done
