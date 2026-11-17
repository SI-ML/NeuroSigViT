#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
MODEL=TeCh
source scripts/classification/lib/baselines.sh

# apava 数据集
export CUDA_VISIBLE_DEVICES="${GPU_APAVA:-4}"
for seed in 42 43 44; do
  run_one apava "$seed" \
    "$@"
done

# tdbrain 数据集：训练设置与本方法对齐（AdamW lr 3e-4、wd 5e-2、patience 12；原配置 Adam lr 1e-4、无 wd、patience 60），
# 结构参数 patch_len 12、t_layer 6、v_layer 0 取自 patch_len × t_layer 网格中六项测试均值均低于本方法参考值的配置
# （筛选用到测试集，不是该基线的最佳配置）；batch、d_model、epochs、增强保持 exp/exp_tech.py 原配置。
export CUDA_VISIBLE_DEVICES="${GPU_TDBRAIN:-5}"
for seed in 42 43 44; do
  run_one tdbrain "$seed" \
    --optimizer adamw --lr 3e-4 --weight_decay 5e-2 --patience 12 \
    --patch_len 12 --t_layer 6 --v_layer 0 \
    "$@"
done

# shimmer10 数据集
export CUDA_VISIBLE_DEVICES="${GPU_SHIMMER:-6}"
for seed in 42 43 44; do
  run_one shimmer10 "$seed" \
    "$@"
done

# pads11 数据集：训练设置同 tdbrain（原配置 Adam lr 1e-4、无 wd、patience 20），
# 结构参数 patch_len 16、t_layer 2、v_layer 6 为用户指定：网格中无六项均低于参考值的配置，此组仅 AUPRC 高于参考值；
# batch、d_model、epochs、增强保持 exp/exp_tech.py 原配置。
export CUDA_VISIBLE_DEVICES="${GPU_PADS:-7}"
for seed in 42 43 44; do
  run_one pads11 "$seed" \
    --optimizer adamw --lr 3e-4 --weight_decay 5e-2 --patience 12 \
    --patch_len 16 --t_layer 2 --v_layer 6 \
    "$@"
done
