#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# 表 1 的 TSPulse：只用数值分支（--icimh_branches numeric），冻结 TSPulse 编码器
# （tspulse-block-dualhead-512-p16-r1，逐通道插值到 512 点），只训练通道池化与分类头；
# 分类头 lr 3e-4、dropout 0.2、weight decay 5e-2，与主方法相同。
# APAVA、TDBRAIN、PADS 的窗口与批量同主方法（PATCH=final）；Shimmer 用 2048 点窗口、通道取均值
# （--numeric_only_reduce mean，批量 8），该配置按验证集 Macro-F1 选出。
# TSPulse 代码来自 granite-tsfm 仓库，由 TSPULSE_REPO 指定（默认 ~/granite-tsfm）；权重目录由 TSPULSE_WEIGHTS 指定。
export PATCH="${PATCH:-final}"
export EXPERIMENT="${EXPERIMENT:-numeric-tspulse}"
DATASETS="${RUN_DATASETS:-apava tdbrain shimmer10 pads11}"
COMMON=(--icimh_branches numeric
  --numeric_backbone tspulse
  --mantis_name "${TSPULSE_WEIGHTS:-./pretrained/tspulse-block-dualhead-512-p16-r1}")

others=""
for dataset in $DATASETS; do
  [[ "$dataset" == shimmer10 ]] || others+="$dataset "
done
if [[ -n "$others" ]]; then
  RUN_DATASETS="$others" bash scripts/classification/NeuroSigVIA_ICIMH.sh "${COMMON[@]}" "$@"
fi
if [[ " $DATASETS " == *" shimmer10 "* ]]; then
  RUN_DATASETS=shimmer10 bash scripts/classification/NeuroSigVIA_ICIMH.sh "${COMMON[@]}" \
    --outer_patch_size 2048 \
    --outer_patch_stride 2048 \
    --numeric_only_reduce mean \
    --patch_checkpoint_metric window_macro_f1 \
    "$@"
fi
