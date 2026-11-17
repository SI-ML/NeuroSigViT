#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# 表 1 的 Zeus*：用冻结的 Zeus 编码器替换 NeuroSigViT 数值分支中的 Mantis，取第 3 阶段隐状态；
# 视觉分支（CLIP ViT-H/14 第 14 层）、融合、分类头与各数据集设置均与主方法相同（PATCH=final）。
# Zeus 需要官方仓库与 BasicTS 1.1.0（见 layers/NumericEncoders.py）：
#   ZEUS_REPO（默认 ~/Zeus）、BASICTS_DIR（默认 ~/pylibs/basicts110）；权重目录由 ZEUS_WEIGHTS 指定。
export PATCH="${PATCH:-final}"
export EXPERIMENT="${EXPERIMENT:-numbb-zeus-s3}"
bash scripts/classification/NeuroSigVIA_ICIMH.sh \
  --numeric_backbone zeus \
  --zeus_stage 3 \
  --mantis_name "${ZEUS_WEIGHTS:-./pretrained/zeus}" \
  "$@"
