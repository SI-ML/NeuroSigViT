#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
# 去掉数值分支后不切块：每个样本整段成一幅图（PADS11 也是整段 976 点），不受外部 PATCH 影响。
export PATCH=nopatch
export EXPERIMENT=ablation-wo-numeric
bash scripts/classification/NeuroSigVIA_ICIMH.sh \
  --icimh_branches visual \
  "$@"
