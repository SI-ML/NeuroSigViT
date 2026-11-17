#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
export PATCH="${PATCH:-patch64}"
# TiViT 灰度图参数分析（默认即 TiViT 原样：stride 0.1、窗长 √T、gamma 0.8）：stride<X> 滑窗步长为窗长的 X 倍；
# patch<N> 窗长 N；gamma<X> 灰度幂次；res<N> 先按 N×N 作图再缩放到 224。用 - 连接即可组合。
for variant in ${GRAY_VARIANTS:-gamma0.7 stride0.5 stride1 gamma1 patch4 patch16 res112 res56}; do
  option=()
  IFS=- read -r -a parts <<< "$variant"
  for part in "${parts[@]}"; do
    case "$part" in
      stride*) option+=(--gray_stride "${part#stride}") ;;
      patch*) option+=(--gray_patch_size "${part#patch}") ;;
      gamma*) option+=(--gray_gamma "${part#gamma}") ;;
      res*) option+=(--line_plot_render_size "${part#res}") ;;
    esac
  done
  export EXPERIMENT="ablation-grayscale-$variant"
  bash scripts/classification/NeuroSigVIA_ICIMH.sh \
    --image_representation tivit_grayscale \
    "${option[@]}" \
    "$@"
done
