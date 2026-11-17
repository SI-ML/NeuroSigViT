#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
export PATCH="${PATCH:-patch64}"
# 折线图成像参数分析：res<N> 先按 N×N 作图再缩放到 224；lw<N> 线宽 N 像素；gap<X> 每格四周留白比例 X；global 块内所有通道共用一个缩放范围；aglane 波形纵向只占与活动图同数据集每行相同的高度。
# 用 - 连接多个参数即可组合；各部分按名称中的顺序追加到同一条命令。
for variant in ${LINEPLOT_VARIANTS:-res112 res56 lw2 lw4 gap0.2 gap0.4}; do
  option=()
  IFS=- read -r -a parts <<< "$variant"
  for part in "${parts[@]}"; do
    case "$part" in
      res*) option+=(--line_plot_render_size "${part#res}") ;;
      lw*) option+=(--line_plot_width "${part#lw}") ;;
      gap*) option+=(--line_plot_gap "${part#gap}") ;;
      overlay) option+=(--line_plot_overlay) ;;
      global) option+=(--line_plot_global_scale) ;;
      aglane) option+=(--line_plot_activity_lane) ;;
    esac
  done
  export EXPERIMENT="ablation-lineplot-$variant"
  bash scripts/classification/NeuroSigVIA_ICIMH.sh \
    --image_representation ordinary_line \
    "${option[@]}" \
    "$@"
done
