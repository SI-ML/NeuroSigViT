#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
export PATCH="${PATCH:-patch64}"
# 骨干消融只替换视觉编码器；层数 L 表示保留前 L 个 Transformer 块（SigLIP2 共 27 块，DINOv3 共 24 块）。
# 不设 LAYERS 时每个骨干只跑默认层；设置后逐层扫描，实验名带 -L<层>。
for backbone in ${BACKBONES:-siglip2 dinov3}; do
  case "$backbone" in
    siglip2) weights=vit_so400m_patch14_siglip_224.v2_webli; default_layer=12 ;;
    dinov3) weights=vit_large_patch16_dinov3.lvd1689m; default_layer=17 ;;
    # ViTST 的视觉骨干 Swin-B（ImageNet-22k）；层号为阶段号，4 = 完整模型（同 ViTST）
    swin) weights=swin_base_patch4_window7_224.ms_in22k; default_layer=4 ;;
    *) echo "BACKBONES must contain siglip2, dinov3 or swin" >&2; exit 2 ;;
  esac
  for layer in ${LAYERS:-$default_layer}; do
    if [[ -n "${LAYERS:-}" ]]; then export EXPERIMENT="ablation-backbone-$backbone-L$layer"
    else export EXPERIMENT="ablation-backbone-$backbone"; fi
    bash scripts/classification/NeuroSigVIA_ICIMH.sh \
      --vision_backbone "$backbone" \
      --vit_1_name "./pretrained/$weights" \
      --vit_1_layer "$layer" \
      "$@"
  done
done
