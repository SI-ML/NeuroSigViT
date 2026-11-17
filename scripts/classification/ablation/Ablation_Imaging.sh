#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
export PATCH="${PATCH:-patch64}"
for representation in ordinary_line gaf adaptive_heatmap tivit_grayscale spectrogram; do
  export EXPERIMENT="ablation-imaging-$representation"
  bash scripts/classification/NeuroSigVIA_ICIMH.sh \
    --image_representation "$representation" \
    "$@"
done
