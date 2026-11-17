#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
export PATCH="${PATCH:-patch64}"
export EXPERIMENT=ablation-wo-visual
bash scripts/classification/NeuroSigVIA_ICIMH.sh \
  --icimh_branches numeric \
  "$@"
