#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PATCH="${PATCH:-final}"
EXPERIMENT="${EXPERIMENT:-main-${PATCH}}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
CACHE_DATE="${CACHE_DATE:-$RUN_DATE}"
RUN_ROOT="${RUN_ROOT:-main_runs}"
PYTHON="${PYTHON:-python}"

# 主表设置：lr、wd、dropout、成像四个数据集相同（早停 patience 12、无热身为 run.py 默认值）；
# 各数据集只有 batch 与切块不同（PATCH=final：PADS 切 64、步长 64，其余不切块）。
SHARED=(--mlp_lr 3e-4 --mlp_weight_decay 5e-2 --mlp_dropout 0.2 --activity_graph_line_width 4.0)

# 数据集与种子筛选仅控制任务列表；所有数据集复用同一条命令构造。
run_one() {
  local dataset="$1" seed="$2" full_length="$3" final_patch="$4"
  shift 4
  if [[ " ${RUN_SEEDS:-42 43 44} " != *" $seed "* ]]; then return; fi
  if [[ " ${RUN_DATASETS:-apava tdbrain shimmer10 pads11} " != *" $dataset "* ]]; then return; fi
  local patch_size=64
  case "$PATCH" in
    final) patch_size="$final_patch" ;;
    patch64) patch_size=64 ;;
    nopatch) patch_size="$full_length" ;;
    *) echo "PATCH must be final, patch64 or nopatch" >&2; return 2 ;;
  esac
  local run="icimh4${dataset}_${EXPERIMENT}_${RUN_DATE}"
  local command=(env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$PYTHON" -u run.py
    --outer_patch_size "$patch_size"
    --outer_patch_stride "$patch_size"
    --random_seed "$seed"
    --feature_cache_dir "$RUN_ROOT/feature_cache/${EXPERIMENT}_icimh_${CACHE_DATE}"
    --result_dir "$RUN_ROOT/results/$run/seed$seed"
    "${SHARED[@]}"
    "$@")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

# APAVA 数据集
export CUDA_VISIBLE_DEVICES="${GPU_APAVA:-4}"
for seed in 42 43 44; do
  run_one apava "$seed" 256 256     --datasets eeg     --dataset_names APAVA     --data_dir "${NEUROSIGVIA_EEG_ROOT:-./data/eeg/processed}"     --batch_size 16     --patch_checkpoint_metric window_macro_f1     "$@"
done

# TDBRAIN 数据集
export CUDA_VISIBLE_DEVICES="${GPU_TDBRAIN:-5}"
for seed in 42 43 44; do
  run_one tdbrain "$seed" 256 256     --datasets eeg     --dataset_names TDBRAIN     --data_dir "${NEUROSIGVIA_EEG_ROOT:-./data/eeg/processed}"     --batch_size 64     --patch_checkpoint_metric window_macro_f1     "$@"
done

# Shimmer10 数据集
export CUDA_VISIBLE_DEVICES="${GPU_SHIMMER:-6}"
for seed in 42 43 44; do
  run_one shimmer10 "$seed" 4096 4096     --datasets wearable     --dataset_names Shimmer_10_session10_AFC     --data_dir "${NEUROSIGVIA_WEARABLE_ROOT:-./data/wearable}"     --wearable_label_mode shimmer_hc_vs_pd     --batch_size 8     --patch_checkpoint_metric subject_macro_f1     "$@"
done

# PADS11 数据集：切块 64、步长 64（主表所用；该组验证 F1 低于不切块，并非按验证集选出）
export CUDA_VISIBLE_DEVICES="${GPU_PADS:-7}"
for seed in 42 43 44; do
  run_one pads11 "$seed" 976 64     --datasets wearable     --dataset_names PADS_11_task08_TouchIndex     --data_dir "${NEUROSIGVIA_WEARABLE_ROOT:-./data/wearable}"     --wearable_label_mode pads_pd_vs_hc     --batch_size 8     --patch_checkpoint_metric window_macro_f1     "$@"
done
