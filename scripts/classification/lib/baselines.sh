#!/usr/bin/env bash
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_ROOT="${RUN_ROOT:-main_runs}"
PYTHON="${PYTHON:-python}"
export NEUROSIGVIA_EEG_ROOT="${NEUROSIGVIA_EEG_ROOT:-./data/eeg/processed}"
export NEUROSIGVIA_WEARABLE_ROOT="${NEUROSIGVIA_WEARABLE_ROOT:-./data/wearable}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

# 数据集和种子筛选只控制任务列表；调度器统一安排设备。
run_one() {
  local dataset="$1" seed="$2"
  shift 2
  if [[ " ${RUN_SEEDS:-42 43 44} " != *" $seed "* ]]; then return; fi
  if [[ " ${RUN_DATASETS:-apava tdbrain shimmer10 pads11} " != *" $dataset "* ]]; then return; fi
  local result="$RUN_ROOT/results/${MODEL,,}4${dataset}_comparison_${RUN_DATE}/seed$seed"
  local runtime=(--random_seed "$seed" --result_dir "$result")
  if [[ "$MODEL" == TeCh ]]; then runtime=(--seed "$seed" --output "$result"); fi
  local command=(env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$PYTHON" -u run.py
    --model "$MODEL"
    --dataset "$dataset"
    "${runtime[@]}"
    "$@")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}
