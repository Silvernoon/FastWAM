#!/bin/bash

# This script runs LIBERO evaluation tasks sequentially on a single GPU.

set -euo pipefail

run_libero_eval() {
  local task_list_file=$1
  echo "task_file: $task_list_file"

  require_non_empty() {
    local var_name="$1"
    local var_val="${!var_name}"
    if [ -z "$var_val" ]; then
      echo "Error: required variable $var_name is not set"
      exit 1
    fi
  }

  ROOT_DIR=${ROOT_DIR:-"$(pwd)"}
  export ROOT_DIR
  RUN_ID=${RUN_ID:-"eval_$(date +%Y%m%d_%H%M%S)"}
  export RUN_ID
  OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/evaluate_results/$RUN_ID"}
  export OUTPUT_DIR
  EXP_NAME=${EXP_NAME:-""}
  export EXP_NAME

  GPU_ID=${GPU_ID:-0}
  require_non_empty "NUM_TRIALS"

  CKPT=${CKPT:-""}
  CONFIG=${CONFIG:-""}
  require_non_empty "CKPT"
  require_non_empty "CONFIG"

  # Normalize CONFIG
  CONFIG="${CONFIG#configs/}"
  CONFIG="${CONFIG#task/}"
  CONFIG="${CONFIG%.yaml}"
  export CONFIG

  mkdir -p "$OUTPUT_DIR"
  cp "$task_list_file" "$OUTPUT_DIR/"

  local TASK_LOG_DIR="$OUTPUT_DIR/task_logs"
  local FAILED_TASKS_FILE="$OUTPUT_DIR/failed_tasks.txt"
  mkdir -p "$TASK_LOG_DIR"
  : >"$FAILED_TASKS_FILE"

  echo "CKPT: $CKPT"
  echo "CONFIG: $CONFIG"
  echo "ROOT_DIR: $ROOT_DIR"
  echo "GPU_ID: $GPU_ID"
  echo "NUM_TRIALS: $NUM_TRIALS"
  echo "OUTPUT_DIR: $OUTPUT_DIR"
  echo "EXP_NAME: $EXP_NAME"

  local total_tasks=$(wc -l <"$task_list_file")
  local completed=0
  local failed=0

  echo "Total tasks: $total_tasks"
  echo "Starting sequential evaluation..."

  while IFS=, read -r suite task_id; do
    [ -z "$suite" ] && continue

    local result_file="$OUTPUT_DIR/$suite/gpu${GPU_ID}_task${task_id}_results.json"
    local log_file="$TASK_LOG_DIR/${suite}_task${task_id}_gpu${GPU_ID}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: $suite task_id=$task_id on GPU$GPU_ID"

    set +e
    CUDA_VISIBLE_DEVICES=$GPU_ID python experiments/libero/eval_libero_single.py \
      task=$CONFIG ckpt=$CKPT \
      EVALUATION.task_suite_name=$suite EVALUATION.task_id=$task_id gpu_id=$GPU_ID \
      EVALUATION.num_trials=$NUM_TRIALS EVALUATION.output_dir=$OUTPUT_DIR \
      $EXTRA_ARGS >"$log_file" 2>&1
    rc=$?
    set -e

    if [ $rc -eq 0 ] && [ -f "$result_file" ]; then
      ((completed++))
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ($completed/$total_tasks): $suite task_id=$task_id"
    else
      ((failed++))
      local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
      echo "$timestamp,$suite,$task_id,gpu=$GPU_ID,rc=$rc,log=$log_file" >>"$FAILED_TASKS_FILE"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $suite task_id=$task_id (rc=$rc, log=$log_file)"
      echo "Stopping due to task failure."
      return 2
    fi
  done <"$task_list_file"

  echo "All $total_tasks tasks completed successfully!"
  echo "Generating evaluation report..."
  python experiments/libero/summarize_results.py --output_dir="$OUTPUT_DIR"
}

# Entrypoint
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  if [ $# -lt 1 ]; then
    echo "Error: task file path is required"
    echo "Usage: $0 <task_file>"
    exit 1
  fi
  run_libero_eval "$1"
  exit $?
fi
