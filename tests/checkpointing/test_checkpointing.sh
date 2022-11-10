#!/usr/bin/env bash
# Script used to automatically test various combinations of plugins when used with
# the checkpointing functionality.
set -euo pipefail
cd tests/checkpointing
rm -rf logs
rm -rf checkpoints
rm -rf metrics_no_checkpoint
rm -rf metrics_checkpoint

export PYTHONUNBUFFERED=1
export PYTHONPATH=../..
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BENCHMARK="TestBenchmark"

# Config from env
# https://blog.stigok.com/2022/02/08/parsing-boolean-string-statements-in-bash.html
function str_bool {
  local str="${1:-false}"
  local pat='^(true|1|yes)$'
  if [[ ${str,,} =~ $pat ]]
  then
    echo 'true'
  else
    echo 'false'
  fi
}

RUN_FAST_TESTS=$(str_bool "${FAST_TEST:-False}")
RUN_GPU_TESTS=$(str_bool "${USE_GPU:-False}")

GPU_PARAM="--cuda -1"

if [ "$RUN_GPU_TESTS" = "true" ]
then
  GPU_PARAM="--cuda 0"
fi

run_and_check() {
  # Without checkpoints
  python -u task_incremental_with_checkpointing.py $GPU_PARAM --checkpoint_at -1 \
    --plugins "$@" --benchmark $BENCHMARK --log_metrics_to './metrics_no_checkpoint'
  rm -r checkpoints

  # Stop after experience 1
  python -u task_incremental_with_checkpointing.py $GPU_PARAM --checkpoint_at 1 \
    --plugins "$@" --benchmark $BENCHMARK --log_metrics_to './metrics_checkpoint'
  echo "Running from checkpoint 1..."
  python -u task_incremental_with_checkpointing.py $GPU_PARAM --checkpoint_at 1  \
    --plugins "$@" --benchmark $BENCHMARK --log_metrics_to './metrics_checkpoint'
  rm -r checkpoints

  python -u check_metrics_aligned.py \
    "./metrics_no_checkpoint" "./metrics_checkpoint"

  rm -r metrics_no_checkpoint
  rm -r metrics_checkpoint
  rm -r logs
}

run_and_check "replay"

if [ "$RUN_FAST_TESTS" = "false" ]
then
  echo "Running slow tests..."
  run_and_check "lwf"
  run_and_check "ewc"
  run_and_check "gdumb"
  run_and_check "cwr" "replay"
fi
