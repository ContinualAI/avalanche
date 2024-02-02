#!/usr/bin/env bash
# Script used to automatically test various combinations of plugins when used with
# the checkpointing functionality.
set -euo pipefail
rm -rf logs
rm -rf checkpoints
rm -rf metrics_no_checkpoint
rm -rf metrics_checkpoint
rm -rf checkpoint.pkl
rm -rf .coverage
rm -rf .coverage_no_checkpoint
rm -rf .coverage_checkpoint1
rm -rf .coverage_checkpoint2
rm -rf coverage_checkpointing_*

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
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
RUN_COVERAGE=$(str_bool "${RUN_COVERAGE:-False}")

GPU_PARAM="--cuda -1"

if [ "$RUN_GPU_TESTS" = "true" ]
then
  GPU_PARAM="--cuda 0"
fi

BASE_RUNNER=(
  "python"
  "-u"
)

if [ "$RUN_COVERAGE" = "true" ]
then
  echo "Running with coverage..."
  BASE_RUNNER=(
    "coverage"
    "run"
  )
fi

BASE_RUNNER=(
  "${BASE_RUNNER[@]}"
  "tests/checkpointing/task_incremental_with_checkpointing.py"
  $GPU_PARAM
  "--benchmark"
  $BENCHMARK
)

run_and_check() {
  # Without checkpoints
  "${BASE_RUNNER[@]}" --checkpoint_at -1 \
    --plugins "$@" --log_metrics_to './metrics_no_checkpoint'
  rm -r ./checkpoint.pkl
  if [ "$RUN_COVERAGE" = "true" ]
  then
    mv .coverage .coverage_no_checkpoint
  fi

  # Stop after experience 1
  "${BASE_RUNNER[@]}" --checkpoint_at 1 \
    --plugins "$@" --log_metrics_to './metrics_checkpoint'
  if [ "$RUN_COVERAGE" = "true" ]
  then
    mv .coverage .coverage_checkpoint1
  fi
  echo "Running from checkpoint 1..."
  "${BASE_RUNNER[@]}" --checkpoint_at 1  \
    --plugins "$@" --log_metrics_to './metrics_checkpoint'
  if [ "$RUN_COVERAGE" = "true" ]
  then
    mv .coverage .coverage_checkpoint2
    coverage combine .coverage_no_checkpoint .coverage_checkpoint1 .coverage_checkpoint2
    mv .coverage "coverage_checkpointing_$@"
  fi
  rm -r ./checkpoint.pkl

  python -u tests/checkpointing/check_metrics_aligned.py \
    "./metrics_no_checkpoint" "./metrics_checkpoint"

  rm -r metrics_no_checkpoint
  rm -r metrics_checkpoint
  rm -r logs

  rm -rf .coverage_no_checkpoint
  rm -rf .coverage_checkpoint1
  rm -rf .coverage_checkpoint2
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
