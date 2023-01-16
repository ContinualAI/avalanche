#!/usr/bin/env bash
# Script used to automatically test various combinations of plugins when used with
# the distributed training functionality.
set -euo pipefail
cd tests/distributed
rm -rf logs
rm -rf metrics_no_distributed
rm -rf metrics_distributed

export PYTHONUNBUFFERED=1
export PYTHONPATH=../..
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

TESTS_PARALLELISM=4

GPU_PARAM=""

if [ "$RUN_GPU_TESTS" = "true" ]
then
  GPU_PARAM="--cuda"
  TESTS_PARALLELISM=$(nvidia-smi -L | wc -l)
  echo "Auto-detected $TESTS_PARALLELISM GPUs."
fi

EXP_RUN_LINE="torchrun --standalone --nnodes=1 --nproc_per_node=$TESTS_PARALLELISM"

run_and_check() {
  set -x
  # Run distributed training
  $EXP_RUN_LINE distributed_training_main.py $GPU_PARAM \
    --plugins "$@" --benchmark $BENCHMARK --log_metrics_to './metrics_distributed'

  # Without distributed training
  python distributed_training_main.py $GPU_PARAM \
    --plugins "$@" --benchmark $BENCHMARK --log_metrics_to './metrics_no_distributed'

  #python -u check_metrics_aligned.py \
  #  "./metrics_no_distributed" "./metrics_distributed"

  rm -r metrics_no_distributed
  rm -r metrics_distributed
  rm -r logs
  set +x
}

run_and_check "replay"

if [ "$RUN_FAST_TESTS" = "false" ]
then
  echo "Running slow tests..."
  run_and_check "lwf"
  run_and_check "gdumb"
  run_and_check "cwr" "replay"
fi
