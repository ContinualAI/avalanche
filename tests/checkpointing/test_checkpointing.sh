#!/usr/bin/env bash
# Script used to automatically test various combinations of plugins when used with
# the checkpointing functionality.
set -euo pipefail
cd examples
rm -rf checkpoints
rm -rf metrics_no_checkpoint
rm -rf metrics_checkpoint

export PYTHONUNBUFFERED=1
export PYTHONPATH=..
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BENCHMARK="SplitMNIST"

run_and_check() {
  # Without checkpoints
  python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at -1 \
    --plugins "$@" --benchmark $BENCHMARK --log_metrics_to './metrics_no_checkpoint'
  rm -r checkpoints

  # Stop after experience 1
  python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at 1 \
    --plugins "$@" --benchmark $BENCHMARK --log_metrics_to './metrics_checkpoint'
  echo "Running from checkpoint 1..."
  python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at 1  \
    --plugins "$@" --benchmark $BENCHMARK --log_metrics_to './metrics_checkpoint'
  rm -r checkpoints

  python -u ../tests/checkpointing/check_metrics_aligned.py \
    "./metrics_no_checkpoint" "./metrics_checkpoint"

  rm -r metrics_no_checkpoint
  rm -r metrics_checkpoint
}

run_and_check "replay"