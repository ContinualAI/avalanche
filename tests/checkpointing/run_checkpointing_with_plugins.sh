#!/usr/bin/env bash
# Used to run the "task_incremental_with_checkpointing.py" script by
# taking the list of plugins as parameters. It will run the script
# by not checkpointing, by checkpointing after the first experience,
# and by checkpointing after the second experience.
set -euo pipefail
rm -rf checkpoints

export PYTHONUNBUFFERED=1
export PYTHONPATH=../..
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PLUGINS_LIST="$@"
BENCHMARK="SplitMNIST"

# Do not use checkpoints
python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at -1 \
  --plugins $PLUGINS_LIST --benchmark $BENCHMARK
rm -r checkpoints

# Stop after experience 0
python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at 0 \
  --plugins $PLUGINS_LIST --benchmark $BENCHMARK
echo "Running from checkpoint 0..."
python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at 0  \
  --plugins $PLUGINS_LIST --benchmark $BENCHMARK
rm -r checkpoints

# Stop after experience 1
python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at 1 \
  --plugins $PLUGINS_LIST --benchmark $BENCHMARK
echo "Running from checkpoint 1..."
python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at 1  \
  --plugins $PLUGINS_LIST --benchmark $BENCHMARK
rm -r checkpoints