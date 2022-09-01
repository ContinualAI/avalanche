#!/usr/bin/env bash
set -euo pipefail
rm -rf checkpoints

export PYTHONPATH=..
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PLUGINS_LIST="$@"

# Do not use checkpoints
python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at -1 \
  --plugins $PLUGINS_LIST
rm -r checkpoints

# Stop after experience 0
python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at 0 \
  --plugins $PLUGINS_LIST
echo "Running from checkpoint 0..."
python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at 0  \
  --plugins $PLUGINS_LIST
rm -r checkpoints

# Stop after experience 1
python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at 1 \
  --plugins $PLUGINS_LIST
echo "Running from checkpoint 1..."
python task_incremental_with_checkpointing.py --cuda 0 --checkpoint_at 1  \
  --plugins $PLUGINS_LIST
rm -r checkpoints