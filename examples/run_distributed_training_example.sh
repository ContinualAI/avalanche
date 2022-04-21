#!/usr/bin/env bash
echo "This will run single-GPU and multi-GPU training for naive and replay"
echo "Run me from the avalanche repo root as 'bash examples/run_distributed_training_example.sh'"
eval "$(conda shell.bash hook)"
conda activate avalanche-dev-env
set -euo pipefail
ngpus=$(nvidia-smi -L | wc -l)
export PYTHONPATH="${PYTHONPATH-}:${PWD}"
CUDA_VISIBLE_DEVICES=-1 torchrun --standalone --nnodes=1 --nproc_per_node=$ngpus examples/distributed_training.py
CUDA_VISIBLE_DEVICES=-1 python examples/distributed_training.py
CUDA_VISIBLE_DEVICES=-1 torchrun --standalone --nnodes=1 --nproc_per_node=$ngpus examples/distributed_training.py  --use_replay
CUDA_VISIBLE_DEVICES=-1 python examples/distributed_training.py --use_replay
CUDA_VISIBLE_DEVICES=-1 torchrun --standalone --nnodes=1 --nproc_per_node=$ngpus examples/distributed_training.py  --use_replay --use_scheduler
CUDA_VISIBLE_DEVICES=-1 python examples/distributed_training.py --use_replay --use_scheduler

#torchrun --standalone --nnodes=1 --nproc_per_node=$ngpus examples/distributed_training.py --use_cuda
#python examples/distributed_training.py --use_cuda
#torchrun --standalone --nnodes=1 --nproc_per_node=$ngpus examples/distributed_training.py --use_cuda --use_replay
#python examples/distributed_training.py --use_cuda --use_replay
