#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate avalanche-dev-env
set -euo pipefail

CPU_PARALLELISM=4
GPU_PARALLELISM=0

usage() {
  echo "This will run single-process and multi-process training for naive, replay, and replay+scheduler setups."
  echo "Used to check for differences between local and distributed training."
  echo ""
  echo "Run me from the avalanche repo root as 'bash examples/run_distributed_training_example.sh'"
  echo
  echo "Syntax: examples/run_distributed_training_example [-h] [-c CPU_PARALLELISM] [-g GPU_PARALLELISM]"
  echo ""
  echo "Options:"
  echo "-h     Print this Help."
  echo "-c     Set the CPU parallelism for distributed experiments. Defaults to 4."
  echo "       Set this value to 0 to skip CPU experiments."
  echo "-g     Set the GPU parallelism for distributed experiments. Defaults to 0 (skip GPU experiments)."
  echo "       Set this value to -1 to auto-detect how many GPUs are in the system."
}

exit_abnormal() {
  usage
  exit 1
}

while getopts ":c:g:" options; do
  case "${options}" in
    c)
      CPU_PARALLELISM=${OPTARG}
      ;;
    g)
      GPU_PARALLELISM=${OPTARG}
      ;;
    h)
      usage
      exit 0
      ;;
    :)
      echo "Error: -${OPTARG} requires an argument!"
      echo ""
      exit_abnormal
      ;;
    *)
      exit_abnormal
      ;;
  esac
done

if [[ "$GPU_PARALLELISM" == "-1" ]]; then
  GPU_PARALLELISM=$(nvidia-smi -L | wc -l)
  echo "Auto-detected $GPU_PARALLELISM GPUs."
fi

export PYTHONPATH="${PYTHONPATH-}:${PWD}"

if [[ "$CPU_PARALLELISM" == "0" ]]; then
  echo "Skipping CPU experiments."
else
  # Naive experiments
  torchrun --standalone --nnodes=1 --nproc_per_node=$CPU_PARALLELISM examples/distributed_training.py \
    --exp_name "distributed_naive_unsched_cpu"
  python examples/distributed_training.py \
    --exp_name "single_process_naive_unsched_cpu"

  # Replay experiments
  torchrun --standalone --nnodes=1 --nproc_per_node=$CPU_PARALLELISM examples/distributed_training.py \
    --use_replay --exp_name "distributed_replay_unsched_cpu"
  python examples/distributed_training.py \
    --use_replay --exp_name "single_process_replay_unsched_cpu"

  # Replay + LR scheduler experiments
  torchrun --standalone --nnodes=1 --nproc_per_node=$CPU_PARALLELISM examples/distributed_training.py \
    --use_replay --use_scheduler --exp_name "distributed_replay_scheduler_cpu"
  python examples/distributed_training.py \
    --use_replay --use_scheduler --exp_name "single_process_replay_scheduler_cpu"
fi

if [[ "$GPU_PARALLELISM" == "0" ]]; then
  echo "Skipping GPU experiments."
  exit 0
fi

# Naive experiments (GPU)
torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_PARALLELISM examples/distributed_training.py \
  --exp_name "distributed_naive_unsched_gpu" --use_cuda
python examples/distributed_training.py \
  --exp_name "single_process_naive_unsched_gpu" --use_cuda

# Replay experiments (GPU)
torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_PARALLELISM examples/distributed_training.py \
  --exp_name "distributed_replay_unsched_gpu" --use_cuda --use_replay
python examples/distributed_training.py \
  --exp_name "single_process_replay_unsched_gpu" --use_cuda --use_replay

# Replay + LR scheduler experiments (GPU)
torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_PARALLELISM examples/distributed_training.py \
  --exp_name "distributed_replay_scheduler_gpu" --use_cuda --use_replay --use_scheduler
python examples/distributed_training.py \
  --exp_name "single_process_replay_scheduler_gpu" --use_cuda --use_replay --use_scheduler