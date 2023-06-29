"""
This example shows how to use FFCV data loading system.
"""

import argparse
from datetime import datetime
import time

import torch
import torch.optim.lr_scheduler
from torch.optim import Adam
from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks.classic.ccifar100 import SplitCIFAR100
from avalanche.benchmarks.classic.core50 import CORe50
from avalanche.benchmarks.classic.ctiny_imagenet import SplitTinyImageNet
from avalanche.benchmarks.utils.ffcv_support import prepare_ffcv_datasets
from avalanche.models import SimpleMLP
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.supervised import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import TensorboardLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(cuda: int):
    # --- CONFIG
    device = torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
    )
    RNGManager.set_random_seeds(1234)

    benchmark_type = 'tinyimagenet'

    # --- BENCHMARK CREATION
    num_workers = 8
    if benchmark_type == 'mnist':
        input_size = 28* 28
        num_workers = 4
        benchmark = SplitMNIST(n_experiences=5, seed=42, class_ids_from_zero_from_first_exp=True)
    elif benchmark_type == 'core50':
        benchmark = CORe50()
        benchmark.n_classes = 50
    elif benchmark_type == 'cifar100':
        benchmark = SplitCIFAR100(5, seed=1234, shuffle=True)
        input_size = 32 * 32 * 3
    elif benchmark_type == 'tinyimagenet':
        benchmark = SplitTinyImageNet()
        input_size = 64 * 64 * 3
    else:
        raise RuntimeError('Unknown benchmark')

    print('Preparing FFCV datasets...')
    prepare_ffcv_datasets(
        benchmark=benchmark,
        write_dir=f'./ffcv_test_{benchmark_type}',
        device=device,
        ffcv_parameters=dict(num_workers=8),
    )
    print('FFCV datasets ready')

    # MODEL CREATION
    model = SimpleMLP(
        input_size=input_size,
        num_classes=benchmark.n_classes
    )

    # choose some metrics and evaluation method
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True, experience=True),
        loggers=[
            TensorboardLogger(f"tb_data/{datetime.now()}"),
            InteractiveLogger()
        ],
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    replay_plugin = ReplayPlugin(mem_size=100, batch_size=125, batch_size_mem=25)
    cl_strategy = Naive(
        model,
        Adam(model.parameters()),
        train_mb_size=128,
        train_epochs=4,
        eval_mb_size=128,
        device=device,
        plugins=[replay_plugin],
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    start_time = time.time()
    for i, experience in enumerate(benchmark.train_stream):
        cl_strategy.train(
            experience,
            shuffle=False,
            persistent_workers=True,
            num_workers=num_workers,
            ffcv_args={
                'print_ffcv_summary': True
            }
        )

        cl_strategy.eval(
            benchmark.test_stream[:i+1],
            shuffle=False,
            num_workers=num_workers,
            ffcv_args={
                'print_ffcv_summary': True
            }
        )
    end_time = time.time()
    print('Overall time:', end_time - start_time, 'seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args.cuda)
