################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 04-08-2023                                                             #
# Author(s): Hamed Hemati                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Naive strategy in the
continuous task-agnostic scenario as described in the paper
"Task-Agnostic Continual Learning Using Online Variational Bayes 
With Fixed-Point Updates"
https://arxiv.org/pdf/2010.00373.pdf

"""

import argparse
import torch
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks import PermutedMNIST, SplitMNIST
from avalanche.benchmarks.datasets.dataset_utils import \
    default_dataset_location
from avalanche.models import SimpleMLP
from avalanche.training.supervised.strategy_wrappers_online import \
    OnlineNaive
from avalanche.benchmarks.scenarios import OnlineCLScenario 
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # Compute device
    device = torch.device(args.device)

    # Benchmark
    benchmark = SplitMNIST(n_experiences=10)

    # Model
    model = SimpleMLP(num_classes=benchmark.n_classes)

    # Loggers and metrics
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # CL Strategy
    cl_strategy = OnlineNaive(
        model,
        torch.optim.Adam(model.parameters(), lr=0.1),
        CrossEntropyLoss(),
        train_passes=1,
        train_mb_size=10,
        eval_mb_size=32,
        device=device,
        evaluator=eval_plugin,
    )

    # Create streams using the continuous task-agnostic scenario
    batch_streams = benchmark.streams.values()
    cta_benchmark = OnlineCLScenario(
        original_streams=batch_streams,
        experience_size=10,
        stream_split_strategy="continuous_linear_decay",
        access_task_boundaries=False,
        overlap_factor=4,
        iters_per_virtual_epoch=50
    )

    # Start training
    for itr, exp in enumerate(cta_benchmark.train_stream):
        cl_strategy.train(exp)
        print(exp.n_samples_from_each_exp)

    results = cl_strategy.eval(cta_benchmark.original_test_stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
