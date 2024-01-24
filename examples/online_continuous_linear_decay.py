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
from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.benchmarks.scenarios import split_continuous_linear_decay_stream
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
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # CL Strategy
    cl_strategy = Naive(
        model,
        torch.optim.Adam(model.parameters(), lr=0.1),
        CrossEntropyLoss(),
        train_epochs=1,
        train_mb_size=10,
        eval_mb_size=32,
        device=device,
        evaluator=eval_plugin,
    )

    # Create a "continuous" stream with linear decay from the original
    # train stream of the benchmark
    ocl_stream = split_continuous_linear_decay_stream(
        benchmark.train_stream,
        experience_size=10,
        iters_per_virtual_epoch=100,
        beta=0.5,
        shuffle=True,
    )

    # Train the strtegy on the continuous stream
    cl_strategy.train(ocl_stream)

    # Test on the benchmark test stream
    results = cl_strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
