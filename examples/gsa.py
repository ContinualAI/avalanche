################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-10-2020                                                             #
# Author(s): Vincenzo Lomonaco, Hamed Hemati                                   #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Replay strategy in an online
benchmark created using OnlineCLScenario.
"""

import argparse
import torch
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks import SplitCIFAR100
from avalanche.models import SlimResNet18
from avalanche.training.supervised.gsa import GSA
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # --- CONFIG
    # device = torch.device(
    #     f"cuda:{args.cuda}"
    #     if torch.cuda.is_available() and args.cuda >= 0
    #     else "cpu"
    # )
    device = torch.device("mps")  # TODO: change default device to cpu/cuda
    n_batches = 5
    # ---------

    # --- Benchmark
    benchmark = SplitCIFAR100(n_experiences=10, return_task_id=True)

    # ---------

    # MODEL CREATION
    model = SlimResNet18(nclasses=benchmark.n_classes)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    cl_strategy = GSA(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001),
        CrossEntropyLoss(),
        train_passes=1,
        train_mb_size=10,
        eval_mb_size=32,
        mem_size=1000,
        batch_size_mem=64,
        device=device,
        evaluator=eval_plugin
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []

    # Create online benchmark
    batch_streams = benchmark.streams.values()
    # ocl_benchmark = OnlineCLScenario(batch_streams)
    for i, exp in enumerate(benchmark.train_stream):
        # Create online scenario from experience exp
        # !!! For GSA, we need to have access to task boundaries
        ocl_benchmark = OnlineCLScenario(
            original_streams=batch_streams, experiences=exp, experience_size=10,
            access_task_boundaries=True
        )
        # Train on the online train stream of the scenario
        cl_strategy.train(ocl_benchmark.train_stream)
        results.append(cl_strategy.eval(benchmark.test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
