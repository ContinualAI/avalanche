################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-04-2020                                                             #
# Author(s): Matthias De Lange                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the CoPE plugin.
It's an example in the online data incremental setting, where both learning and
evaluation is completely task-agnostic.

Original implementation:
https://github.com/Mattdl/ContinualPrototypeEvolution

Reference:
De Lange, Matthias, et al. "Continual prototype evolution: Learning online from
non-stationary data streams." ICCV. 2021.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import torch.optim.lr_scheduler

from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.training.plugins import CoPEPlugin
from avalanche.evaluation.metrics import (
    StreamForgetting,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.generators.benchmark_generators import (
    data_incremental_benchmark,
)


def main(args):
    """
    Last Avalanche version reference performance (online = 1 epoch):

    Class-incremental (online):
        Top1_Acc_Stream/eval_phase/test_stream = 0.9421
    Data-incremental (online:
        Top1_Acc_Stream/eval_phase/test_stream = 0.9309

    These are reference results for a single run.
    """
    # --- DEFAULT PARAMS ONLINE DATA INCREMENTAL LEARNING
    nb_tasks = 5  # Can still design the data stream based on tasks
    batch_size = 10  # Learning agent only has small amount of data available
    epochs = 1  # How many times to process each mini-batch
    return_task_id = False  # Data incremental (task-agnostic/task-free)

    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # ---------

    # --- SCENARIO CREATION
    n_classes = 10
    task_scenario = SplitMNIST(
        nb_tasks,
        return_task_id=return_task_id,
        fixed_class_order=[i for i in range(n_classes)],
    )

    # Make data incremental (one batch = one experience)
    scenario = data_incremental_benchmark(
        task_scenario, experience_size=batch_size
    )
    print(f"{scenario.n_experiences} batches in online data incremental setup.")
    # 6002 batches for SplitMNIST with batch size 10
    # ---------

    # MODEL CREATION
    model = SimpleMLP(
        num_classes=args.featsize, hidden_size=400, hidden_layers=2, drop_rate=0
    )

    # choose some metrics and evaluation method
    logger = TextLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(experience=False, stream=True),
        StreamForgetting(),
        loggers=[logger],
        benchmark=scenario,
    )

    # CoPE PLUGIN
    cope = CoPEPlugin(
        mem_size=2000, alpha=0.99, p_size=args.featsize, n_classes=n_classes
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE) WITH CoPE PLUGIN
    cl_strategy = Naive(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        cope.ppp_loss,  # CoPE PPP-Loss
        train_mb_size=batch_size,
        train_epochs=epochs,
        eval_mb_size=100,
        device=device,
        plugins=[cope],
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    cl_strategy.train(scenario.train_stream)

    print("Computing accuracy on the whole test set")
    results.append(cl_strategy.eval(scenario.test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--featsize",
        type=int,
        default=32,
        help="Feature size for the embedding.",
    )
    args = parser.parse_args()
    main(args)
