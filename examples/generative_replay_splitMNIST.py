################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 13-02-2024                                                             #
# Author(s): Imron Gamidli                                                     #
#                                                                              #
################################################################################

"""
This is a simple example on how to use the GenerativeReplay strategy with weighted replay loss.
"""
import datetime
import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler
from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import GenerativeReplay
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    # --- BENCHMARK CREATION
    benchmark = SplitMNIST(
        n_experiences=5, seed=1234, fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=benchmark.n_classes, hidden_size=10)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    file_name = "logs/log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    text_logger = TextLogger(open(file_name, "a"))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(minibatch=True),
        loggers=[interactive_logger, text_logger],
    )

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = GenerativeReplay(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=2,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
        is_weighted_replay=True,
        weight_replay_loss_factor=2.0,
        weight_replay_loss=0.001,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
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
    print(args)
    main(args)
