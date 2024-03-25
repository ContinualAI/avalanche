################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 25-03-2024                                                             #
# Author(s): Albin Soutif                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This example trains a Multi-head model on Split MNIST with Elastic Weight
Consolidation. Each experience has a different task label, which is used at test
time to select the appropriate head. Additionally, it assigns different parameter groups
to the classifier and the backbone, assigning lower learning rate to 
the backbone than to the classifier. When the multihead classifier grows, 
new parameters are automatically assigned to the corresponding parameter group
"""

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import MTSimpleMLP
from avalanche.training.supervised import EWC
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # Config
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    # model
    model = MTSimpleMLP()

    # CL Benchmark Creation
    benchmark = SplitMNIST(n_experiences=5, return_task_id=True)
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    g1 = {"params": [], "lr": 0.1}
    g2 = {"params": [], "lr": 0.01}

    for n, p in model.named_parameters():
        if "classifier" in n:
            g1["params"].append(p)
        else:
            g2["params"].append(p)

    optimizer = SGD([g1, g2])
    criterion = CrossEntropyLoss()

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # Choose a CL strategy
    strategy = EWC(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=128,
        train_epochs=3,
        eval_mb_size=128,
        device=device,
        evaluator=eval_plugin,
        ewc_lambda=0.4,
    )

    # train and test loop
    for train_task in train_stream:
        strategy.train(train_task, num_workers=4, verbose_optimizer=True)
        strategy.eval(test_stream, num_workers=4)


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
