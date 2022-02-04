################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-11-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
In this simple example we show all the different ways you can use MNIST with
Avalanche.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST, RotatedMNIST, SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive


def main(args):
    # Device config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # model
    model = SimpleMLP(num_classes=10)

    # Here we show all the MNIST variation we offer in the "classic" benchmarks
    if args.mnist_type == "permuted":
        scenario = PermutedMNIST(n_experiences=5, seed=1)
    elif args.mnist_type == "rotated":
        scenario = RotatedMNIST(
            n_experiences=5, rotations_list=[30, 60, 90, 120, 150], seed=1
        )
    else:
        scenario = SplitMNIST(n_experiences=5, seed=1)

    # Than we can extract the parallel train and test streams
    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Continual learning strategy with default logger
    cl_strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=32,
        train_epochs=2,
        eval_mb_size=32,
        device=device,
    )

    # train and test loop
    results = []
    for train_task in train_stream:
        print("Current Classes: ", train_task.classes_in_this_experience)
        cl_strategy.train(train_task)
        results.append(cl_strategy.eval(test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnist_type",
        type=str,
        default="split",
        choices=["rotated", "permuted", "split"],
        help="Choose between MNIST variations: " "rotated, permuted or split.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()

    main(args)
