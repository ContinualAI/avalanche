################################################################################
# Copyright (c) 2024 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 31-05-2024                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
An exmaple that shows how to create a class-incremental benchmark from a pytorch dataset.
"""

import torch
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.transforms import Compose, Normalize, ToTensor

from avalanche.benchmarks.datasets import MNIST, default_dataset_location
from avalanche.benchmarks.scenarios import class_incremental_benchmark
from avalanche.benchmarks.utils import (
    make_avalanche_dataset,
    TransformGroups,
    DataAttribute,
)
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive


def main(args):
    # Device config
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    # create pytorch dataset
    train_data = MNIST(root=default_dataset_location("mnist"), train=True)
    test_data = MNIST(root=default_dataset_location("mnist"), train=False)

    # prepare transformations
    train_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    eval_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    tgroups = TransformGroups({"train": train_transform, "eval": eval_transform})

    # create Avalanche datasets with targets attributes (needed to split by class)
    da = DataAttribute(train_data.targets, "targets")
    train_data = make_avalanche_dataset(
        train_data, data_attributes=[da], transform_groups=tgroups
    )

    da = DataAttribute(test_data.targets, "targets")
    test_data = make_avalanche_dataset(
        test_data, data_attributes=[da], transform_groups=tgroups
    )

    # create benchmark
    bm = class_incremental_benchmark(
        {"train": train_data, "test": test_data}, num_experiences=5
    )

    # Continual learning strategy
    model = SimpleMLP(num_classes=10)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()
    cl_strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=32,
        train_epochs=100,
        eval_mb_size=32,
        device=device,
        eval_every=1,
    )

    # train and test loop
    results = []
    for train_task, test_task in zip(bm.train_stream, bm.test_stream):
        print("Current Classes: ", train_task.classes_in_this_experience)
        cl_strategy.train(train_task, eval_streams=[test_task])
        results.append(cl_strategy.eval(bm.test_stream))


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
