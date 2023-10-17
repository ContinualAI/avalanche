################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the new strategy API.
"""

from os.path import expanduser

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import AvalancheDataset
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils import as_supervised_dataset
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    # ---------

    # --- TRANSFORMATIONS
    train_transform = transforms.Compose(
        [
            RandomCrop(28, padding=4),
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = transforms.Compose(
        [ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # ---------

    # --- BENCHMARK CREATION
    mnist_train = as_supervised_dataset(
        MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
            transform=train_transform,
        )
    )
    mnist_test = as_supervised_dataset(
        MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
            transform=test_transform,
        )
    )
    benchmark = class_incremental_benchmark(
        datasets_dict={"train": mnist_train, "test": mnist_test},
        num_experiences=5,
        seed=1234,
    )
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=benchmark.n_classes)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=4,
        eval_mb_size=100,
        device=device,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

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
    main(args)
