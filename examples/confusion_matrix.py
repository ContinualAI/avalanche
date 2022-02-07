################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-05-2020                                                             #
# Author(s): Andrea Cossu                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This example shows how to produce confusion matrix during training and
evaluation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import expanduser

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import nc_benchmark
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.evaluation.metrics import (
    confusion_matrix_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
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

    # --- SCENARIO CREATION
    mnist_train = MNIST(
        root=expanduser("~") + "/.avalanche/data/mnist/",
        train=True,
        download=True,
        transform=train_transform,
    )
    mnist_test = MNIST(
        root=expanduser("~") + "/.avalanche/data/mnist/",
        train=False,
        download=True,
        transform=test_transform,
    )
    scenario = nc_benchmark(
        mnist_train, mnist_test, 5, task_labels=False, seed=1234
    )
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=scenario.n_classes)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        # save image should be False to appropriately view
        # results in Interactive Logger.
        # a tensor will be printed
        confusion_matrix_metrics(
            save_image=False, normalize="all", stream=True
        ),
        loggers=InteractiveLogger(),
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=4,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
        plugins=[ReplayPlugin(5000)],
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print("Training completed")

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
    args = parser.parse_args()
    main(args)
