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
This is a simple example on how to use the Evaluation Plugin.
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
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    cpu_usage_metrics,
    timing_metrics,
    gpu_usage_metrics,
    ram_usage_metrics,
    disk_usage_metrics,
    MAC_metrics,
    bwt_metrics,
    forward_transfer_metrics,
)
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive


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

    # DEFINE THE EVALUATION PLUGIN AND LOGGER
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics and a list of loggers.
    # The evaluation plugin calls the loggers to serialize the metrics
    # and save them in persistent memory or print them in the standard output.

    # log to text file
    text_logger = TextLogger(open("log.txt", "a"))

    # print to stdout
    interactive_logger = InteractiveLogger()

    csv_logger = CSVLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        forward_transfer_metrics(experience=True, stream=True),
        cpu_usage_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        timing_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        ram_usage_metrics(
            every=0.5, minibatch=True, epoch=True, experience=True, stream=True
        ),
        gpu_usage_metrics(
            args.cuda,
            every=0.5,
            minibatch=True,
            epoch=True,
            experience=True,
            stream=True,
        ),
        disk_usage_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        MAC_metrics(minibatch=True, epoch=True, experience=True),
        loggers=[interactive_logger, text_logger, csv_logger],
        collect_all=True,
    )  # collect all metrics (set to True by default)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(),
        train_mb_size=500,
        train_epochs=1,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
        eval_every=1,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for i, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary containing last recorded value
        # for each metric.
        res = cl_strategy.train(experience, eval_streams=[scenario.test_stream])
        print("Training completed")

        print("Computing accuracy on the whole test set")
        # test returns a dictionary with the last metric collected during
        # evaluation on that stream
        results.append(cl_strategy.eval(scenario.test_stream))

    print(f"Test metrics:\n{results}")

    # Dict with all the metric curves,
    # only available when `collect_all` is True.
    # Each entry is a (x, metric value) tuple.
    # You can use this dictionary to manipulate the
    # metrics without avalanche.
    all_metrics = cl_strategy.evaluator.get_all_metrics()
    print(f"Stored metrics: {list(all_metrics.keys())}")


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
