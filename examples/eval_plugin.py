################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

"""
This is a simple example on how to use the Evaluation Plugin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import nc_scenario
from avalanche.evaluation.metrics import StepForgetting, accuracy_metrics, \
    loss_metrics, cpu_usage_metrics, timing_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive


def main(args):
    # --- CONFIG
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    # ---------

    # --- TRANSFORMATIONS
    train_transform = transforms.Compose([
        RandomCrop(28, padding=4),
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # ---------

    # --- SCENARIO CREATION
    mnist_train = MNIST('./data/mnist', train=True,
                        download=True, transform=train_transform)
    mnist_test = MNIST('./data/mnist', train=False,
                       download=True, transform=test_transform)
    scenario = nc_scenario(
        mnist_train, mnist_test, 5, task_labels=False, seed=1234)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=scenario.n_classes)

    # DEFINE THE EVALUATION PLUGIN AND LOGGER
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics and a list of loggers.
    # The evaluation plugin calls the loggers to serialize the metrics
    # and save them in persistent memory or print them in the standard output.

    # log to text file
    text_logger = TextLogger(open('log.txt', 'a'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, step=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, step=True, stream=True),
        cpu_usage_metrics(minibatch=True, epoch=True, step=True, stream=True),
        timing_metrics(minibatch=True, epoch=True, step=True, stream=True),
        StepForgetting(),
        loggers=[interactive_logger, text_logger])


    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model, SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
        device=device, evaluator=eval_plugin)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for step in scenario.train_stream:
        print("Start of step: ", step.current_step)
        print("Current Classes: ", step.classes_in_this_step)

        # train returns a list of dictionaries (one for each step). Each
        # dictionary stores the last value of each metric curve emitted
        # during training.
        res = cl_strategy.train(step)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        # test also returns a dictionary
        results.append(cl_strategy.eval(scenario.test_stream))

    print(f"Test metrics:\n{results}")

    # All the metric curves (x,y values) are stored inside the evaluator
    # (can be disabled). You can use this dictionary to manipulate the
    # metrics without avalanche.
    all_metrics = cl_strategy.evaluator.all_metrics
    print(f"Stored metrics: {list(all_metrics.keys())}")
    mname = 'Top1_Acc_Task/Task000'
    print(f"{mname}: {cl_strategy.evaluator.all_metrics[mname]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    args = parser.parse_args()
    main(args)
