#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 08-02-2021                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

"""
This is a simple example on how to use the AR1 strategy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize

from avalanche.benchmarks import SplitCIFAR10
from avalanche.evaluation.metrics import Forgetting, accuracy_metrics, \
    loss_metrics, TaskConfusionMatrix
from avalanche.logging import InteractiveLogger
from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import AR1


def main():
    # --- CONFIG
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ---------

    # --- TRANSFORMATIONS
    train_transform = transforms.Compose([
        Resize(224),
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        Resize(224),
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # ---------

    # --- SCENARIO CREATION
    scenario = SplitCIFAR10(5, train_transform=train_transform,
                            test_transform=test_transform)
    # ---------

    # DEFINE THE EVALUATION PLUGIN AND LOGGER
    my_logger = TensorboardLogger(
        tb_log_dir="logs", tb_log_exp_name="logging_example")

    # print to stdout
    interactive_logger = InteractiveLogger()

    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, task=True),
        loss_metrics(minibatch=True, epoch=True, task=True),
        Forgetting(compute_for_step=True),
        TaskConfusionMatrix(num_classes=scenario.n_classes),
        loggers=[my_logger, interactive_logger])

    # CREATE THE STRATEGY INSTANCE
    cl_strategy = AR1(criterion=CrossEntropyLoss(), device=device,
                      evaluator=evaluation_plugin)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for step in scenario.train_stream:
        print("Start of step: ", step.current_step)
        print("Current Classes: ", step.classes_in_this_step)

        cl_strategy.train(step, num_workers=4)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.test(scenario.test_stream, num_workers=4))


if __name__ == '__main__':
    main()
