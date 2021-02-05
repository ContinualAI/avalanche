#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 26-01-2021                                                            #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

"""
This is a simple example on how to use the Synaptic Intelligence Plugin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import nc_scenario
from avalanche.evaluation.metrics import TaskForgetting, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, TaskConfusionMatrix, \
    DiskUsageMonitor, GpuUsageMonitor, RamUsageMonitor
from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.models import SimpleMLP
from avalanche.logging import DotTrace
from avalanche.training.plugins import EvaluationPlugin, \
    SynapticIntelligencePlugin
from avalanche.training.strategies import Naive


def main():
    # --- CONFIG
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    my_logger = TensorboardLogger(
        tb_log_dir="logs", tb_log_exp_name="logging_example")
    text_logger = DotTrace(stdout=True, trace_file='./logs/my_log.txt')

    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, task=True),
        loss_metrics(minibatch=True, epoch=True, task=True),
        TaskForgetting(),
        TaskConfusionMatrix(num_classes=scenario.n_classes),
        loggers=[my_logger, text_logger])

    # CREATE THE STRATEGY INSTANCE (NAIVE with the Synaptic Intelligence plugin)
    cl_strategy = Naive(
        model, Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(), train_mb_size=64, train_epochs=10, test_mb_size=64,
        device=device, plugins=[evaluation_plugin,
                                SynapticIntelligencePlugin(0.0001)])

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for step in scenario.train_stream:
        print("Start of step: ", step.current_step)
        print("Current Classes: ", step.classes_in_this_step)

        cl_strategy.train(step)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.test(scenario.test_stream))


if __name__ == '__main__':
    main()
