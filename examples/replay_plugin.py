#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-10-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

"""
This is a simple example on how to use the new strategy API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks.scenarios import DatasetPart, \
    create_nc_single_dataset_sit_scenario, NCBatchInfo
from avalanche.evaluation import EvalProtocol
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.training.plugins import ReplayPlugin


def main():
    # --- CONFIG
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_batches = 5
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
    nc_scenario = create_nc_single_dataset_sit_scenario(
        mnist_train, mnist_test, n_batches, shuffle=True, seed=1234)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=nc_scenario.n_classes)

    # DEFINE THE EVALUATION PROTOCOL
    evaluation_protocol = EvalProtocol(
        metrics=[ACC(num_class=nc_scenario.n_classes),  # Accuracy metric
                 CF(num_class=nc_scenario.n_classes),  # Catastrophic forgetting
                 RAMU(),  # Ram usage
                 CM()],  # Confusion matrix
        tb_logdir='../logs/mnist_test_sit')

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model, 'classifier', SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(), train_mb_size=100, train_epochs=4, test_mb_size=100,
        evaluation_protocol=evaluation_protocol, device=device,
        plugins=[ReplayPlugin(mem_size=10000)]
    )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    batch_info: NCBatchInfo
    for batch_info in nc_scenario:
        print("Start of step ", cl_strategy.training_step_id + 1)

        cl_strategy.train(batch_info, num_workers=4)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.test(batch_info, DatasetPart.COMPLETE,
                                        num_workers=4))


if __name__ == '__main__':
    main()
