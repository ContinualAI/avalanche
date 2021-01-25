#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-11-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

"""
In this simple example we show all the different ways you can use MNIST with
Avalanche.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST, RotatedMNIST, \
    SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive


def main():

    # Device config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    model = SimpleMLP(num_classes=10)

    # Here we show all the MNIST variation we offer in the "classic" benchmarks
    # perm_mnist = PermutedMNIST(n_steps=5, seed=1)
    perm_mnist = RotatedMNIST(n_steps=5, rotations_list=[30, 60, 90, 120, 150], seed=1)
    # perm_mnist = SplitMNIST(n_steps=5, seed=1)

    # Than we can extract the parallel train and test streams
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Continual learning strategy
    cl_strategy = Naive(
        model, optimizer, criterion, train_mb_size=32, train_epochs=2,
        test_mb_size=32, device=device
    )

    # train and test loop
    results = []
    for train_task in train_stream:
        print("Current Classes: ", train_task.classes_in_this_step)
        cl_strategy.train(train_task, num_workers=4)
        results.append(cl_strategy.test(test_stream))

if __name__ == '__main__':
    main()
