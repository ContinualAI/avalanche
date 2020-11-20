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
This is a simple example on the Permuted MNIST benchmark.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.evaluation import EvalProtocol
from avalanche.evaluation.metrics import ACC
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.training.plugins import MultiHeadPlugin

def main():

    # Config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    model = SimpleMLP(num_classes=10)

    # CL Benchmark Creation
    perm_mnist = PermutedMNIST(incremental_steps=3, seed=1)
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()
    evaluation_protocol = EvalProtocol(
        metrics=[ACC(num_class=10)])

    # Continual learning strategy
    cl_strategy = Naive(
        model, optimizer, criterion, train_mb_size=32, train_epochs=1,
        test_mb_size=32, evaluation_protocol=evaluation_protocol, device=device
    )

    # train and test loop
    results = []
    for train_task in train_stream:
        cl_strategy.train(train_task, num_workers=4)
        results.append(cl_strategy.test(test_stream))

if __name__ == '__main__':
    main()
