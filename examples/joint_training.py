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
This is a simple example to show how a simple "offline" upper bound can be
computed. This is useful to see what's the maximum accuracy a model can get
without the hindering of learning continually. This is often referred to as
"cumulative", "joint-training" or "offline" upper bound.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitMNIST, PermutedMNIST
from avalanche.evaluation import EvalProtocol
from avalanche.evaluation.metrics import ACC
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import JointTraining, Naive

def main():

    # Config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    model = SimpleMLP(num_classes=10)

    # CL Benchmark Creation
    perm_mnist = PermutedMNIST(n_steps=5)
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()
    evaluation_protocol = EvalProtocol(
        metrics=[ACC(num_class=10)])

    # Joint training strategy
    joint_train = JointTraining(
        model, optimizer, criterion, train_mb_size=32, train_epochs=1,
        test_mb_size=32, evaluation_protocol=evaluation_protocol, device=device
    )

    # train and test loop
    results = []
    joint_train.train(train_stream, num_workers=4)
    results.append(joint_train.test(test_stream))

if __name__ == '__main__':
    main()
