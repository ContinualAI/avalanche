#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

"""
This is a simple example on how to use the new strategy API.
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
from avalanche.models import SimpleMLP
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

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model, SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(), train_mb_size=100, train_epochs=4, test_mb_size=100,
        device=device)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    args = parser.parse_args()
    main(args)
