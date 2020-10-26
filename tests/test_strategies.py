#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-06-2020                                                              #
# Author(s): Andrea Cossu                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

import unittest

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from avalanche.extras.models import SimpleMLP
from avalanche.evaluation import EvalProtocol
from avalanche.evaluation.metrics import ACC
from avalanche.benchmarks.scenarios import \
    create_nc_single_dataset_sit_scenario, DatasetPart, NCBatchInfo
from avalanche.training.strategies import Naive, Cumulative, Replay, GDumb
#from avalanche.training.plugins import ReplayPlugin, GDumbPlugin

device = 'cpu'

class StrategyTest(unittest.TestCase):

    def test_naive(self):
        model = SimpleMLP()
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        mnist_train, mnist_test = self.load_dataset()
        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        eval_protocol = EvalProtocol(
            metrics=[
                ACC(num_class=nc_scenario.n_classes)
            ])

        strategy = Naive(model, 'classifier', optimizer, criterion,
                evaluation_protocol=eval_protocol, train_mb_size=100, 
                train_epochs=4, test_mb_size=100, device=device)

        self.run_strategy(nc_scenario, strategy)


    def test_replay(self):
        model = SimpleMLP()
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        mnist_train, mnist_test = self.load_dataset()
        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        eval_protocol = EvalProtocol(
            metrics=[
                ACC(num_class=nc_scenario.n_classes)
            ])

        strategy = Replay(model, 'classifier', optimizer, criterion,
                mem_size=200,
                evaluation_protocol=eval_protocol,
                train_mb_size=100, 
                train_epochs=4, test_mb_size=100, device=device, plugins=None
                )

        self.run_strategy(nc_scenario, strategy)


    def test_cumulative(self):
        model = SimpleMLP()
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        mnist_train, mnist_test = self.load_dataset()
        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        eval_protocol = EvalProtocol(
            metrics=[
                ACC(num_class=nc_scenario.n_classes)
            ])

        strategy = Cumulative(model, 'classifier', optimizer, criterion,
                train_mb_size=100, 
                evaluation_protocol=eval_protocol,
                train_epochs=4, test_mb_size=100, device=device)

        self.run_strategy(nc_scenario, strategy)


    def test_gdumb(self):
        model = SimpleMLP()
        optimizer = SGD(model.parameters(), lr=1e-2)
        criterion = CrossEntropyLoss()
        mnist_train, mnist_test = self.load_dataset()
        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 5, seed=1234)

        eval_protocol = EvalProtocol(
            metrics=[
                ACC(num_class=nc_scenario.n_classes)
            ])

        strategy = GDumb(model, 'classifier', optimizer, criterion,
                mem_size=2000,
                train_mb_size=64,
                evaluation_protocol=eval_protocol,
                train_epochs=10, test_mb_size=100, device=device,
                plugins=None)

        print('Starting experiment...')
        results = []
        batch_info: NCBatchInfo
        for batch_info in nc_scenario:
            print("Start of step ", batch_info.current_step)

            # retrain from scratch each step
            strategy.model = SimpleMLP()
            strategy.optimizer = SGD(strategy.model.parameters(), lr=1e-2)
            strategy.train(batch_info, num_workers=4)
            print('Training completed')

            print('Computing accuracy on the whole test set')
            results.append(strategy.test(batch_info, DatasetPart.COMPLETE,
                                            num_workers=4))


    def load_dataset(self):

        mnist_train = MNIST('./data/mnist', train=True, download=True, 
                transform=Compose([ToTensor()]))
        mnist_test = MNIST('./data/mnist', train=False, download=True,
                transform=Compose([ToTensor()]))
        return mnist_train, mnist_test


    def run_strategy(self, scenario, cl_strategy):

        print('Starting experiment...')
        results = []
        batch_info: NCBatchInfo
        for batch_info in scenario:
            print("Start of step ", batch_info.current_step)

            cl_strategy.train(batch_info, num_workers=4)
            print('Training completed')

            print('Computing accuracy on the whole test set')
            results.append(cl_strategy.test(batch_info, DatasetPart.COMPLETE,
                                            num_workers=4))


if __name__ == '__main__':
    unittest.main()
