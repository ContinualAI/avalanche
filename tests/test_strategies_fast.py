#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-06-2020                                                              #
# Author(s): Andrea Cossu                                                      #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

import unittest

import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive, Replay, CWRStar, \
    GDumb, Cumulative, LwF, AGEM
from avalanche.benchmarks import nc_scenario


class BaseStrategyTest(unittest.TestCase):

    def _is_param_in_optimizer(self, param, optimizer):
        for group in optimizer.param_groups:
            for curr_p in group['params']:
                if hash(curr_p) == hash(param):
                    return True
        return False

    def test_optimizer_update(self):
        model = SimpleMLP()
        optimizer = SGD(model.parameters(), lr=1e-3)
        strategy = Naive(model, optimizer, None)

        # check add_param_group
        p = torch.nn.Parameter(torch.zeros(10, 10))
        strategy.add_new_params_to_optimizer(p)
        assert self._is_param_in_optimizer(p, strategy.optimizer)

        # check new_param is in optimizer
        # check old_param is NOT in optimizer
        p_new = torch.nn.Parameter(torch.zeros(10, 10))
        strategy.update_optimizer([p], [p_new])
        assert self._is_param_in_optimizer(p_new, strategy.optimizer)
        assert not self._is_param_in_optimizer(p, strategy.optimizer)


class StrategyTest(unittest.TestCase):

    def test_naive(self):
        model = SimpleMLP(input_size=6)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        my_nc_scenario = self.load_scenario()

        strategy = Naive(model, optimizer, criterion, train_mb_size=64)
        self.run_strategy(my_nc_scenario, strategy)

    def test_cwrstar(self):
        model = SimpleMLP(input_size=6)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        my_nc_scenario = self.load_scenario()

        strategy = CWRStar(model, optimizer, criterion, 'features.0.bias',
                           train_mb_size=64)
        self.run_strategy(my_nc_scenario, strategy)

    def test_replay(self):
        model = SimpleMLP(input_size=6)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        my_nc_scenario = self.load_scenario()

        strategy = Replay(model, optimizer, criterion,
                          mem_size=200, train_mb_size=64)
        self.run_strategy(my_nc_scenario, strategy)
    
    def test_gdumb(self):
        model = SimpleMLP(input_size=6)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        my_nc_scenario = self.load_scenario()

        strategy = GDumb(
                model, optimizer, criterion,
                mem_size=200, train_mb_size=64)
        self.run_strategy(my_nc_scenario, strategy)

    def test_cumulative(self):
        model = SimpleMLP(input_size=6)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        my_nc_scenario = self.load_scenario()

        strategy = Cumulative(model, optimizer, criterion, train_mb_size=64)
        self.run_strategy(my_nc_scenario, strategy)
    
    def test_lwf(self):
        model = SimpleMLP(input_size=6)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        my_nc_scenario = self.load_scenario()

        strategy = LwF(model, optimizer, criterion,
                       alpha=[0, 1/2, 2*(2/3), 3*(3/4), 4*(4/5)], 
                       temperature=2, train_mb_size=64)
        self.run_strategy(my_nc_scenario, strategy)
    
    def test_agem(self):
        model = SimpleMLP(input_size=6)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        my_nc_scenario = self.load_scenario()

        strategy = AGEM(model, optimizer, criterion,
                        patterns_per_step=500/5,
                        train_mb_size=64)

        self.run_strategy(my_nc_scenario, strategy)

    def load_scenario(self):
        """
        Returns a NC Scenario from a fake dataset of 10 classes, 5 steps,
        2 classes per step.
        """

        n_samples_per_class = 200

        dataset = make_classification(
            n_samples=10 * n_samples_per_class,
            n_classes=10,
            n_features=6, n_informative=6, n_redundant=0)

        X = torch.from_numpy(dataset[0]).float()
        y = torch.from_numpy(dataset[1]).long()

        train_X, test_X, train_y, test_y = train_test_split(
            X, y, train_size=0.6, shuffle=True, stratify=y)

        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        return nc_scenario(train_dataset, test_dataset, 5, task_labels=False)

    def run_strategy(self, scenario, cl_strategy):
        print('Starting experiment...')
        results = []
        for train_batch_info in scenario.train_stream:
            print("Start of step ", train_batch_info.current_step)

            cl_strategy.train(train_batch_info, num_workers=4)
            print('Training completed')

            print('Computing accuracy on the current test set')
            results.append(cl_strategy.test(scenario.test_stream[:]))


if __name__ == '__main__':
    unittest.main()
