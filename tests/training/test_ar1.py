################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-05-2021                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import torch
import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from avalanche.benchmarks import nc_benchmark
from avalanche.training.supervised import AR1
from tests.training.test_strategies import StrategyTest


class AR1Test(unittest.TestCase):
    def test_ar1(self):
        my_nc_benchmark = self.load_ar1_benchmark()
        strategy = AR1(
            train_epochs=1, train_mb_size=10, eval_mb_size=10, rm_sz=20
        )
        StrategyTest.run_strategy(self, my_nc_benchmark, strategy)

    def load_ar1_benchmark(self):
        """
        Returns a NC benchmark from a fake dataset of 10 classes, 5 experiences,
        2 classes per experience. This toy benchmark is intended
        """
        n_samples_per_class = 5
        dataset = make_classification(
            n_samples=10 * n_samples_per_class,
            n_classes=9,
            n_features=224 * 224 * 3,
            n_informative=6,
            n_redundant=0,
        )

        X = torch.from_numpy(dataset[0]).reshape(-1, 3, 224, 224).float()
        y = torch.from_numpy(dataset[1]).long()

        train_X, test_X, train_y, test_y = train_test_split(
            X, y, train_size=0.6, shuffle=True, stratify=y
        )

        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        my_nc_benchmark = nc_benchmark(
            train_dataset, test_dataset, 3, task_labels=False
        )
        return my_nc_benchmark
