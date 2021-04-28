################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-03-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import unittest

import torch
from torchvision.transforms import ToTensor, Compose, transforms, Resize
import os
import sys

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

from avalanche.benchmarks.datasets import MNIST
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.logging import TextLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.strategies import Naive, Replay, CWRStar, \
    GDumb, LwF, AGEM, GEM, EWC, \
    SynapticIntelligence, JointTraining
from avalanche.training.strategies.ar1 import AR1
from avalanche.training.strategies.cumulative import Cumulative
from avalanche.benchmarks import nc_benchmark, SplitCIFAR10
from avalanche.training.utils import get_last_fc_layer
from avalanche.evaluation.metrics import StreamAccuracy
from avalanche.benchmarks.utils.data_loader import \
    MultiTaskJoinedBatchDataLoader


def get_fast_scenario():
    n_samples_per_class = 100
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
    my_nc_benchmark = nc_benchmark(train_dataset, test_dataset, 5,
                                   task_labels=True)
    return my_nc_benchmark


class DataLoaderTests(unittest.TestCase):
    def test_dataload_reinit(self):
        scenario = get_fast_scenario()
        model = SimpleMLP(input_size=6, hidden_size=10)

        replayPlugin = ReplayPlugin(mem_size=5)
        cl_strategy = Naive(
            model,
            SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001),
            CrossEntropyLoss(), train_mb_size=16, train_epochs=1,
            eval_mb_size=16,
            plugins=[replayPlugin]
        )
        for step in scenario.train_stream[:2]:
            cl_strategy.train(step)

    def test_dataload_batch_balancing(self):
        scenario = get_fast_scenario()
        model = SimpleMLP(input_size=6, hidden_size=10)
        batch_size = 32
        replayPlugin = ReplayPlugin(mem_size=20)
        cl_strategy = Naive(
            model, 
            SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001),
            CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=1,
            eval_mb_size=100, plugins=[replayPlugin]
        )

        for step in scenario.train_stream:
            adapted_dataset = step.dataset
            dataloader = MultiTaskJoinedBatchDataLoader(
                    adapted_dataset,
                    AvalancheConcatDataset(replayPlugin.ext_mem.values()),
                    oversample_small_tasks=True,
                    num_workers=0,
                    batch_size=batch_size,
                    shuffle=True)

            for mini_batch in dataloader:
                lengths = []
                for task_id in mini_batch.keys():
                    lengths.append(len(mini_batch[task_id][1]))
                if sum(lengths) == batch_size:
                    difference = max(lengths) - min(lengths)
                    self.assertLessEqual(difference, 1)
                self.assertLessEqual(sum(lengths), batch_size)
            cl_strategy.train(step)


if __name__ == '__main__':
    unittest.main()
