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
from collections import defaultdict

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
from avalanche.training.supervised import (
    Naive,
    Replay,
    CWRStar,
    GDumb,
    LwF,
    AGEM,
    GEM,
    EWC,
    SynapticIntelligence,
    JointTraining,
)
from avalanche.training.supervised.ar1 import AR1
from avalanche.training.supervised.cumulative import Cumulative
from avalanche.benchmarks import nc_benchmark, SplitCIFAR10, Experience
from avalanche.training.utils import get_last_fc_layer
from avalanche.evaluation.metrics import StreamAccuracy
from avalanche.benchmarks.utils.data_loader import (
    ReplayDataLoader,
    TaskBalancedDataLoader,
    GroupBalancedDataLoader,
)


def get_fast_benchmark():
    n_samples_per_class = 100
    dataset = make_classification(
        n_samples=10 * n_samples_per_class,
        n_classes=10,
        n_features=6,
        n_informative=6,
        n_redundant=0,
    )

    X = torch.from_numpy(dataset[0]).float()
    y = torch.from_numpy(dataset[1]).long()

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.6, shuffle=True, stratify=y
    )

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    my_nc_benchmark = nc_benchmark(
        train_dataset, test_dataset, 5, task_labels=True
    )
    return my_nc_benchmark


class DataLoaderTests(unittest.TestCase):
    def test_basic(self):
        benchmark = get_fast_benchmark()
        ds = [el.dataset for el in benchmark.train_stream]
        data = AvalancheConcatDataset(ds)
        dl = TaskBalancedDataLoader(data)
        for el in dl:
            pass

        dl = GroupBalancedDataLoader(ds)
        for el in dl:
            pass

        dl = ReplayDataLoader(data, data)
        for el in dl:
            pass

    def test_dataload_reinit(self):
        benchmark = get_fast_benchmark()
        model = SimpleMLP(input_size=6, hidden_size=10)

        replayPlugin = ReplayPlugin(mem_size=5)
        cl_strategy = Naive(
            model,
            SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001),
            CrossEntropyLoss(),
            train_mb_size=16,
            train_epochs=1,
            eval_mb_size=16,
            plugins=[replayPlugin],
        )
        for step in benchmark.train_stream[:2]:
            cl_strategy.train(step)

    def test_dataload_batch_balancing(self):
        benchmark = get_fast_benchmark()
        batch_size = 32
        replayPlugin = ReplayPlugin(mem_size=20)

        model = SimpleMLP(input_size=6, hidden_size=10)
        cl_strategy = Naive(
            model,
            SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001),
            CrossEntropyLoss(),
            train_mb_size=batch_size,
            train_epochs=1,
            eval_mb_size=100,
            plugins=[replayPlugin],
        )
        for step in benchmark.train_stream:
            adapted_dataset = step.dataset
            if len(replayPlugin.storage_policy.buffer) > 0:
                dataloader = ReplayDataLoader(
                        adapted_dataset,
                        replayPlugin.storage_policy.buffer,
                        oversample_small_tasks=True,
                        num_workers=0,
                        batch_size=batch_size,
                        shuffle=True)
            else:
                dataloader = TaskBalancedDataLoader(adapted_dataset)

            for mini_batch in dataloader:
                mb_task_labels = mini_batch[-1]
                lengths = []
                for task_id in adapted_dataset.task_set:
                    len_task = (mb_task_labels == task_id).sum()
                    lengths.append(len_task)
                if sum(lengths) == batch_size:
                    difference = max(lengths) - min(lengths)
                    self.assertLessEqual(difference, 1)
                self.assertLessEqual(sum(lengths), batch_size)
            cl_strategy.train(step)

    def test_dataload_batch_balancing_forced_size(self):
        benchmark = get_fast_benchmark()
        batch_size = 32
        mem_batch_size = 12
        total_mem_size = 100
        replayPlugin = ReplayPlugin(mem_size=total_mem_size)

        model = SimpleMLP(input_size=6, hidden_size=10)
        cl_strategy = Naive(
            model,
            SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001),
            CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=1,
            eval_mb_size=100, plugins=[replayPlugin]
        )
        step: Experience
        for step in benchmark.train_stream:
            adapted_dataset = step.dataset
            if step.current_experience > 0:
                dataloader = ReplayDataLoader(
                    adapted_dataset,
                    replayPlugin.storage_policy.buffer,
                    oversample_small_tasks=False,
                    num_workers=0,
                    batch_size=batch_size,
                    batch_size_mem=mem_batch_size,
                    shuffle=True,
                    drop_last=False)

                current_t_label = step.task_label
                task_n_instances = defaultdict(int)

                for mini_batch in dataloader:
                    self.assertLessEqual(len(mini_batch[0]),
                                         batch_size + mem_batch_size)

                    mb_task_labels = mini_batch[-1]
                    t_label, t_count = torch.unique(
                        mb_task_labels, return_counts=True)

                    for t_l, t_c in zip(t_label, t_count):
                        task_n_instances[int(t_l)] += int(t_c)

                    lengths = []
                    for task_id in adapted_dataset.task_set:
                        len_task = (mb_task_labels == task_id).sum()
                        lengths.append(len_task)
                    if sum(lengths) == batch_size:
                        difference = max(lengths) - min(lengths)
                        self.assertLessEqual(difference, 1)
                    self.assertLessEqual(sum(lengths), batch_size)

                self.assertEqual(len(adapted_dataset),
                                 task_n_instances[current_t_label])

                prev_tasks_instances = sum(
                    n_mem for t_mem, n_mem in
                    task_n_instances.items() if t_mem != current_t_label)

                self.assertEqual(total_mem_size, prev_tasks_instances)

            cl_strategy.train(step)


if __name__ == "__main__":
    unittest.main()
