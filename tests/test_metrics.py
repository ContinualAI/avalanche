""" Metrics Tests"""

import unittest
import torch
from torch.utils.data import TensorDataset
import numpy as np
import random
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from copy import deepcopy
from avalanche.evaluation.metrics import (
    Accuracy,
    Loss,
    ConfusionMatrix,
    DiskUsage,
    MAC,
    CPUUsage,
    MaxGPU,
    MaxRAM,
    Mean,
    Sum,
    ElapsedTime,
    Forgetting,
    ForwardTransfer,
)
from avalanche.training.templates.supervised import SupervisedTemplate
import pathlib
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.utils import (
    AvalancheTensorDataset,
    AvalancheDatasetType,
)
from avalanche.benchmarks import nc_benchmark, dataset_benchmark
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    cpu_usage_metrics,
    timing_metrics,
    ram_usage_metrics,
    disk_usage_metrics,
    MAC_metrics,
    bwt_metrics,
    confusion_matrix_metrics,
    forward_transfer_metrics,
)
from avalanche.models import SimpleMLP
from avalanche.logging import TextLogger
from avalanche.training.plugins import EvaluationPlugin


#################################
#################################
#    STANDALONE METRIC TEST     #
#################################
#################################
from tests.unit_tests_utils import UPDATE_METRICS


class GeneralMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 3
        self.input_size = 10
        self.n_classes = 3
        self.n_tasks = 2
        self.out = torch.randn(self.batch_size, self.input_size)
        self.y = torch.randint(0, self.n_classes, (self.batch_size,))
        self.task_labels = torch.randint(0, self.n_tasks, (self.batch_size,))

    def test_accuracy(self):
        metric = Accuracy()
        self.assertEqual(metric.result(), {})
        metric.update(self.out, self.y, 0)
        self.assertLessEqual(metric.result(0)[0], 1)
        self.assertGreaterEqual(metric.result(0)[0], 0)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_accuracy_task_per_pattern(self):
        metric = Accuracy()
        self.assertEqual(metric.result(), {})
        metric.update(self.out, self.y, self.task_labels)
        out = metric.result()
        for k, v in out.items():
            self.assertIn(k, self.task_labels.tolist())
            self.assertLessEqual(v, 1)
            self.assertGreaterEqual(v, 0)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_loss(self):
        metric = Loss()
        self.assertEqual(metric.result(0)[0], 0)
        metric.update(torch.tensor(1.0), self.batch_size, 0)
        self.assertGreaterEqual(metric.result(0)[0], 0)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_loss_multi_task(self):
        metric = Loss()
        self.assertEqual(metric.result(), {})
        metric.update(torch.tensor(1.0), 1, 0)
        metric.update(torch.tensor(2.0), 1, 1)
        out = metric.result()
        for k, v in out.items():
            self.assertIn(k, [0, 1])
            if k == 0:
                self.assertEqual(v, 1)
            else:
                self.assertEqual(v, 2)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_cm(self):
        metric = ConfusionMatrix()
        cm = metric.result()
        self.assertTrue((cm == 0).all().item())
        metric.update(self.y, self.out)
        cm = metric.result()
        self.assertTrue((cm >= 0).all().item())
        metric.reset()
        cm = metric.result()
        self.assertTrue((cm == 0).all().item())

    def test_ram(self):
        metric = MaxRAM()
        self.assertEqual(metric.result(), 0)
        metric.start_thread()  # start thread
        self.assertGreaterEqual(metric.result(), 0)
        metric.stop_thread()  # stop thread
        metric.reset()  # stop thread
        self.assertEqual(metric.result(), 0)

    def test_gpu(self):
        if torch.cuda.is_available():
            metric = MaxGPU(0)
            self.assertEqual(metric.result(), 0)
            metric.start_thread()  # start thread
            self.assertGreaterEqual(metric.result(), 0)
            metric.stop_thread()  # stop thread
            metric.reset()  # stop thread
            self.assertEqual(metric.result(), 0)

    def test_cpu(self):
        metric = CPUUsage()
        self.assertEqual(metric.result(), 0)
        metric.update()
        self.assertGreaterEqual(metric.result(), 0)
        metric.reset()
        self.assertEqual(metric.result(), 0)

    def test_disk(self):
        metric = DiskUsage()
        self.assertEqual(metric.result(), 0)
        metric.update()
        self.assertGreaterEqual(metric.result(), 0)
        metric.reset()
        self.assertEqual(metric.result(), 0)

    def test_timing(self):
        metric = ElapsedTime()
        self.assertEqual(metric.result(), 0)
        metric.update()  # need two update calls
        self.assertEqual(metric.result(), 0)
        metric.update()
        self.assertGreaterEqual(metric.result(), 0)
        metric.reset()
        self.assertEqual(metric.result(), 0)

    def test_mac(self):
        model = torch.nn.Linear(self.input_size, 2)
        metric = MAC()
        self.assertEqual(metric.result(), 0)
        metric.update(model, self.out)
        self.assertGreaterEqual(metric.result(), 0)

    def test_mean(self):
        metric = Mean()
        self.assertEqual(metric.result(), 0)
        metric.update(0.1, 1)
        self.assertEqual(metric.result(), 0.1)
        metric.reset()
        self.assertEqual(metric.result(), 0)

    def test_sum(self):
        metric = Sum()
        self.assertEqual(metric.result(), 0)
        metric.update(5)
        self.assertEqual(metric.result(), 5)
        metric.reset()
        self.assertEqual(metric.result(), 0)

    def test_forgetting(self):
        metric = Forgetting()
        f = metric.result()
        self.assertEqual(f, {})
        f = metric.result(k=0)
        self.assertIsNone(f)
        metric.update(0, 1, initial=True)
        f = metric.result(k=0)
        self.assertIsNone(f)
        metric.update(0, 0.4)
        f = metric.result(k=0)
        self.assertEqual(f, 0.6)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_forward_transfer(self):
        metric = ForwardTransfer()
        f = metric.result()
        self.assertEqual(f, {})
        f = metric.result(k=0)
        self.assertIsNone(f)
        metric.update(0, 1, initial=True)
        f = metric.result(k=0)
        self.assertIsNone(f)
        metric.update(0, 0.4)
        f = metric.result(k=0)
        self.assertEqual(f, -0.6)
        metric.reset()
        self.assertEqual(metric.result(), {})


if __name__ == "__main__":
    unittest.main()
