""" Metrics Tests"""
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import List

import torch
from torch import nn

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import Accuracy, Loss, ConfusionMatrix, \
    DiskUsage, MAC, accuracy_metrics, loss_metrics
from avalanche.evaluation.metrics.cpu_usage import CpuUsage, cpu_usage_metrics
from avalanche.evaluation.metrics.ram_usage import RamUsage


class MACMetricTests(unittest.TestCase):
    def test_ff_model(self):
        xn, hn, yn = 50, 100, 10

        model = nn.Sequential(
            nn.Linear(xn, hn),  # 5'000 mul
            nn.ReLU(),
            nn.Linear(hn, hn),  # 10'000 mul
            nn.ReLU(),
            nn.Linear(hn, yn)  # 1'000 mul
        )  # 16'000 mul
        dummy = torch.randn(32, xn)
        met = MAC()
        # met.update(model, dummy)
        # self.assertEqual(16000, met.result())

    def test_cnn_model(self):
        xn, hn, yn = 50, 100, 10
        model = nn.Sequential(
            nn.Conv2d(1, 10, 4),  # 353'440 mul
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22090, hn),  # 2'209'000 mul
            nn.Linear(hn, yn)  # 1'000 mul
        )  # 2'563'440 mul
        dummy = torch.randn(32, 1, xn, xn)
        met = MAC()
        # print(2563440, met.compute(model, dummy))


class AccuracyMetricTests(unittest.TestCase):
    def test_standalone_accuracy(self):
        uut = Accuracy()

        # Initial accuracy should be 0
        self.assertEqual(0.0, uut.result())

        truth = torch.as_tensor([0, 5, 2, 1, 0])
        predicted = torch.as_tensor([2, 3, 2, 5, 0])  # correct 2/5 = 40%
        uut.update(truth, predicted)

        self.assertEqual(0.4, uut.result())

        truth = torch.as_tensor([0, 3, 2, 1, 0])
        predicted = torch.as_tensor([2, 3, 2, 5, 0])  # correct 3/5 = 60%
        uut.update(truth, predicted)

        self.assertEqual(0.5, uut.result())

        # After-reset accuracy should be 0
        uut.reset()
        self.assertEqual(0.0, uut.result())

        # Check if handles 0 accuracy
        truth = torch.as_tensor([0, 0, 0, 0])
        predicted = torch.as_tensor([1, 1, 1, 1])  # correct 0/4 = 0%
        uut.update(truth, predicted)

        self.assertEqual(0.0, uut.result())

        # Should throw exception when len(truth) != len(predicted)
        with self.assertRaises(ValueError):
            truth = torch.as_tensor([0, 0, 1, 0])
            predicted = torch.as_tensor([1, 1, 1])
            uut.update(truth, predicted)

        # Check accuracy didn't change after error
        self.assertEqual(0.0, uut.result())

        # Test logits / one-hot support
        uut.reset()

        # Test one-hot (truth)
        truth = torch.as_tensor([[0, 1], [1, 0], [1, 0], [1, 0]])
        predicted = torch.as_tensor([1, 1, 1, 1])  # correct 1/4 = 25%
        uut.update(truth, predicted)
        self.assertEqual(0.25, uut.result())

        # Test logits (predictions)
        truth = torch.as_tensor([1, 1, 0, 0])
        predicted = torch.as_tensor(
            [[0.73, 0.1], [0.22, 0.33],
             [0.99, 0.01], [0.12, 0.11]])  # correct 3/4 = 75%
        uut.update(truth, predicted)
        self.assertEqual(0.5, uut.result())

        # Test one-hot (truth) + logits (predictions)
        truth = torch.as_tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
        predicted = torch.as_tensor(
            [[0.73, 0.1], [0.22, 0.33],
             [0.99, 0.01], [0.12, 0.11]])  # correct 1/4 = 25%
        uut.update(truth, predicted)
        self.assertEqual(5.0/12.0, uut.result())

    def test_accuracy_helper(self):
        metrics = accuracy_metrics(minibatch=True, epoch=True)
        self.assertEqual(2, len(metrics))
        self.assertIsInstance(metrics, List)
        self.assertIsInstance(metrics[0], PluginMetric)
        self.assertIsInstance(metrics[1], PluginMetric)

        with self.assertRaises(ValueError):
            accuracy_metrics(train=False, test=False)

        with self.assertRaises(ValueError):
            accuracy_metrics(task=True, test=False)


class LossMetricTests(unittest.TestCase):
    def test_standalone_forgetting(self):
        uut = Loss()

        # Initial loss should be 0
        self.assertEqual(0.0, uut.result())

        loss = torch.as_tensor([0.0, 0.1, 0.2, 0.3, 0.4])  # Avg = 0.2
        uut.update(loss, 5)

        self.assertAlmostEqual(0.2, uut.result())

        loss = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5])  # Avg = 0.3
        uut.update(loss, 5)

        self.assertAlmostEqual(0.25, uut.result())

        # After-reset loss should be 0
        uut.reset()
        self.assertEqual(0.0, uut.result())

        # Check if handles 0 loss
        loss = torch.as_tensor([0.0, 0.0, 0.0])
        uut.update(loss, 3)

        self.assertEqual(0.0, uut.result())

        # Check handles losses with different reductions
        uut.reset()
        loss = torch.rand((5, 20))
        expected_mean = loss.mean().item()
        uut.update(loss, loss.shape[0])

        self.assertAlmostEqual(expected_mean, uut.result())

        # Check that the last call to result didn't change the value
        self.assertAlmostEqual(expected_mean, uut.result())

    def test_loss_helper(self):
        metrics = loss_metrics(minibatch=True, epoch_running=True)
        self.assertEqual(2, len(metrics))
        self.assertIsInstance(metrics, List)
        self.assertIsInstance(metrics[0], PluginMetric)
        self.assertIsInstance(metrics[1], PluginMetric)

        with self.assertRaises(ValueError):
            loss_metrics(train=False, test=False)

        with self.assertRaises(ValueError):
            loss_metrics(task=True, test=False)


class ConfusionMatrixMetricTests(unittest.TestCase):
    def test_standalone_cm_fixed_size(self):
        uut = ConfusionMatrix(num_classes=10)

        # Initial confusion matrix should be a 10x10 matrix filled with zeros
        self.assertTrue(torch.equal(torch.zeros((10, 10), dtype=torch.long),
                                    uut.result()))

        truth = torch.as_tensor([0, 5, 2, 1, 0])
        predicted = torch.as_tensor([2, 3, 2, 5, 0])
        uut.update(truth, predicted)

        expected = torch.zeros((10, 10), dtype=torch.long)
        expected[0][2] += 1
        expected[5][3] += 1
        expected[2][2] += 1
        expected[1][5] += 1
        expected[0][0] += 1

        self.assertTrue(torch.equal(expected, uut.result()))

        # After-reset matrix should be a 10x10 matrix filled with zeros
        uut.reset()
        self.assertTrue(torch.equal(torch.zeros((10, 10), dtype=torch.long),
                                    uut.result()))

        # Should throw exception when len(truth) != len(predicted)
        with self.assertRaises(ValueError):
            truth = torch.as_tensor([0, 0, 1, 0])
            predicted = torch.as_tensor([1, 1, 1])
            uut.update(truth, predicted)

        # Check that matrix didn't change after error
        self.assertTrue(torch.equal(torch.zeros((10, 10), dtype=torch.long),
                                    uut.result()))

        # Test logits / one-hot support
        uut.reset()

        # Test one-hot (truth)
        truth = torch.as_tensor([[0, 1], [1, 0], [1, 0], [1, 0]])
        predicted = torch.as_tensor([1, 1, 1, 1])
        uut.update(truth, predicted)

        expected = torch.zeros((10, 10), dtype=torch.long)
        expected[1][1] += 1
        expected[0][1] += 1
        expected[0][1] += 1
        expected[0][1] += 1

        self.assertTrue(torch.equal(expected, uut.result()))

        # Test logits (predictions)
        truth = torch.as_tensor([1, 1, 0, 0])
        predicted = torch.as_tensor(
            [[0.73, 0.1], [0.22, 0.33],
             [0.99, 0.01], [0.12, 0.11]])
        uut.update(truth, predicted)

        expected[1][0] += 1
        expected[1][1] += 1
        expected[0][0] += 1
        expected[0][0] += 1

        self.assertTrue(torch.equal(expected, uut.result()))

        # Test one-hot (truth) + logits (predictions)
        truth = torch.as_tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
        predicted = torch.as_tensor(
            [[0.73, 0.1], [0.22, 0.33],
             [0.99, 0.01], [0.12, 0.11]])
        uut.update(truth, predicted)

        expected[0][0] += 1
        expected[0][1] += 1
        expected[1][0] += 1
        expected[1][0] += 1

        self.assertTrue(torch.equal(expected, uut.result()))

    def test_standalone_cm_error_handling(self):
        uut = ConfusionMatrix(num_classes=10)

        # Initial confusion matrix should be a 10x10 matrix filled with zeros
        self.assertTrue(torch.equal(torch.zeros((10, 10), dtype=torch.long),
                                    uut.result()))

        truth = torch.as_tensor([0, 5, 2, 1, 0])
        predicted = torch.as_tensor([2, 3, -1, 5, 0])

        with self.assertRaises(ValueError):
            uut.update(truth, predicted)

        truth = torch.as_tensor([0, 5, 2, 1, -5])
        predicted = torch.as_tensor([2, 3, 1, 5, 0])

        with self.assertRaises(ValueError):
            uut.update(truth, predicted)


class CpuUsageMetricTests(unittest.TestCase):
    def test_standalone_cpu_usage(self):
        uut = CpuUsage()

        # Assert result is 0 when created
        self.assertEqual(0.0, uut.result())

        # Base usage
        uut.update()
        for i in range(5):
            a = 0
            for j in range(1000000):
                a += 1
            uut.update()
            self.assertLessEqual(0.0, uut.result())
            time.sleep(0.3)

        # Assert reset actually resets
        uut.reset()
        self.assertEqual(0.0, uut.result())

        # Assert reset doesn't restart the tracking
        a = 0
        for j in range(1000000):
            a += 1
        time.sleep(0.3)
        uut.update()
        self.assertEqual(0.0, uut.result())

        # Assert result doesn't update the metric value
        a = 0
        for j in range(1000000):
            a += 1
        time.sleep(0.3)

        last_result = uut.result()
        self.assertLessEqual(0.0, last_result)

        a = 0
        for j in range(1000000):
            a += 1
        time.sleep(0.3)

        self.assertEqual(last_result, uut.result())

    def test_cpu_usage_helper(self):
        metrics = cpu_usage_metrics(epoch=True, step=True)
        self.assertEqual(2, len(metrics))
        self.assertIsInstance(metrics, List)
        self.assertIsInstance(metrics[0], PluginMetric)
        self.assertIsInstance(metrics[1], PluginMetric)

        with self.assertRaises(ValueError):
            cpu_usage_metrics(train=False, test=False)


class RamUsageMetricTests(unittest.TestCase):
    def test_standalone_cpu_usage(self):
        uut = RamUsage()

        # Assert result is None when created
        self.assertEqual(None, uut.result())

        # Base usage
        uut.update()
        last_result = uut.result()
        self.assertLessEqual(0.0, last_result)

        huge_blob = torch.rand(100000, dtype=torch.float64)

        self.assertLessEqual(last_result, uut.result())
        self.assertLessEqual(-1, huge_blob[0].item())  # Just to prevent removal

        # Assert reset actually resets
        uut.reset()
        self.assertEqual(None, uut.result())


class DiskUsageMetricTests(unittest.TestCase):
    def test_standalone_disk_usage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            uut = DiskUsage(tmp_dir)

            # Assert result is None when created
            self.assertEqual(None, uut.result())

            # Base usage
            uut.update()
            self.assertLessEqual(0, uut.result())

            # Assert reset actually resets
            uut.reset()
            self.assertEqual(None, uut.result())

            uut.update()
            base_value = uut.result()
            with open(str(Path(tmp_dir) / 'blob_file.bin'), "wb") as f:
                f.write(os.urandom(512))

            # Shouldn't change between calls to update
            self.assertEqual(base_value, uut.result())

            uut.update()

            self.assertLessEqual(base_value, uut.result())

    def test_standalone_disk_usage_multi_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dirA:
            with tempfile.TemporaryDirectory() as tmp_dirB:
                uut = DiskUsage([tmp_dirA, tmp_dirB])

                uut.update()
                base_value = uut.result()
                with open(str(Path(tmp_dirA) / 'blob_fileA.bin'), "wb") as f:
                    f.write(os.urandom(512))

                uut.update()
                self.assertLessEqual(base_value, uut.result())

                base_value = uut.result()

                with open(str(Path(tmp_dirB) / 'blob_fileB.bin'), "wb") as f:
                    f.write(os.urandom(512))

                uut.update()
                self.assertLessEqual(base_value, uut.result())


if __name__ == '__main__':
    unittest.main()
