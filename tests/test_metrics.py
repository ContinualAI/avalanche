""" Metrics Tests"""

import unittest

import torch
import numpy as np
import random
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from copy import deepcopy
from os.path import expanduser
from avalanche.evaluation.metrics import Accuracy, Loss, ConfusionMatrix, \
    DiskUsage, MAC, CPUUsage, MaxGPU, MaxRAM, Mean, Sum, ElapsedTime, \
    Forgetting
import pathlib
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop
from avalanche.benchmarks.utils import AvalancheTensorDataset, \
    AvalancheDatasetType
from avalanche.benchmarks import nc_benchmark, dataset_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, \
    accuracy_metrics, loss_metrics, cpu_usage_metrics, timing_metrics, \
    ram_usage_metrics, disk_usage_metrics, MAC_metrics, \
    bwt_metrics, confusion_matrix_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive


#################################
#################################
#    STANDALONE METRIC TEST     #
#################################
#################################

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
        metric.update(torch.tensor(1.), self.batch_size, 0)
        self.assertGreaterEqual(metric.result(0)[0], 0)
        metric.reset()
        self.assertEqual(metric.result(), {})

    def test_loss_multi_task(self):
        metric = Loss()
        self.assertEqual(metric.result(), {})
        metric.update(torch.tensor(1.), 1, 0)
        metric.update(torch.tensor(2.), 1, 1)
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


#################################
#################################
#      PLUGIN METRIC TEST       #
#################################
#################################

DEVICE = 'cpu'
DELTA = 0.01


def filter_dict(d, name):
    out = {}
    for k, v in sorted(d.items()):
        if name in k:
            out[k] = deepcopy(v)
    return out


class PluginMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        train_transform = transforms.Compose([
            RandomCrop(28, padding=4),
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True,
                            transform=train_transform)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True,
                           transform=test_transform)
        scenario = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=False, seed=0)
        model = SimpleMLP(num_classes=scenario.n_classes)

        f = open('log.txt', 'w')
        text_logger = TextLogger(f)
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(
                minibatch=True, epoch=True, epoch_running=True,
                experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, epoch_running=True,
                         experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True, task=True),
            confusion_matrix_metrics(num_classes=10, save_image=False,
                                     normalize='all', stream=True),
            bwt_metrics(experience=True, stream=True, task=True),
            cpu_usage_metrics(
                minibatch=True, epoch=True, epoch_running=True,
                experience=True, stream=True),
            timing_metrics(
                minibatch=True, epoch=True, epoch_running=True,
                experience=True, stream=True),
            ram_usage_metrics(
                every=0.5, minibatch=True, epoch=True,
                experience=True, stream=True),
            disk_usage_metrics(
                minibatch=True, epoch=True, experience=True, stream=True),
            MAC_metrics(
                minibatch=True, epoch=True, experience=True),
            loggers=[text_logger],
            collect_all=True)  # collect all metrics (set to True by default)
        cl_strategy = Naive(
            model, SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(), train_mb_size=500, train_epochs=2,
            eval_mb_size=100, device=DEVICE, evaluator=eval_plugin,
            eval_every=1)
        for i, experience in enumerate(scenario.train_stream):
            cl_strategy.train(experience,
                              eval_streams=[scenario.test_stream[i]])
            cl_strategy.eval(scenario.test_stream)
        self.all_metrics = cl_strategy.evaluator.get_all_metrics()
        f.close()
        # with open('sit.pickle', 'wb') as f:
        #     pickle.dump(dict(self.all_metrics), f,
        #                 protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(pathlib.Path(__file__).parent.absolute(),
                               'sit.pickle'), 'rb') as f:
            self.ref = pickle.load(f)

    def metric_check(self, name):
        d = filter_dict(self.all_metrics, name)
        d_ref = filter_dict(self.ref, name)
        for (k, v), (kref, vref) in zip(d.items(), d_ref.items()):
            self.assertEqual(k, kref)
            init = -1
            for el in v[0]:
                self.assertTrue(el > init)
                init = el
            for el, elref in zip(v[0], vref[0]):
                self.assertEqual(el, elref)
            for el, elref in zip(v[1], vref[1]):
                self.assertAlmostEqual(el, elref, delta=DELTA)

    def test_accuracy(self):
        self.metric_check('Acc')

    def test_loss(self):
        self.metric_check('Loss')

    def test_mac(self):
        self.metric_check('MAC')

    def test_forgetting_bwt(self):
        df = filter_dict(self.all_metrics, 'Forgetting')
        db = filter_dict(self.all_metrics, 'BWT')
        self.metric_check('Forgetting')
        self.metric_check('BWT')
        for (kf, vf), (kb, vb) in zip(df.items(), db.items()):
            self.assertTrue(
                (kf.startswith('Stream') and kb.startswith('Stream')) or
                (kf.startswith('Experience') and kb.startswith('Experience')) or
                (kf.startswith('Task') and kb.startswith('Task'))
            )
            for f, b in zip(vf[1], vb[1]):
                self.assertEqual(f, -b)

    def test_cm(self):
        d = filter_dict(self.all_metrics, 'ConfusionMatrix')
        d_ref = filter_dict(self.ref, 'ConfusionMatrix')
        for (k, v), (kref, vref) in zip(d.items(), d_ref.items()):
            self.assertEqual(k, kref)
            for el, elref in zip(v[0], vref[0]):
                self.assertEqual(el, elref)
            for el, elref in zip(v[1], vref[1]):
                self.assertTrue((el == elref).all())


class PluginMetricMultiTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        train_transform = transforms.Compose([
            RandomCrop(28, padding=4),
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True,
                            transform=train_transform)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True,
                           transform=test_transform)
        scenario = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=True, seed=0)
        model = SimpleMLP(num_classes=scenario.n_classes)

        f = open('log.txt', 'w')
        text_logger = TextLogger(f)
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(
                minibatch=True, epoch=True, epoch_running=True,
                experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, epoch_running=True,
                         experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True, task=True),
            confusion_matrix_metrics(num_classes=10, save_image=False,
                                     normalize='all', stream=True),
            bwt_metrics(experience=True, stream=True, task=True),
            cpu_usage_metrics(
                minibatch=True, epoch=True, epoch_running=True,
                experience=True, stream=True),
            timing_metrics(
                minibatch=True, epoch=True, epoch_running=True,
                experience=True, stream=True),
            ram_usage_metrics(
                every=0.5, minibatch=True, epoch=True,
                experience=True, stream=True),
            disk_usage_metrics(
                minibatch=True, epoch=True, experience=True, stream=True),
            MAC_metrics(
                minibatch=True, epoch=True, experience=True),
            loggers=[text_logger],
            collect_all=True)  # collect all metrics (set to True by default)
        cl_strategy = Naive(
            model, SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(), train_mb_size=500, train_epochs=2,
            eval_mb_size=100, device=DEVICE, evaluator=eval_plugin,
            eval_every=1)
        for i, experience in enumerate(scenario.train_stream):
            cl_strategy.train(experience,
                              eval_streams=[scenario.test_stream[i]])
            cl_strategy.eval(scenario.test_stream)
        self.all_metrics = cl_strategy.evaluator.get_all_metrics()
        f.close()
        # with open('mt.pickle', 'wb') as f:
        #     pickle.dump(dict(self.all_metrics), f,
        #                 protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(pathlib.Path(__file__).parent.absolute(),
                               'sit.pickle'), 'rb') as f:
            self.ref = pickle.load(f)

    def metric_check(self, name):
        d = filter_dict(self.all_metrics, name)
        d_ref = filter_dict(self.ref, name)
        for (k, v), (kref, vref) in zip(d.items(), d_ref.items()):
            self.assertEqual(k, kref)
            init = -1
            for el in v[0]:
                self.assertTrue(el > init)
                init = el
            for el, elref in zip(v[0], vref[0]):
                self.assertEqual(el, elref)
            for el, elref in zip(v[1], vref[1]):
                self.assertAlmostEqual(el, elref, delta=DELTA)

    def test_accuracy(self):
        self.metric_check('Acc')

    def test_loss(self):
        self.metric_check('Loss')

    def test_mac(self):
        self.metric_check('MAC')

    def test_forgetting_bwt(self):
        df = filter_dict(self.all_metrics, 'Forgetting')
        db = filter_dict(self.all_metrics, 'BWT')
        self.metric_check('Forgetting')
        self.metric_check('BWT')
        for (kf, vf), (kb, vb) in zip(df.items(), db.items()):
            self.assertTrue(
                (kf.startswith('Stream') and kb.startswith('Stream')) or
                (kf.startswith('Experience') and kb.startswith('Experience')) or
                (kf.startswith('Task') and kb.startswith('Task')),
            )
            for f, b in zip(vf[1], vb[1]):
                self.assertEqual(f, -b)

    def test_cm(self):
        d = filter_dict(self.all_metrics, 'ConfusionMatrix')
        d_ref = filter_dict(self.ref, 'ConfusionMatrix')
        for (k, v), (kref, vref) in zip(d.items(), d_ref.items()):
            self.assertEqual(k, kref)
            for el, elref in zip(v[0], vref[0]):
                self.assertEqual(el, elref)
            for el, elref in zip(v[1], vref[1]):
                self.assertTrue((el == elref).all())


class PluginMetricTaskLabelPerPatternTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        n_samples_per_class = 100
        datasets = []
        for i in range(3):
            dataset = make_classification(
                n_samples=10 * n_samples_per_class,
                n_classes=3,
                n_features=3, n_informative=3, n_redundant=0)
            X = torch.from_numpy(dataset[0]).float()
            y = torch.from_numpy(dataset[1]).long()
            train_X, test_X, train_y, test_y = train_test_split(
                X, y, train_size=0.5, shuffle=True, stratify=y)
            datasets.append((train_X, train_y, test_X, test_y))

        tr_ds = [AvalancheTensorDataset(
            tr_X, tr_y,
            dataset_type=AvalancheDatasetType.CLASSIFICATION,
            task_labels=torch.randint(0, 3, (500,)).tolist())
            for tr_X, tr_y, _, _ in datasets]
        ts_ds = [AvalancheTensorDataset(
            ts_X, ts_y,
            dataset_type=AvalancheDatasetType.CLASSIFICATION,
            task_labels=torch.randint(0, 3, (500,)).tolist())
            for _, _, ts_X, ts_y in datasets]
        scenario = dataset_benchmark(train_datasets=tr_ds, test_datasets=ts_ds)
        model = SimpleMLP(num_classes=3, input_size=3)

        f = open('log.txt', 'w')
        text_logger = TextLogger(f)
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(
                minibatch=True, epoch=True, epoch_running=True,
                experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, epoch_running=True,
                         experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True, task=True),
            confusion_matrix_metrics(num_classes=10, save_image=False,
                                     normalize='all', stream=True),
            bwt_metrics(experience=True, stream=True, task=True),
            cpu_usage_metrics(
                minibatch=True, epoch=True, epoch_running=True,
                experience=True, stream=True),
            timing_metrics(
                minibatch=True, epoch=True, epoch_running=True,
                experience=True, stream=True),
            ram_usage_metrics(
                every=0.5, minibatch=True, epoch=True,
                experience=True, stream=True),
            disk_usage_metrics(
                minibatch=True, epoch=True, experience=True, stream=True),
            MAC_metrics(
                minibatch=True, epoch=True, experience=True),
            loggers=[text_logger],
            collect_all=True)  # collect all metrics (set to True by default)
        cl_strategy = Naive(
            model, SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(), train_mb_size=2, train_epochs=2,
            eval_mb_size=2, device=DEVICE,
            evaluator=eval_plugin, eval_every=1)
        for i, experience in enumerate(scenario.train_stream):
            cl_strategy.train(experience,
                              eval_streams=[scenario.test_stream[i]])
            cl_strategy.eval(scenario.test_stream)
        self.all_metrics = cl_strategy.evaluator.get_all_metrics()
        f.close()
        # with open('tpp.pickle', 'wb') as f:
        #     pickle.dump(dict(self.all_metrics), f,
        #                 protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(pathlib.Path(__file__).parent.absolute(),
                               'sit.pickle'), 'rb') as f:
            self.ref = pickle.load(f)

    def metric_check(self, name):
        d = filter_dict(self.all_metrics, name)
        d_ref = filter_dict(self.ref, name)
        for (k, v), (kref, vref) in zip(d.items(), d_ref.items()):
            self.assertEqual(k, kref)
            init = -1
            for el in v[0]:
                self.assertTrue(el > init)
                init = el
            for el, elref in zip(v[0], vref[0]):
                self.assertEqual(el, elref)
            for el, elref in zip(v[1], vref[1]):
                self.assertAlmostEqual(el, elref, delta=DELTA)

    def test_accuracy(self):
        self.metric_check('Acc')

    def test_loss(self):
        self.metric_check('Loss')

    def test_mac(self):
        self.metric_check('MAC')

    def test_forgetting_bwt(self):
        df = filter_dict(self.all_metrics, 'Forgetting')
        db = filter_dict(self.all_metrics, 'BWT')
        self.metric_check('Forgetting')
        self.metric_check('BWT')
        for (kf, vf), (kb, vb) in zip(df.items(), db.items()):
            self.assertTrue(
                (kf.startswith('Stream') and kb.startswith('Stream')) or
                (kf.startswith('Experience') and kb.startswith('Experience') or
                 (kf.startswith('Task') and kb.startswith('Task'))))
            for f, b in zip(vf[1], vb[1]):
                self.assertEqual(f, -b)

    def test_cm(self):
        d = filter_dict(self.all_metrics, 'ConfusionMatrix')
        d_ref = filter_dict(self.ref, 'ConfusionMatrix')
        for (k, v), (kref, vref) in zip(d.items(), d_ref.items()):
            self.assertEqual(k, kref)
            for el, elref in zip(v[0], vref[0]):
                self.assertEqual(el, elref)
            for el, elref in zip(v[1], vref[1]):
                self.assertTrue((el == elref).all())


if __name__ == '__main__':
    unittest.main()
