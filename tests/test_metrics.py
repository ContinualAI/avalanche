""" Metrics Tests"""

import unittest

import torch
import numpy as np
import random
from copy import deepcopy
from os.path import expanduser
from avalanche.evaluation.metrics import Accuracy, Loss, ConfusionMatrix, \
    DiskUsage, MAC, CPUUsage, MaxGPU, MaxRAM, Mean, Sum, ElapsedTime, Forgetting
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop
from avalanche.benchmarks import nc_benchmark
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


class PluginMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        device = 'cpu'
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
                            train=True, download=True, transform=train_transform)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True, transform=test_transform)
        scenario = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=False, seed=0)
        model = SimpleMLP(num_classes=scenario.n_classes)

        f = open('log.txt', 'w')
        text_logger = TextLogger(f)
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(
                minibatch=True, epoch=True, epoch_running=True, experience=True,
                stream=True),
            loss_metrics(minibatch=True, epoch=True, epoch_running=True,
                         experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            confusion_matrix_metrics(num_classes=10, save_image=False, normalize='all', stream=True),
            bwt_metrics(experience=True, stream=True),
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
            CrossEntropyLoss(), train_mb_size=500, train_epochs=2, eval_mb_size=100,
            device=device, evaluator=eval_plugin, eval_every=1)
        for i, experience in enumerate(scenario.train_stream):
            cl_strategy.train(experience,
                              eval_streams=[scenario.test_stream[i]])
            cl_strategy.eval(scenario.test_stream)
        self.all_metrics = cl_strategy.evaluator.get_all_metrics()
        f.close()
        self.delta = 0.01

    def filter_dict(self, name):
        out = {}
        for k, v in self.all_metrics.items():
            if name in k:
                out[k] = deepcopy(v)
        return out

    def test_accuracy(self):
        d = self.filter_dict('Acc')
        for k, v in d.items():
            if k == 'Top1_Acc_MB/train_phase/train_stream/Task000':
                self.assertAlmostEqual(v[1][-1], 0.6941, delta=self.delta)
            elif k == 'Top1_RunningAcc_Epoch/train_phase/train_stream/Task000':
                self.assertAlmostEqual(v[1][-1], 0.5752, delta=self.delta)
            elif k == 'Top1_Acc_Epoch/train_phase/train_stream/Task000':
                self.assertAlmostEqual(v[1][-1], 0.5752, delta=self.delta)
            elif k == 'Top1_Acc_Stream/eval_phase/test_stream/Task000':
                self.assertAlmostEqual(v[1][-1], 0.1689, delta=self.delta)
            elif k == 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000':
                self.assertAlmostEqual(v[1][-1], 0, delta=self.delta)
            elif k == 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001':
                self.assertAlmostEqual(v[1][-1], 0, delta=self.delta)
            elif k == 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002':
                self.assertAlmostEqual(v[1][-1], 0, delta=self.delta)
            elif k == 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp003':
                self.assertAlmostEqual(v[1][-1], 0, delta=self.delta)
            elif k == 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp004':
                self.assertAlmostEqual(v[1][-1], 0.8487, delta=self.delta)
            else:
                raise KeyError("Key in dictionary not recognized: {}".format(k))

    def test_loss(self):
        d = self.filter_dict('Loss')
        for k, v in d.items():
            if k == 'Loss_MB/train_phase/train_stream/Task000':
                self.assertAlmostEqual(v[1][-1], 0.9629, delta=self.delta)
            elif k == 'RunningLoss_Epoch/train_phase/train_stream/Task000':
                self.assertAlmostEqual(v[1][-1], 1.4509, delta=self.delta)
            elif k == 'Loss_Epoch/train_phase/train_stream/Task000':
                self.assertAlmostEqual(v[1][-1], 1.4509, delta=self.delta)
            elif k == 'Loss_Stream/eval_phase/test_stream/Task000':
                self.assertAlmostEqual(v[1][-1], 2.8269, delta=self.delta)
            elif k == 'Loss_Exp/eval_phase/test_stream/Task000/Exp000':
                self.assertAlmostEqual(v[1][-1], 3.4117, delta=self.delta)
            elif k == 'Loss_Exp/eval_phase/test_stream/Task000/Exp001':
                self.assertAlmostEqual(v[1][-1], 3.6090, delta=self.delta)
            elif k == 'Loss_Exp/eval_phase/test_stream/Task000/Exp002':
                self.assertAlmostEqual(v[1][-1], 3.4580, delta=self.delta)
            elif k == 'Loss_Exp/eval_phase/test_stream/Task000/Exp003':
                self.assertAlmostEqual(v[1][-1], 2.8601, delta=self.delta)
            elif k == 'Loss_Exp/eval_phase/test_stream/Task000/Exp004':
                self.assertAlmostEqual(v[1][-1], 0.7772, delta=self.delta)
            else:
                raise KeyError("Key in dictionary not recognized: {}".format(k))

    def test_mac(self):
        d = self.filter_dict('MAC')
        for k, v in d.items():
            if k == 'MAC_MB/train_phase/train_stream/Task000':
                self.assertEqual(v[1][-1], 406528)
            elif k == 'MAC_Epoch/train_phase/train_stream/Task000':
                self.assertEqual(v[1][-1], 406528)
            elif k == 'MAC_Exp/eval_phase/test_stream/Task000/Exp000':
                self.assertEqual(v[1][-1], 406528)
            elif k == 'MAC_Exp/eval_phase/test_stream/Task000/Exp001':
                self.assertEqual(v[1][-1], 406528)
            elif k == 'MAC_Exp/eval_phase/test_stream/Task000/Exp002':
                self.assertEqual(v[1][-1], 406528)
            elif k == 'MAC_Exp/eval_phase/test_stream/Task000/Exp003':
                self.assertEqual(v[1][-1], 406528)
            elif k == 'MAC_Exp/eval_phase/test_stream/Task000/Exp004':
                self.assertEqual(v[1][-1], 406528)
            else:
                raise KeyError("Key in dictionary not recognized: {}".format(k))

    def test_forgetting_bwt(self):
        df = self.filter_dict('Forgetting')
        db = self.filter_dict('BWT')
        for (kf, vf), (kb, vb) in zip(df.items(), db.items()):
            self.assertTrue(
                (kf.startswith('Stream') and kb.startswith('Stream')) or
                (kf.startswith('Experience') and kb.startswith('Experience'))
            )
            for f, b in zip(vf[1], vb[1]):
                self.assertEqual(f, -b)

            if kf == 'StreamForgetting/eval_phase/test_stream' and \
                    kb == 'StreamBWT/eval_phase/test_stream':
                self.assertAlmostEqual(vf[1][-1], 0.9376, delta=self.delta)
            elif kf == 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp000' and \
                    kb == 'ExperienceBWT/eval_phase/test_stream/Task000/Exp000':
                self.assertAlmostEqual(vf[1][-1], 0.9541, delta=self.delta)
            elif kf == 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp001' and \
                    kb == 'ExperienceBWT/eval_phase/test_stream/Task000/Exp001':
                self.assertAlmostEqual(vf[1][-1], 0.9145, delta=self.delta)
            elif kf == 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp002' and \
                    kb == 'ExperienceBWT/eval_phase/test_stream/Task000/Exp002':
                self.assertAlmostEqual(vf[1][-1], 0.9390, delta=self.delta)
            elif kf == 'ExperienceForgetting/eval_phase/test_stream/Task000/Exp003' and \
                    kb == 'ExperienceBWT/eval_phase/test_stream/Task000/Exp003':
                self.assertAlmostEqual(vf[1][-1], 0.9426, delta=self.delta)
            else:
                raise KeyError("Keys in dictionary not recognized: {} and {}".format(kf, kb))

    def test_cm(self):
        d = self.filter_dict('ConfusionMatrix')
        for k, v in d.items():
            if k == 'ConfusionMatrix_Stream/eval_phase/test_stream':
                self.assertTrue(v[1][-1].size() == torch.Size([10, 10]))
                target_cm = torch.tensor([
                    [0.0000, 0.0000, 0.0923, 0.0000, 0.0000, 0.0000, 0.0057,
                     0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0971, 0.0000, 0.0000, 0.0000, 0.0164,
                     0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0994, 0.0000, 0.0000, 0.0000, 0.0038,
                     0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0901, 0.0000, 0.0000, 0.0000, 0.0109,
                     0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0143, 0.0000, 0.0000, 0.0000, 0.0839,
                     0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0605, 0.0000, 0.0000, 0.0000, 0.0287,
                     0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0263, 0.0000, 0.0000, 0.0000, 0.0695,
                     0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0806, 0.0000, 0.0000, 0.0000, 0.0222,
                     0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0761, 0.0000, 0.0000, 0.0000, 0.0213,
                     0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0434, 0.0000, 0.0000, 0.0000, 0.0575,
                     0.0000, 0.0000, 0.0000]],
                    dtype=torch.float64)
                self.assertTrue((v[1][-1] == target_cm).all())
            else:
                raise KeyError("Key in dictionary not recognized: {}".format(k))


if __name__ == '__main__':
    unittest.main()
