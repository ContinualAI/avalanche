################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-06-2020                                                              #
# Author(s): Andrea Cossu                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import unittest

import torch
from torchvision.transforms import ToTensor, Compose, transforms, Resize
import os
import sys

from os.path import expanduser

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

from avalanche.benchmarks.datasets import MNIST
from avalanche.logging import TextLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive, Replay, CWRStar, \
    GDumb, LwF, AGEM, GEM, EWC, \
    SynapticIntelligence, JointTraining, SIW
from avalanche.training.strategies.ar1 import AR1
from avalanche.training.strategies.cumulative import Cumulative
from avalanche.benchmarks import nc_scenario, SplitCIFAR10
from avalanche.training.utils import get_last_fc_layer
from avalanche.evaluation.metrics import StreamAccuracy


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
    my_nc_scenario = nc_scenario(train_dataset, test_dataset, 5,
                                 task_labels=False)
    return my_nc_scenario


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

    def test_periodic_eval(self):
        model = SimpleMLP(input_size=6, hidden_size=10)
        scenario = get_fast_scenario()
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        curve_key = 'Top1_Acc_Stream/eval_phase/train_stream'

        ###################
        # Case #1: No eval
        ###################
        # we use stream acc. because it emits a single value
        # for each eval loop.
        acc = StreamAccuracy()
        strategy = Naive(model, optimizer, criterion, train_epochs=2,
                         eval_every=-1, evaluator=EvaluationPlugin(acc))
        strategy.train(scenario.train_stream[0])
        # eval is not called in this case
        assert len(strategy.evaluator.get_all_metrics()) == 0

        ###################
        # Case #2: Eval at the end only
        ###################
        acc = StreamAccuracy()
        strategy = Naive(model, optimizer, criterion, train_epochs=2,
                         eval_every=0, evaluator=EvaluationPlugin(acc))
        strategy.train(scenario.train_stream[0])
        # eval is called once at the end of the training loop
        curve = strategy.evaluator.get_all_metrics()[curve_key][1]
        assert len(curve) == 1

        ###################
        # Case #3: Eval after every epoch
        ###################
        acc = StreamAccuracy()
        strategy = Naive(model, optimizer, criterion, train_epochs=2,
                         eval_every=1, evaluator=EvaluationPlugin(acc))
        strategy.train(scenario.train_stream[0])
        # eval is called after every epoch + the end of the training loop
        curve = strategy.evaluator.get_all_metrics()[curve_key][1]
        assert len(curve) == 3


class StrategyTest(unittest.TestCase):
    if "FAST_TEST" in os.environ:
        fast_test = os.environ['FAST_TEST'].lower() in ["true"]
    else:
        fast_test = False
    if "USE_GPU" in os.environ:
        use_gpu = os.environ['USE_GPU'].lower() in ["true"]
    else:
        use_gpu = False

    print("Fast Test:", fast_test)
    print("Test on GPU:", use_gpu)

    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    def test_naive(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)
        strategy = Naive(model, optimizer, criterion, train_mb_size=64,
                         device=self.device, eval_mb_size=50, train_epochs=2)
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = Naive(model, optimizer, criterion, train_mb_size=64,
                         device=self.device, eval_mb_size=50, train_epochs=2)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_joint(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)
        strategy = JointTraining(model, optimizer, criterion, train_mb_size=64,
                                 device=self.device, eval_mb_size=50,
                                 train_epochs=2)
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = Naive(model, optimizer, criterion, train_mb_size=64,
                         device=self.device, eval_mb_size=50, train_epochs=2)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_cwrstar(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)
        last_fc_name, _ = get_last_fc_layer(model)
        strategy = CWRStar(model, optimizer, criterion, last_fc_name,
                           train_mb_size=64, device=self.device)
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = CWRStar(model, optimizer, criterion, last_fc_name,
                           train_mb_size=64, device=self.device)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_replay(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)
        strategy = Replay(model, optimizer, criterion,
                          mem_size=10, train_mb_size=64, device=self.device,
                          eval_mb_size=50, train_epochs=2)
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = Replay(model, optimizer, criterion,
                          mem_size=10, train_mb_size=64, device=self.device,
                          eval_mb_size=50, train_epochs=2)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_gdumb(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)
        strategy = GDumb(
            model, optimizer, criterion,
            mem_size=200, train_mb_size=64, device=self.device,
            eval_mb_size=50, train_epochs=2
        )
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = GDumb(
            model, optimizer, criterion,
            mem_size=200, train_mb_size=64, device=self.device,
            eval_mb_size=50, train_epochs=2
        )
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_cumulative(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)
        strategy = Cumulative(model, optimizer, criterion, train_mb_size=64,
                              device=self.device, eval_mb_size=50,
                              train_epochs=2)
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = Cumulative(model, optimizer, criterion, train_mb_size=64,
                              device=self.device, eval_mb_size=50,
                              train_epochs=2)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_lwf(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)
        strategy = LwF(model, optimizer, criterion,
                       alpha=[0, 1 / 2, 2 * (2 / 3), 3 * (3 / 4), 4 * (4 / 5)],
                       temperature=2, device=self.device,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = LwF(model, optimizer, criterion,
                       alpha=[0, 1 / 2, 2 * (2 / 3), 3 * (3 / 4), 4 * (4 / 5)],
                       temperature=2, device=self.device,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_agem(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)
        strategy = AGEM(model, optimizer, criterion,
                        patterns_per_exp=250, sample_size=256,
                        train_mb_size=10, eval_mb_size=50,
                        train_epochs=2)
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = AGEM(model, optimizer, criterion,
                        patterns_per_exp=250, sample_size=256,
                        train_mb_size=10, eval_mb_size=50,
                        train_epochs=2)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_gem(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-1)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)
        strategy = GEM(model, optimizer, criterion,
                       patterns_per_exp=256,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)

        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = GEM(model, optimizer, criterion,
                       patterns_per_exp=256,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        self.run_strategy(my_nc_scenario, strategy)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_ewc(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)

        # SIT scenario
        strategy = EWC(model, optimizer, criterion, ewc_lambda=0.4,
                       mode='separate',
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)

        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = EWC(model, optimizer, criterion, ewc_lambda=0.4,
                       mode='separate',
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_ewc_online(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test,
                                            use_task_labels=False)
        strategy = EWC(model, optimizer, criterion, ewc_lambda=0.4,
                       mode='online', decay_factor=0.1,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = EWC(model, optimizer, criterion, ewc_lambda=0.4,
                       mode='online', decay_factor=0.1,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_synaptic_intelligence(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        # SIT scenario
        strategy = SynapticIntelligence(
            model, optimizer, criterion, si_lambda=0.0001,
            train_epochs=1, train_mb_size=10, eval_mb_size=10)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=False)
        self.run_strategy(scenario, strategy)

        # MT scenario
        strategy = SynapticIntelligence(
            model, optimizer, criterion, si_lambda=0.0001,
            train_epochs=1, train_mb_size=10, eval_mb_size=10)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_ar1(self):
        my_nc_scenario = self.load_ar1_scenario(fast_test=self.fast_test)

        strategy = AR1(train_epochs=1, train_mb_size=10, eval_mb_size=10,
                       rm_sz=200)

        self.run_strategy(my_nc_scenario, strategy)

    def test_siw(self):
        model = self.get_model(fast_test=self.fast_test)
        optimizer = SGD(model.parameters(), lr=0.1)
        criterion = CrossEntropyLoss()

        # SIT scenario
        my_nc_scenario = self.load_scenario(fast_test=self.fast_test)
        strategy = SIW(model, optimizer, criterion, siw_layer_name='classifier',
                       batch_size=32, num_workers=8, train_mb_size=128,
                       device=self.device, eval_mb_size=32, train_epochs=2)
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = SIW(model, optimizer, criterion, siw_layer_name='classifier',
                       batch_size=32, num_workers=8, train_mb_size=128,
                       device=self.device, eval_mb_size=32, train_epochs=2)
        scenario = self.load_scenario(fast_test=self.fast_test,
                                      use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def load_ar1_scenario(self, fast_test=False):
        """
        Returns a NC Scenario from a fake dataset of 10 classes, 5 experiences,
        2 classes per experience. This toy scenario is intended

        :param fast_test: if True loads fake data, MNIST otherwise.
        """

        if fast_test:
            n_samples_per_class = 50

            dataset = make_classification(
                n_samples=10 * n_samples_per_class,
                n_classes=10,
                n_features=224 * 224 * 3, n_informative=6, n_redundant=0)

            X = torch.from_numpy(dataset[0]).reshape(-1, 3, 224, 224).float()
            y = torch.from_numpy(dataset[1]).long()

            train_X, test_X, train_y, test_y = train_test_split(
                X, y, train_size=0.6, shuffle=True, stratify=y)

            train_dataset = TensorDataset(train_X, train_y)
            test_dataset = TensorDataset(test_X, test_y)
            my_nc_scenario = nc_scenario(
                train_dataset, test_dataset, 5, task_labels=False
            )
        else:
            train_transform = transforms.Compose([
                Resize(224),
                ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_transform = transforms.Compose([
                Resize(224),
                ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            my_nc_scenario = SplitCIFAR10(5, train_transform=train_transform,
                                          eval_transform=test_transform)

        return my_nc_scenario

    def load_scenario(self, fast_test=False, use_task_labels=False):
        """
        Returns a NC Scenario from a fake dataset of 10 classes, 5 experiences,
        2 classes per experience.

        :param fast_test: if True loads fake data, MNIST otherwise.
        """

        if fast_test:
            my_nc_scenario = get_fast_scenario()
        else:
            mnist_train = MNIST(
                root=expanduser("~") + "/.avalanche/data/mnist/",
                train=True, download=True,
                transform=Compose([ToTensor()]))

            mnist_test = MNIST(
                root=expanduser("~") + "/.avalanche/data/mnist/",
                train=False, download=True,
                transform=Compose([ToTensor()]))
            my_nc_scenario = nc_scenario(
                mnist_train, mnist_test, 5,
                task_labels=use_task_labels, seed=1234)

        return my_nc_scenario

    def get_model(self, fast_test=False):
        if fast_test:
            return SimpleMLP(input_size=6, hidden_size=10)
        else:
            return SimpleMLP()

    def run_strategy(self, scenario, cl_strategy):
        print('Starting experiment...')
        cl_strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        results = []
        for train_batch_info in scenario.train_stream:
            print("Start of experience ", train_batch_info.current_experience)

            cl_strategy.train(train_batch_info)
            print('Training completed')

            print('Computing accuracy on the current test set')
            results.append(cl_strategy.eval(scenario.test_stream[:]))


if __name__ == '__main__':
    unittest.main()
