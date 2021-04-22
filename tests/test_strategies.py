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
from torchvision.transforms import ToTensor, transforms, Resize
import os
import sys

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

from avalanche.logging import TextLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive, Replay, CWRStar, \
    GDumb, LwF, AGEM, GEM, EWC, \
    SynapticIntelligence, JointTraining, SIW
from avalanche.training.strategies.ar1 import AR1
from avalanche.training.strategies.cumulative import Cumulative
from avalanche.benchmarks import nc_benchmark, SplitCIFAR10
from avalanche.training.utils import get_last_fc_layer
from avalanche.evaluation.metrics import StreamAccuracy

from tests.unit_tests_utils import common_setups, get_fast_scenario


class BaseStrategyTest(unittest.TestCase):
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

    def init_sit(self):
        model = self.get_model(fast_test=True)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        scenario = self.load_scenario(use_task_labels=False)
        return model, optimizer, criterion, scenario

    def test_naive(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = Naive(model, optimizer, criterion, train_mb_size=64,
                         device=self.device, eval_mb_size=50, train_epochs=2)
        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = Naive(model, optimizer, criterion, train_mb_size=64,
                         device=self.device, eval_mb_size=50, train_epochs=2)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_joint(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = JointTraining(model, optimizer, criterion, train_mb_size=64,
                                 device=self.device, eval_mb_size=50,
                                 train_epochs=2)
        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = Naive(model, optimizer, criterion, train_mb_size=64,
                         device=self.device, eval_mb_size=50, train_epochs=2)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_cwrstar(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        last_fc_name, _ = get_last_fc_layer(model)
        strategy = CWRStar(model, optimizer, criterion, last_fc_name,
                           train_mb_size=64, device=self.device)
        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = CWRStar(model, optimizer, criterion, last_fc_name,
                           train_mb_size=64, device=self.device)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_replay(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = Replay(model, optimizer, criterion,
                          mem_size=10, train_mb_size=64, device=self.device,
                          eval_mb_size=50, train_epochs=2)
        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = Replay(model, optimizer, criterion,
                          mem_size=10, train_mb_size=64, device=self.device,
                          eval_mb_size=50, train_epochs=2)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_gdumb(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = GDumb(
            model, optimizer, criterion,
            mem_size=200, train_mb_size=64, device=self.device,
            eval_mb_size=50, train_epochs=2
        )
        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = GDumb(
            model, optimizer, criterion,
            mem_size=200, train_mb_size=64, device=self.device,
            eval_mb_size=50, train_epochs=2
        )
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_cumulative(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = Cumulative(model, optimizer, criterion, train_mb_size=64,
                              device=self.device, eval_mb_size=50,
                              train_epochs=2)
        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = Cumulative(model, optimizer, criterion, train_mb_size=64,
                              device=self.device, eval_mb_size=50,
                              train_epochs=2)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_lwf(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = LwF(model, optimizer, criterion,
                       alpha=[0, 1 / 2, 2 * (2 / 3), 3 * (3 / 4), 4 * (4 / 5)],
                       temperature=2, device=self.device,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = LwF(model, optimizer, criterion,
                       alpha=[0, 1 / 2, 2 * (2 / 3), 3 * (3 / 4), 4 * (4 / 5)],
                       temperature=2, device=self.device,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_agem(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = AGEM(model, optimizer, criterion,
                        patterns_per_exp=250, sample_size=256,
                        train_mb_size=10, eval_mb_size=50,
                        train_epochs=2)
        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = AGEM(model, optimizer, criterion,
                        patterns_per_exp=250, sample_size=256,
                        train_mb_size=10, eval_mb_size=50,
                        train_epochs=2)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_gem(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = GEM(model, optimizer, criterion,
                       patterns_per_exp=256,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)

        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = GEM(model, optimizer, criterion,
                       patterns_per_exp=256,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        self.run_strategy(my_nc_benchmark, strategy)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_ewc(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = EWC(model, optimizer, criterion, ewc_lambda=0.4,
                       mode='separate',
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)

        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = EWC(model, optimizer, criterion, ewc_lambda=0.4,
                       mode='separate',
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_ewc_online(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = EWC(model, optimizer, criterion, ewc_lambda=0.4,
                       mode='online', decay_factor=0.1,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        self.run_strategy(my_nc_benchmark, strategy)

        # MT scenario
        strategy = EWC(model, optimizer, criterion, ewc_lambda=0.4,
                       mode='online', decay_factor=0.1,
                       train_mb_size=10, eval_mb_size=50,
                       train_epochs=2)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_synaptic_intelligence(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_benchmark = self.init_sit()
        strategy = SynapticIntelligence(
            model, optimizer, criterion, si_lambda=0.0001,
            train_epochs=1, train_mb_size=10, eval_mb_size=10)
        scenario = self.load_scenario(use_task_labels=False)
        self.run_strategy(scenario, strategy)

        # MT scenario
        strategy = SynapticIntelligence(
            model, optimizer, criterion, si_lambda=0.0001,
            train_epochs=1, train_mb_size=10, eval_mb_size=10)
        scenario = self.load_scenario(use_task_labels=True)
        self.run_strategy(scenario, strategy)

    def test_ar1(self):
        my_nc_benchmark = self.load_ar1_scenario()
        strategy = AR1(train_epochs=1, train_mb_size=10, eval_mb_size=10,
                       rm_sz=200)
        self.run_strategy(my_nc_benchmark, strategy)

    def test_siw(self):
        # SIT scenario
        model, optimizer, criterion, my_nc_scenario = self.init_sit()
        strategy = SIW(model, optimizer, criterion, siw_layer_name='classifier',
                       batch_size=32, num_workers=8, train_mb_size=128,
                       device=self.device, eval_mb_size=32, train_epochs=2)
        self.run_strategy(my_nc_scenario, strategy)

        # MT scenario
        strategy = SIW(model, optimizer, criterion, siw_layer_name='classifier',
                       batch_size=32, num_workers=8, train_mb_size=128,
                       device=self.device, eval_mb_size=32, train_epochs=2)
        scenario = self.load_scenario(use_task_labels=False)
        self.run_strategy(scenario, strategy)

    def load_ar1_scenario(self):
        """
        Returns a NC Scenario from a fake dataset of 10 classes, 5 experiences,
        2 classes per experience. This toy scenario is intended
        """
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
        my_nc_benchmark = nc_benchmark(
            train_dataset, test_dataset, 5, task_labels=False
        )
        return my_nc_benchmark

    def load_scenario(self, use_task_labels=False):
        """
        Returns a NC Scenario from a fake dataset of 10 classes, 5 experiences,
        2 classes per experience.

        :param fast_test: if True loads fake data, MNIST otherwise.
        """
        return get_fast_scenario(use_task_labels=use_task_labels)

    def get_model(self, fast_test=False):
        if fast_test:
            return SimpleMLP(input_size=6, hidden_size=10)
        else:
            return SimpleMLP()

    def run_strategy(self, scenario, cl_strategy):
        print('Starting experiment...')
        cl_strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        results = []
        for i, train_batch_info in enumerate(scenario.train_stream):
            print("Start of experience ", train_batch_info.current_experience)

            cl_strategy.train(train_batch_info)
            print('Training completed')

            print('Computing accuracy on the current test set')
            results.append(cl_strategy.eval(scenario.test_stream[:i+1]))


if __name__ == '__main__':
    unittest.main()
