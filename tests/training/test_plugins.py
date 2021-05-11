import sys

import torch
import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset

from avalanche.benchmarks import nc_benchmark
from avalanche.logging import TextLogger
from avalanche.models import SimpleMLP
from avalanche.training import EvaluationPlugin
from avalanche.training.plugins import StrategyPlugin, ReplayPlugin, \
    ExperienceBalancedStoragePolicy, ClassBalancedStoragePolicy
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.strategies import Naive


class MockPlugin(StrategyPlugin):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.activated = [False for _ in range(22)]

    def before_training_exp(self, strategy, **kwargs):
        self.activated[0] = True

    def after_train_dataset_adaptation(self, strategy, **kwargs):
        self.activated[1] = True

    def before_training_epoch(self, strategy, **kwargs):
        self.activated[2] = True

    def before_training_iteration(self, strategy, **kwargs):
        self.activated[3] = True

    def before_forward(self, strategy, **kwargs):
        self.activated[4] = True

    def after_forward(self, strategy, **kwargs):
        self.activated[5] = True

    def before_backward(self, strategy, **kwargs):
        self.activated[6] = True

    def after_backward(self, strategy, **kwargs):
        self.activated[7] = True

    def after_training_iteration(self, strategy, **kwargs):
        self.activated[8] = True

    def before_update(self, strategy, **kwargs):
        self.activated[9] = True

    def after_update(self, strategy, **kwargs):
        self.activated[10] = True

    def after_training_epoch(self, strategy, **kwargs):
        self.activated[11] = True

    def after_training_exp(self, strategy, **kwargs):
        self.activated[12] = True

    def before_eval(self, strategy, **kwargs):
        self.activated[13] = True

    def after_eval_dataset_adaptation(self, strategy, **kwargs):
        self.activated[14] = True

    def before_eval_exp(self, strategy, **kwargs):
        self.activated[15] = True

    def after_eval_exp(self, strategy, **kwargs):
        self.activated[16] = True

    def after_eval(self, strategy, **kwargs):
        self.activated[17] = True

    def before_eval_iteration(self, strategy, **kwargs):
        self.activated[18] = True

    def before_eval_forward(self, strategy, **kwargs):
        self.activated[19] = True

    def after_eval_forward(self, strategy, **kwargs):
        self.activated[20] = True

    def after_eval_iteration(self, strategy, **kwargs):
        self.activated[21] = True


class PluginTests(unittest.TestCase):
    def test_callback_reachability(self):
        # Check that all the callbacks are called during
        # training and test loops.
        model = SimpleMLP(input_size=6, hidden_size=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        scenario = self.create_scenario()

        plug = MockPlugin()
        strategy = Naive(model, optimizer, criterion,
                         train_mb_size=100, train_epochs=1, eval_mb_size=100,
                         device='cpu', plugins=[plug]
                         )
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        strategy.train(scenario.train_stream[0], num_workers=4)
        strategy.eval([scenario.test_stream[0]], num_workers=4)
        assert all(plug.activated)

    def create_scenario(self, task_labels=False):
        n_samples_per_class = 20

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
        return nc_benchmark(train_dataset, test_dataset, 5,
                            task_labels=task_labels,
                            fixed_class_order=list(range(10)))

    def test_scheduler_plugin(self):
        self._test_scheduler_plugin(gamma=1 / 2.,
                                    milestones=[2, 3],
                                    base_lr=4.,
                                    epochs=3,
                                    reset_lr=True,
                                    reset_scheduler=True,
                                    expected=[[4., 2., 1.],
                                              [4., 2., 1.]],
                                    )

        self._test_scheduler_plugin(gamma=1 / 2.,
                                    milestones=[2, 3],
                                    base_lr=4.,
                                    epochs=3,
                                    reset_lr=False,
                                    reset_scheduler=True,
                                    expected=[[4., 2., 1.],
                                              [1., .5, .25]],
                                    )

        self._test_scheduler_plugin(gamma=1 / 2.,
                                    milestones=[2, 3],
                                    base_lr=4.,
                                    epochs=3,
                                    reset_lr=True,
                                    reset_scheduler=False,
                                    expected=[[4., 2., 1.],
                                              [4., 4., 4.]],
                                    )

        self._test_scheduler_plugin(gamma=1 / 2.,
                                    milestones=[2, 3],
                                    base_lr=4.,
                                    epochs=3,
                                    reset_lr=False,
                                    reset_scheduler=False,
                                    expected=[[4., 2., 1.],
                                              [1., 1., 1.]],
                                    )

    def _test_scheduler_plugin(self, gamma, milestones, base_lr, epochs,
                               reset_lr, reset_scheduler, expected):

        class TestPlugin(StrategyPlugin):
            def __init__(self, expected_lrs):
                super().__init__()
                self.expected_lrs = expected_lrs

            def after_training_epoch(self, strategy, **kwargs):
                exp_id = strategy.training_exp_counter

                expected_lr = self.expected_lrs[exp_id][strategy.epoch]
                for group in strategy.optimizer.param_groups:
                    assert group['lr'] == expected_lr

        scenario = self.create_scenario()
        model = SimpleMLP(input_size=6, hidden_size=10)

        optim = SGD(model.parameters(), lr=base_lr)
        lrSchedulerPlugin = LRSchedulerPlugin(
            MultiStepLR(optim, milestones=milestones, gamma=gamma),
            reset_lr=reset_lr, reset_scheduler=reset_scheduler)

        cl_strategy = Naive(model, optim, CrossEntropyLoss(), train_mb_size=32,
                            train_epochs=epochs, eval_mb_size=100,
                            plugins=[lrSchedulerPlugin, TestPlugin(expected)])

        cl_strategy.train(scenario.train_stream[0])
        cl_strategy.train(scenario.train_stream[1])


if __name__ == '__main__':
    unittest.main()
