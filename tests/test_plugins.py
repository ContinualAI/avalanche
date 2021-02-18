import sys

import torch
import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import TensorDataset

from avalanche.benchmarks import nc_scenario
from avalanche.logging import TextLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import StrategyPlugin, MultiHeadPlugin
from avalanche.training.strategies import Naive


class MockPlugin(StrategyPlugin):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.activated = [False for _ in range(22)]

    def before_training_step(self, strategy, **kwargs):
        self.activated[0] = True

    def adapt_train_dataset(self, strategy, **kwargs):
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

    def after_training_step(self, strategy, **kwargs):
        self.activated[12] = True

    def before_eval(self, strategy, **kwargs):
        self.activated[13] = True

    def adapt_eval_dataset(self, strategy, **kwargs):
        self.activated[14] = True

    def before_eval_step(self, strategy, **kwargs):
        self.activated[15] = True

    def after_eval_step(self, strategy, **kwargs):
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

    def test_multihead_optimizer_update(self):
        # Check if the optimizer is updated correctly
        # when heads are created and updated.
        model = SimpleMLP(input_size=6, hidden_size=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        scenario = self.create_scenario()

        plug = MultiHeadPlugin(model, 'classifier')
        strategy = Naive(model, optimizer, criterion,
                         train_mb_size=100, train_epochs=1, eval_mb_size=100,
                         device='cpu', plugins=[plug]
                         )
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        print("Current Classes: ", scenario.train_stream[0].classes_in_this_step)
        print("Current Classes: ", scenario.train_stream[4].classes_in_this_step)

        # head creation
        strategy.train(scenario.train_stream[0])
        w_ptr = model.classifier.weight.data_ptr()
        b_ptr = model.classifier.bias.data_ptr()
        opt_params_ptrs = [w.data_ptr() for group in optimizer.param_groups
                           for w in group['params']]
        assert w_ptr in opt_params_ptrs
        assert b_ptr in opt_params_ptrs

        # head update
        strategy.train(scenario.train_stream[4])
        w_ptr_new = model.classifier.weight.data_ptr()
        b_ptr_new = model.classifier.bias.data_ptr()
        opt_params_ptrs = [w.data_ptr() for group in optimizer.param_groups
                           for w in group['params']]
        assert w_ptr not in opt_params_ptrs
        assert b_ptr not in opt_params_ptrs
        assert w_ptr_new in opt_params_ptrs
        assert b_ptr_new in opt_params_ptrs

    def create_scenario(self):
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
        return nc_scenario(train_dataset, test_dataset, 5, task_labels=False,
                           fixed_class_order=list(range(10)))


if __name__ == '__main__':
    unittest.main()
