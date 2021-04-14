import sys

import unittest
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.logging import TextLogger
from avalanche.models import MTSimpleMLP
from avalanche.training.strategies import Naive
from tests.unit_tests_utils import common_setups, load_scenario


class PluginTests(unittest.TestCase):
    def setUp(self):
        common_setups()

    def test_multihead_optimizer_update(self):
        # Check if the optimizer is updated correctly
        # when heads are created and updated.
        model = MTSimpleMLP(input_size=6, hidden_size=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        scenario = load_scenario(use_task_labels=True)

        strategy = Naive(model, optimizer, criterion,
                         train_mb_size=100, train_epochs=1,
                         eval_mb_size=100, device='cpu')
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        print("Current Classes: ", scenario.train_stream[0].classes_in_this_experience)
        print("Current Classes: ", scenario.train_stream[4].classes_in_this_experience)

        # head creation
        strategy.train(scenario.train_stream[0])
        w_ptr = model.classifier.classifiers['0'].classifier.weight.data_ptr()
        b_ptr = model.classifier.classifiers['0'].classifier.bias.data_ptr()
        opt_params_ptrs = [w.data_ptr() for group in optimizer.param_groups
                           for w in group['params']]
        assert w_ptr in opt_params_ptrs
        assert b_ptr in opt_params_ptrs

        # head update
        strategy.train(scenario.train_stream[4])
        w_ptr_t0 = model.classifier.classifiers['0'].classifier.weight.data_ptr()
        b_ptr_t0 = model.classifier.classifiers['0'].classifier.bias.data_ptr()
        w_ptr_new = model.classifier.classifiers['4'].classifier.weight.data_ptr()
        b_ptr_new = model.classifier.classifiers['4'].classifier.bias.data_ptr()
        opt_params_ptrs = [w.data_ptr() for group in optimizer.param_groups
                           for w in group['params']]

        assert w_ptr not in opt_params_ptrs
        assert b_ptr not in opt_params_ptrs
        assert w_ptr_t0 in opt_params_ptrs
        assert b_ptr_t0 in opt_params_ptrs
        assert w_ptr_new in opt_params_ptrs
        assert b_ptr_new in opt_params_ptrs
