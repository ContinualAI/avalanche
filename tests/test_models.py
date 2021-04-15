import copy
import sys

import unittest
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from avalanche.logging import TextLogger
from avalanche.models import MTSimpleMLP, SimpleMLP, IncrementalClassifier, MultiHeadClassifier
from avalanche.training.strategies import Naive
from tests.unit_tests_utils import common_setups, load_scenario, get_fast_scenario


class PluginTests(unittest.TestCase):
    def setUp(self):
        common_setups()
        self.scenario = get_fast_scenario(use_task_labels=False, shuffle=False)

    def test_incremental_classifier(self):
        model = SimpleMLP(input_size=6, hidden_size=10)
        model.classifier = IncrementalClassifier(in_features=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        scenario = self.scenario

        strategy = Naive(model, optimizer, criterion,
                         train_mb_size=100, train_epochs=1,
                         eval_mb_size=100, device='cpu')
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        print("Current Classes: ", scenario.train_stream[0].classes_in_this_experience)
        print("Current Classes: ", scenario.train_stream[4].classes_in_this_experience)

        # train on first task
        strategy.train(scenario.train_stream[0])
        w_ptr = model.classifier.classifier.weight.data_ptr()
        b_ptr = model.classifier.classifier.bias.data_ptr()
        opt_params_ptrs = [w.data_ptr() for group in optimizer.param_groups
                           for w in group['params']]
        # classifier params should be optimized
        assert w_ptr in opt_params_ptrs
        assert b_ptr in opt_params_ptrs

        # train again on the same task.
        strategy.train(scenario.train_stream[0])
        # parameters should not change.
        assert w_ptr == model.classifier.classifier.weight.data_ptr()
        assert b_ptr == model.classifier.classifier.bias.data_ptr()
        # the same classifier params should still be optimized
        assert w_ptr in opt_params_ptrs
        assert b_ptr in opt_params_ptrs

        # update classifier with new classes.
        old_w_ptr, old_b_ptr = w_ptr, b_ptr
        strategy.train(scenario.train_stream[4])
        opt_params_ptrs = [w.data_ptr() for group in optimizer.param_groups
                           for w in group['params']]
        new_w_ptr = model.classifier.classifier.weight.data_ptr()
        new_b_ptr = model.classifier.classifier.bias.data_ptr()
        # weights should change.
        assert old_w_ptr != new_w_ptr
        assert old_b_ptr != new_b_ptr
        # Old params should not be optimized. New params should be optimized.
        assert old_w_ptr not in opt_params_ptrs
        assert old_b_ptr not in opt_params_ptrs
        assert new_w_ptr in opt_params_ptrs
        assert new_b_ptr in opt_params_ptrs

    def test_incremental_classifier_weight_update(self):
        model = IncrementalClassifier(in_features=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        scenario = self.scenario

        strategy = Naive(model, optimizer, criterion,
                         train_mb_size=100, train_epochs=1,
                         eval_mb_size=100, device='cpu')
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]

        # train on first task
        w_old = model.classifier.weight.clone()
        b_old = model.classifier.bias.clone()

        # adaptation. Increase number of classes
        dataset = scenario.train_stream[4].dataset
        model.adaptation(dataset)
        w_new = model.classifier.weight.clone()
        b_new = model.classifier.bias.clone()

        # old weights should be copied correctly.
        assert torch.equal(w_old, w_new[:w_old.shape[0]])
        assert torch.equal(b_old, b_new[:w_old.shape[0]])

        # shape should be correct.
        assert w_new.shape[0] == max(dataset.targets) + 1
        assert b_new.shape[0] == max(dataset.targets) + 1

    def test_multihead_head_creation(self):
        # Check if the optimizer is updated correctly
        # when heads are created and updated.
        model = MTSimpleMLP(input_size=6, hidden_size=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        scenario = get_fast_scenario(use_task_labels=True, shuffle=False)

        strategy = Naive(model, optimizer, criterion,
                         train_mb_size=100, train_epochs=1,
                         eval_mb_size=100, device='cpu')
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        print("Current Classes: ", scenario.train_stream[4].classes_in_this_experience)
        print("Current Classes: ", scenario.train_stream[0].classes_in_this_experience)

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

        assert w_ptr not in opt_params_ptrs  # head0 has been updated
        assert b_ptr not in opt_params_ptrs  # head0 has been updated
        assert w_ptr_t0 in opt_params_ptrs
        assert b_ptr_t0 in opt_params_ptrs
        assert w_ptr_new in opt_params_ptrs
        assert b_ptr_new in opt_params_ptrs

    def test_multihead_head_selection(self):
        # Check if the optimizer is updated correctly
        # when heads are created and updated.
        model = MultiHeadClassifier(in_features=6)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        scenario = get_fast_scenario(use_task_labels=True, shuffle=False)

        strategy = Naive(model, optimizer, criterion,
                         train_mb_size=100, train_epochs=1,
                         eval_mb_size=100, device='cpu')
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]

        # initialize head
        strategy.train(scenario.train_stream[0])
        strategy.train(scenario.train_stream[4])

        # create models with fixed head
        model_t0 = model.classifiers['0']
        model_t4 = model.classifiers['4']

        # check head task0
        for x, y, t in DataLoader(scenario.train_stream[0].dataset):
            y_mh = model(x, t)
            y_t = model_t0(x)
            assert ((y_mh - y_t) ** 2).sum() < 1.e-7
            break

        # check head task4
        for x, y, t in DataLoader(scenario.train_stream[4].dataset):
            y_mh = model(x, t)
            y_t = model_t4(x)
            assert ((y_mh - y_t) ** 2).sum() < 1.e-7
            break
