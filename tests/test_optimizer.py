#!/usr/bin/env python3
import copy
import os
import sys
import tempfile
import unittest

import numpy as np
import pytorchcv.models.pyramidnet_cifar
import torch
import torch.nn.functional as F
from tests.benchmarks.utils.test_avalanche_classification_dataset import get_mbatch
from tests.unit_tests_utils import common_setups, get_fast_benchmark, load_benchmark
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.logging import TextLogger
from avalanche.models import (
    IncrementalClassifier,
    MTSimpleMLP,
    MultiHeadClassifier,
    SimpleMLP,
)
from avalanche.models.cosine_layer import CosineLinear, SplitCosineLinear
from avalanche.models.dynamic_optimizers import (
    add_new_params_to_optimizer,
    update_optimizer,
)
from avalanche.models.pytorchcv_wrapper import densenet, get_model, pyramidnet, resnet
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.supervised import Naive


class TorchWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, *args):
        return self.backbone(*args)


class DynamicOptimizersTests(unittest.TestCase):
    if "USE_GPU" in os.environ:
        use_gpu = os.environ["USE_GPU"].lower() in ["true"]
    else:
        use_gpu = False

    print("Test on GPU:", use_gpu)

    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    def setUp(self):
        common_setups()

    def _iterate_optimizers(self, model, *optimizers):
        for opt_class in optimizers:
            if opt_class == "SGDmom":
                yield torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            if opt_class == "SGD":
                yield torch.optim.SGD(model.parameters(), lr=0.1)
            if opt_class == "Adam":
                yield torch.optim.Adam(model.parameters(), lr=0.001)
            if opt_class == "AdamW":
                yield torch.optim.AdamW(model.parameters(), lr=0.001)

    def _is_param_in_optimizer(self, param, optimizer):
        for group in optimizer.param_groups:
            for curr_p in group["params"]:
                if hash(curr_p) == hash(param):
                    return True
        return False

    def _is_param_in_optimizer_group(self, param, optimizer):
        for group_idx, group in enumerate(optimizer.param_groups):
            for curr_p in group["params"]:
                if hash(curr_p) == hash(param):
                    return group_idx
        return None

    def load_benchmark(self, use_task_labels=False):
        return get_fast_benchmark(use_task_labels=use_task_labels)

    def init_scenario(self, multi_task=False):
        model = self.get_model(multi_task=multi_task)
        criterion = CrossEntropyLoss()
        benchmark = self.load_benchmark(use_task_labels=multi_task)
        return model, criterion, benchmark

    def test_optimizer_update(self):
        model = SimpleMLP()
        optimizer = SGD(model.parameters(), lr=1e-3)
        strategy = Naive(model=model, optimizer=optimizer)

        # check add new parameter
        p = torch.nn.Parameter(torch.zeros(10, 10))
        optimizer.param_groups[0]["params"].append(p)
        assert self._is_param_in_optimizer(p, strategy.optimizer)

        # check new_param is in optimizer
        # check old_param is NOT in optimizer
        p_new = torch.nn.Parameter(torch.zeros(10, 10))

        # Here we cannot know what parameter group but there is only one so it should work
        new_parameters = {"new_param": p_new}
        new_parameters.update(dict(model.named_parameters()))
        optimized = update_optimizer(
            optimizer, new_parameters, {"old_param": p}, remove_params=True
        )
        self.assertTrue("new_param" in optimized)
        self.assertFalse("old_param" in optimized)
        self.assertTrue(self._is_param_in_optimizer(p_new, strategy.optimizer))
        self.assertFalse(self._is_param_in_optimizer(p, strategy.optimizer))

    def test_optimizers(self):
        """
        Run a series of tests on various pytorch optimizers
        """

        # SIT scenario
        model, criterion, benchmark = self.init_scenario(multi_task=True)
        for optimizer in self._iterate_optimizers(
            model, "SGDmom", "Adam", "SGD", "AdamW"
        ):
            strategy = Naive(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_mb_size=64,
                device=self.device,
                eval_mb_size=50,
                train_epochs=2,
            )
            self._test_optimizer(strategy)
            self._test_optimizer_state(strategy)

    def test_optimizer_groups_clf_til(self):
        """
        Tests the automatic assignation of new
        MultiHead parameters to the optimizer
        """
        model, criterion, benchmark = self.init_scenario(multi_task=True)

        g1 = []
        g2 = []
        for n, p in model.named_parameters():
            if "classifier" in n:
                g1.append(p)
            else:
                g2.append(p)

        optimizer = SGD([{"params": g1, "lr": 0.1}, {"params": g2, "lr": 0.05}])

        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=64,
            device=self.device,
            eval_mb_size=50,
            train_epochs=2,
        )

        for experience in benchmark.train_stream:
            strategy.train(experience)

            for n, p in model.named_parameters():
                assert self._is_param_in_optimizer(p, strategy.optimizer)
                if "classifier" in n:
                    self.assertEqual(
                        self._is_param_in_optimizer_group(p, strategy.optimizer), 0
                    )
                else:
                    self.assertEqual(
                        self._is_param_in_optimizer_group(p, strategy.optimizer), 1
                    )

    def test_optimizer_groups_clf_cil(self):
        """
        Tests the automatic assignation of new
        IncrementalClassifier parameters to the optimizer
        """
        model, criterion, benchmark = self.init_scenario(multi_task=False)

        g1 = []
        g2 = []
        for n, p in model.named_parameters():
            if "classifier" in n:
                g1.append(p)
            else:
                g2.append(p)

        optimizer = SGD([{"params": g1, "lr": 0.1}, {"params": g2, "lr": 0.05}])

        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=64,
            device=self.device,
            eval_mb_size=50,
            train_epochs=2,
        )

        for experience in benchmark.train_stream:
            strategy.train(experience)

            for n, p in model.named_parameters():
                assert self._is_param_in_optimizer(p, strategy.optimizer)
                if "classifier" in n:
                    self.assertEqual(
                        self._is_param_in_optimizer_group(p, strategy.optimizer), 0
                    )
                else:
                    self.assertEqual(
                        self._is_param_in_optimizer_group(p, strategy.optimizer), 1
                    )

    def test_optimizer_groups_manual_addition(self):
        """
        Tests the manual addition of a new parameter group
        mixed with existing MultiHeadClassifier
        """
        model, criterion, benchmark = self.init_scenario(multi_task=True)

        g1 = []
        g2 = []
        for n, p in model.named_parameters():
            if "classifier" in n:
                g1.append(p)
            else:
                g2.append(p)

        optimizer = SGD([{"params": g1, "lr": 0.1}, {"params": g2, "lr": 0.05}])

        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=64,
            device=self.device,
            eval_mb_size=50,
            train_epochs=2,
        )

        experience_0 = benchmark.train_stream[0]
        experience_1 = benchmark.train_stream[1]

        strategy.train(experience_0)

        # Add some new parameter and assign it manually to param group
        model.new_module1 = torch.nn.Linear(10, 10)
        model.new_module2 = torch.nn.Linear(10, 10)
        strategy.optimizer.param_groups[1]["params"] += list(
            model.new_module1.parameters()
        )
        strategy.optimizer.add_param_group(
            {"params": list(model.new_module2.parameters()), "lr": 0.001}
        )

        # Also add one but to a new param group

        strategy.train(experience_1)

        for n, p in model.named_parameters():
            assert self._is_param_in_optimizer(p, strategy.optimizer)
            if "classifier" in n:
                self.assertEqual(
                    self._is_param_in_optimizer_group(p, strategy.optimizer), 0
                )
            elif "new_module2" in n:
                self.assertEqual(
                    self._is_param_in_optimizer_group(p, strategy.optimizer), 2
                )
            else:
                self.assertEqual(
                    self._is_param_in_optimizer_group(p, strategy.optimizer), 1
                )

    def test_optimizer_groups_rename(self):
        """
        Tests the correct reassignation to
        existing parameter groups after
        parameter renaming
        """
        model, criterion, benchmark = self.init_scenario(multi_task=False)

        g1 = []
        g2 = []
        for n, p in model.named_parameters():
            if "classifier" in n:
                g1.append(p)
            else:
                g2.append(p)

        optimizer = SGD([{"params": g1, "lr": 0.1}, {"params": g2, "lr": 0.05}])

        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=64,
            device=self.device,
            eval_mb_size=50,
            train_epochs=2,
        )

        strategy.make_optimizer()

        # Check parameter groups
        for n, p in model.named_parameters():
            assert self._is_param_in_optimizer(p, strategy.optimizer)
            if "classifier" in n:
                self.assertEqual(
                    self._is_param_in_optimizer_group(p, strategy.optimizer), 0
                )
            else:
                self.assertEqual(
                    self._is_param_in_optimizer_group(p, strategy.optimizer), 1
                )

        # Rename parameters
        strategy.model = TorchWrapper(strategy.model)

        strategy.make_optimizer()

        # Check parameter groups are still the same
        for n, p in model.named_parameters():
            assert self._is_param_in_optimizer(p, strategy.optimizer)
            if "classifier" in n:
                self.assertEqual(
                    self._is_param_in_optimizer_group(p, strategy.optimizer), 0
                )
            else:
                self.assertEqual(
                    self._is_param_in_optimizer_group(p, strategy.optimizer), 1
                )

    # Needs torch 2.0 ?
    def test_checkpointing(self):
        model, criterion, benchmark = self.init_scenario(multi_task=True)
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=64,
            device=self.device,
            eval_mb_size=50,
            train_epochs=2,
        )
        experience_0 = benchmark.train_stream[0]
        strategy.train(experience_0)
        old_state = copy.deepcopy(strategy.optimizer.state)
        save_checkpoint(strategy, "./checkpoint.pt")

        del strategy

        model, criterion, benchmark = self.init_scenario(multi_task=True)
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=64,
            device=self.device,
            eval_mb_size=50,
            train_epochs=2,
        )
        strategy, exp_counter = maybe_load_checkpoint(
            strategy, "./checkpoint.pt", strategy.device
        )

        # Check that the state has been well serialized
        self.assertEqual(len(strategy.optimizer.state), len(old_state))
        for (key_new, value_new_dict), (key_old, value_old_dict) in zip(
            strategy.optimizer.state.items(), old_state.items()
        ):
            self.assertTrue(torch.equal(key_new, key_old))

            value_new = value_new_dict["momentum_buffer"]
            value_old = value_old_dict["momentum_buffer"]

            # Empty state
            if len(value_new) == 0 or len(value_old) == 0:
                self.assertTrue(len(value_new) == len(value_old))
            else:
                self.assertTrue(torch.equal(value_new, value_old))

        experience_1 = benchmark.train_stream[1]
        strategy.train(experience_1)
        os.remove("./checkpoint.pt")

    def _test_optimizer(self, strategy):
        # Add a parameter
        module = torch.nn.Linear(10, 10)
        param1 = list(module.parameters())[0]
        strategy.make_optimizer()
        self.assertFalse(self._is_param_in_optimizer(param1, strategy.optimizer))
        strategy.model.add_module("new_module", module)
        strategy.make_optimizer()
        self.assertTrue(self._is_param_in_optimizer(param1, strategy.optimizer))
        # Remove a parameter
        del strategy.model.new_module

        strategy.make_optimizer(remove_params=False)
        self.assertTrue(self._is_param_in_optimizer(param1, strategy.optimizer))

        strategy.make_optimizer(remove_params=True)
        self.assertFalse(self._is_param_in_optimizer(param1, strategy.optimizer))

    def _test_optimizer_state(self, strategy):
        # Add Two modules
        module1 = torch.nn.Linear(10, 10)
        module2 = torch.nn.Linear(10, 10)
        param1 = list(module1.parameters())[0]
        param2 = list(module2.parameters())[0]
        strategy.model.add_module("new_module1", module1)
        strategy.model.add_module("new_module2", module2)

        strategy.make_optimizer(remove_params=True)

        self.assertTrue(self._is_param_in_optimizer(param1, strategy.optimizer))
        self.assertTrue(self._is_param_in_optimizer(param2, strategy.optimizer))

        # Make an operation
        self._optimizer_op(strategy.optimizer, module1.weight + module2.weight)

        if len(strategy.optimizer.state) > 0:
            assert param1 in strategy.optimizer.state
            assert param2 in strategy.optimizer.state

        # Remove one module
        del strategy.model.new_module1

        strategy.make_optimizer(remove_params=True)

        # Make an operation
        self._optimizer_op(strategy.optimizer, module1.weight + module2.weight)

        if len(strategy.optimizer.state) > 0:
            assert param1 not in strategy.optimizer.state
            assert param2 in strategy.optimizer.state

        # Change one module size
        strategy.model.new_module2 = torch.nn.Linear(10, 5)
        strategy.make_optimizer(remove_params=True)

        # Make an operation
        self._optimizer_op(strategy.optimizer, module1.weight + module2.weight)

        if len(strategy.optimizer.state) > 0:
            assert param1 not in strategy.optimizer.state
            assert param2 not in strategy.optimizer.state

    def _optimizer_op(self, optimizer, param):
        optimizer.zero_grad()
        loss = torch.mean(param)
        loss.backward()
        optimizer.step()

    def get_model(self, multi_task=False):
        if multi_task:
            model = MTSimpleMLP(input_size=6, hidden_size=10)
        else:
            model = SimpleMLP(input_size=6, hidden_size=10)
            model.classifier = IncrementalClassifier(10, 1)
        return model
