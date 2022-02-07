#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.optim import SGD
import unittest
import copy

from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)
from avalanche.models import SimpleCNN, SimpleMLP, as_multitask
from avalanche.training.supervised import Naive
from avalanche.training.plugins import LwFPlugin, EWCPlugin

from tests.unit_tests_utils import common_setups, get_fast_benchmark, get_device


class ConversionMethodTests(unittest.TestCase):
    def setUp(self):
        common_setups()
        self.benchmark = get_fast_benchmark(use_task_labels=True, shuffle=True)
        self.device = get_device()

    def test_modules(self):
        modules = [
            (SimpleMLP(input_size=32 * 32 * 3), "classifier"),
            (SimpleCNN(), "classifier"),
        ]
        for m in modules:
            self._test_modules(*m)

    def test_outputs(self):
        modules = [
            (SimpleMLP(input_size=32 * 32 * 3), "classifier"),
            (SimpleCNN(), "classifier"),
        ]
        for m in modules:
            self._test_outputs(*m)

    def test_integration(self):
        modules = [(SimpleMLP(input_size=6), "classifier")]
        for m, name in modules:
            self._test_integration(copy.deepcopy(m), name)
            self._test_integration(
                copy.deepcopy(m), name, plugins=[LwFPlugin()]
            )
            self._test_integration(
                copy.deepcopy(m), name, plugins=[EWCPlugin(0.5)]
            )

    def test_initialisation(self):
        module = SimpleMLP()
        module = module.to(self.device)
        old_classifier_weight = torch.clone(module.classifier.weight)
        old_classifier_bias = torch.clone(module.classifier.bias)
        module = as_multitask(module, "classifier")
        module = module.to(self.device)
        new_classifier_weight = torch.clone(
            module.classifier.classifiers["0"].classifier.weight
        )
        new_classifier_bias = torch.clone(
            module.classifier.classifiers["0"].classifier.bias
        )
        self.assertTrue(
            torch.equal(old_classifier_weight, new_classifier_weight)
        )
        self.assertTrue(torch.equal(old_classifier_bias, new_classifier_bias))

    def _test_outputs(self, module, clf_name):
        test_input = torch.rand(10, 3, 32, 32)
        test_input = test_input.to(self.device)
        module_singletask = copy.deepcopy(module)
        module_multitask = as_multitask(module, clf_name)
        module_multitask = module_multitask.to(self.device)
        module_singletask = module_singletask.to(self.device)

        # Put in eval mode to deactivate dropouts
        module_singletask.eval()
        module_multitask.eval()

        out_single_task = module_singletask(test_input)
        out_multi_task = module_multitask(test_input, task_labels=0)
        self.assertTrue(torch.equal(out_single_task, out_multi_task))

    def _test_modules(self, module, clf_name):
        old_param_total = sum([torch.numel(p) for p in module.parameters()])

        module = as_multitask(module, clf_name)
        module = module.to(self.device)
        self.assertIsInstance(module, MultiTaskModule)
        self.assertIsInstance(getattr(module, clf_name), MultiHeadClassifier)

        test_input = torch.ones(5, 3, 32, 32)
        task_labels = torch.zeros(5, dtype=torch.long)

        test_input = test_input.to(self.device)
        task_labels = task_labels.to(self.device)

        # One task label
        output = module(test_input, task_labels=0)

        # Several ones
        output = module(test_input, task_labels=task_labels)

        # Change attribute
        module.non_module_attribute = 10
        self.assertEqual(module.model.non_module_attribute, 10)

        module.non_module_attribute += 5

        # Extract params and state dict
        new_param_total = sum([torch.numel(p) for p in module.parameters()])
        self.assertEqual(new_param_total, old_param_total)
        state = module.state_dict()

        # Functions returning references
        module = module.train()
        self.assertIsInstance(module, MultiTaskModule)
        self.assertTrue(module.training)
        module = module.eval()
        self.assertIsInstance(module, MultiTaskModule)
        self.assertFalse(module.training)
        if self.device == "cuda":
            module = module.cuda()
            self.assertIsInstance(module, MultiTaskModule)

    def _test_integration(self, module, clf_name, plugins=[]):
        module = as_multitask(module, clf_name)
        module = module.to(self.device)
        optimizer = SGD(
            module.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0002
        )

        strategy = Naive(
            module,
            optimizer,
            train_mb_size=32,
            eval_mb_size=32,
            device=self.device,
            plugins=plugins,
        )

        for t, experience in enumerate(self.benchmark.train_stream):
            strategy.train(experience)
            strategy.eval(self.benchmark.test_stream[: t + 1])
