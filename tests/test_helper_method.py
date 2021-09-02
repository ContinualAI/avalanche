#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.optim import SGD
import unittest

from avalanche.models.dynamic_modules import MultiTaskModule, MultiHeadClassifier
from avalanche.models import SimpleCNN, SimpleMLP
from avalanche.models.helper_method import as_multitask
from avalanche.training.strategies import Naive

from tests.unit_tests_utils import common_setups, get_fast_benchmark

class ConversionMethodTests(unittest.TestCase):
    def setUp(self):
        common_setups()
        self.benchmark = get_fast_benchmark(use_task_labels=True, shuffle=True)

    def test_modules(self):
        modules = [(SimpleMLP(input_size=32*32*3), 'classifier'), (SimpleCNN(), 'classifier')]
        for m in modules:self._test_modules(*m)

    def test_integration(self):
        modules = [(SimpleMLP(input_size=6), 'classifier')]
        for m in modules:self._test_integration(*m)

    def _test_modules(self, module, clf_name):
        old_param_total = sum([torch.numel(p) for p in module.parameters()])

        module = as_multitask(module, clf_name)
        self.assertIsInstance(module, MultiTaskModule)
        self.assertIsInstance(getattr(module, clf_name), MultiHeadClassifier)
    
        test_input = torch.ones(5, 3, 32, 32)
        task_labels = torch.zeros(5, dtype=torch.long)
    
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
        module = module.cuda()
        self.assertIsInstance(module, MultiTaskModule)

    def _test_integration(self, module, clf_name):
        module = as_multitask(module, clf_name)
        optimizer = SGD(module.parameters(), lr=0.05, 
        momentum=0.9, weight_decay=0.0002)
        
        strategy = Naive(module, optimizer, 
        train_mb_size=32, eval_mb_size=32, device='cpu')

        for t, experience in enumerate(self.benchmark.train_stream):
            strategy.train(experience)
            strategy.eval(self.benchmark.test_stream[:t+1])
