import unittest

import torch
import torch.nn as nn

from avalanche.models import FeCAMClassifier, NCMClassifier, SimpleMLP, TrainEvalModel
from avalanche.training.plugins import (
    CurrentDataFeCAMUpdate,
    CurrentDataNCMUpdate,
    FeCAMOracle,
    MemoryFeCAMUpdate,
    MemoryNCMUpdate,
    NCMOracle,
)
from avalanche.training.supervised import Naive
from tests.unit_tests_utils import load_benchmark


class UpdateNCMTest(unittest.TestCase):
    def create_strategy_and_benchmark(self):
        model = SimpleMLP(input_size=6)
        old_layer = model.classifier
        model.classifier = nn.Identity()
        model = TrainEvalModel(
            model, train_classifier=old_layer, eval_classifier=NCMClassifier()
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = Naive(model, optimizer)
        benchmark = load_benchmark()
        return strategy, benchmark

    def test_current_update(self):
        plugin = CurrentDataNCMUpdate()
        self._test_plugin(plugin)

    def test_memory_update(self):
        plugin = MemoryNCMUpdate(100)
        self._test_plugin(plugin)

    def test_oracle_update(self):
        plugin = NCMOracle()
        self._test_plugin(plugin)

    def _test_plugin(self, plugin):
        strategy, benchmark = self.create_strategy_and_benchmark()
        strategy.plugins.append(plugin)
        strategy.experience = benchmark.train_stream[0]
        test_experience = benchmark.test_stream[0]
        strategy._after_training_exp()
        strategy.model.eval()

        loader = iter(
            torch.utils.data.DataLoader(test_experience.dataset, batch_size=10)
        )
        batch_x, batch_y, batch_t = next(loader)
        result = strategy.model(batch_x)


class UpdateFeCAMTest(unittest.TestCase):
    def create_strategy_and_benchmark(self):
        model = SimpleMLP(input_size=6)
        old_layer = model.classifier
        model.classifier = nn.Identity()
        model = TrainEvalModel(
            model, train_classifier=old_layer, eval_classifier=FeCAMClassifier()
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = Naive(model, optimizer)
        benchmark = load_benchmark()
        return strategy, benchmark

    def test_current_update(self):
        plugin = CurrentDataFeCAMUpdate()
        self._test_plugin(plugin)

    def test_memory_update(self):
        plugin = MemoryFeCAMUpdate(100)
        self._test_plugin(plugin)

    def test_oracle_update(self):
        plugin = FeCAMOracle()
        self._test_plugin(plugin)

    def _test_plugin(self, plugin):
        strategy, benchmark = self.create_strategy_and_benchmark()
        strategy.plugins.append(plugin)
        strategy.experience = benchmark.train_stream[0]
        test_experience = benchmark.test_stream[0]
        strategy._after_training_exp()
        strategy.model.eval()

        loader = iter(
            torch.utils.data.DataLoader(test_experience.dataset, batch_size=10)
        )
        batch_x, batch_y, batch_t = next(loader)
        result = strategy.model(batch_x)
