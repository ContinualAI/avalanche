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
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from tests.unit_tests_utils import get_fast_benchmark


class TestStreamCompleteness(unittest.TestCase):
    @classmethod
    def setUp(cls) -> None:
        cls.model = SimpleMLP(input_size=6, hidden_size=10)
        cls.optimizer = SGD(cls.model.parameters(), lr=1e-3)
        cls.criterion = CrossEntropyLoss()

        cls.benchmark = get_fast_benchmark()

    def test_raise_error(self):
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(stream=True),
            loggers=None,
            benchmark=self.benchmark,
            strict_checks=True,
        )
        strategy = Naive(
            self.model,
            self.optimizer,
            self.criterion,
            train_epochs=2,
            eval_every=-1,
            evaluator=eval_plugin,
        )
        for exp in self.benchmark.train_stream:
            strategy.train(exp)
            strategy.eval(self.benchmark.test_stream)
        with self.assertRaises(ValueError):
            strategy.eval(self.benchmark.test_stream[:2])

    def test_raise_warning(self):
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(stream=True),
            loggers=None,
            benchmark=self.benchmark,
            strict_checks=False,
        )
        strategy = Naive(
            self.model,
            self.optimizer,
            self.criterion,
            train_epochs=2,
            eval_every=-1,
            evaluator=eval_plugin,
        )
        for exp in self.benchmark.train_stream:
            strategy.train(exp)
            strategy.eval(self.benchmark.test_stream)
        with self.assertWarns(UserWarning):
            strategy.eval(self.benchmark.test_stream[:2])

    def test_no_errors(self):
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(stream=True),
            loggers=None,
            benchmark=self.benchmark,
            strict_checks=True,
        )
        strategy = Naive(
            self.model,
            self.optimizer,
            self.criterion,
            train_epochs=2,
            eval_every=0,
            evaluator=eval_plugin,
        )
        for exp in self.benchmark.train_stream:
            strategy.train(exp, eval_streams=[self.benchmark.test_stream])
            strategy.eval(self.benchmark.test_stream)
