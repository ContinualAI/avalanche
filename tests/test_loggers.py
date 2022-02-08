"""Test for loggers

Right now they don't do much but they at least check that the loggers run
without errors.
"""
import unittest

from torch.optim import SGD

from avalanche.evaluation.metrics import loss_metrics
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from tests.unit_tests_utils import get_fast_benchmark

from avalanche.logging import TextLogger, TensorboardLogger, InteractiveLogger
from avalanche.logging.csv_logger import CSVLogger
import tempfile


class TestLoggers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = get_fast_benchmark()
        cls.model = SimpleMLP(input_size=6, hidden_size=10)
        cls.optimizer = SGD(cls.model.parameters(), lr=0.001)
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.logdir = cls.tempdir.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.__exit__(None, None, None)

    def test_text_logger(self):
        logp = TextLogger()
        self._test_logger(logp)

    def test_tensorboard_logger(self):
        logp = TensorboardLogger(self.logdir)
        self._test_logger(logp)

    def test_interactive_logger(self):
        logp = InteractiveLogger()
        self._test_logger(logp)

    def test_csv_logger(self):
        logp = CSVLogger(log_folder=self.logdir)
        self._test_logger(logp)

    def _test_logger(self, logp):
        evalp = EvaluationPlugin(
            loss_metrics(minibatch=True, epoch=True,
                         experience=True, stream=True),
            loggers=[logp]
        )
        strat = Naive(self.model, self.optimizer, evaluator=evalp,
                      train_mb_size=32)
        for e in self.benchmark.train_stream:
            strat.train(e)
        strat.eval(self.benchmark.train_stream)
