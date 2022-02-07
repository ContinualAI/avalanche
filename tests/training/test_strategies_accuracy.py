################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2021                                                             #
# Author(s): Antonio                                                           #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import unittest

from torch import nn

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from avalanche.models import MultiHeadClassifier, PNN
from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised.cumulative import Cumulative
from avalanche.evaluation.metrics import StreamAccuracy, ExperienceAccuracy
from avalanche.training.supervised.strategy_wrappers import PNNStrategy

from tests.unit_tests_utils import get_fast_benchmark, get_device


class TestMLP(nn.Module):
    def __init__(self, num_classes=10, input_size=6, hidden_size=50):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

        self._hidden_size = hidden_size
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.layers(x)
        x = self.classifier(x)
        return x


class MHTestMLP(TestMLP, MultiTaskModule):
    def __init__(self, num_classes=10, input_size=6, hidden_size=50):
        super().__init__()
        self.classifier = MultiHeadClassifier(self._hidden_size, num_classes)

    def forward(self, x, task_labels):
        x = self.layers(x)
        x = self.classifier(x, task_labels)
        return x


class StrategyTest(unittest.TestCase):
    def test_multihead_cumulative(self):
        # check that multi-head reaches high enough accuracy.
        # Ensure nothing weird is happening with the multiple heads.
        model = MHTestMLP(input_size=6, hidden_size=100)
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=1)

        main_metric = StreamAccuracy()
        exp_acc = ExperienceAccuracy()
        evalp = EvaluationPlugin(main_metric, exp_acc, loggers=None)
        strategy = Cumulative(
            model,
            optimizer,
            criterion,
            train_mb_size=32,
            device=get_device(),
            eval_mb_size=512,
            train_epochs=3,
            evaluator=evalp,
        )
        benchmark = get_fast_benchmark(use_task_labels=True)

        for train_batch_info in benchmark.train_stream:
            strategy.train(train_batch_info)
        strategy.eval(benchmark.train_stream[:])
        print("TRAIN STREAM ACC: ", main_metric.result())
        assert (
            sum(main_metric.result().values())
            / float(len(main_metric.result().keys()))
            > 0.7
        )

    def test_pnn(self):
        # check that pnn reaches high enough accuracy.
        # Ensure nothing weird is happening with the multiple heads.
        main_metric = StreamAccuracy()
        exp_acc = ExperienceAccuracy()
        evalp = EvaluationPlugin(main_metric, exp_acc, loggers=None)
        model = PNN(num_layers=1, in_features=6, hidden_features_per_column=50)
        optimizer = SGD(model.parameters(), lr=0.1)
        strategy = PNNStrategy(
            model,
            optimizer,
            train_mb_size=32,
            device=get_device(),
            eval_mb_size=512,
            train_epochs=1,
            evaluator=evalp,
        )
        benchmark = get_fast_benchmark(use_task_labels=True)

        for train_batch_info in benchmark.train_stream:
            strategy.train(train_batch_info)

        strategy.eval(benchmark.train_stream[:])
        print("TRAIN STREAM ACC: ", main_metric.result())
        assert (
            sum(main_metric.result().values())
            / float(len(main_metric.result().keys()))
            > 0.5
        )


if __name__ == "__main__":
    unittest.main()
