import os
import sys
import unittest

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.evaluation.metrics import StreamAccuracy, loss_metrics
from avalanche.logging import TextLogger, InteractiveLogger
from avalanche.models import SimpleMLP, MTSimpleMLP, IncrementalClassifier, PNN
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.supervised import Naive
from avalanche.training.supervised.cumulative import Cumulative
from avalanche.training.supervised.icarl import ICaRL
from avalanche.training.supervised.joint_training import AlreadyTrainedError
from avalanche.training.supervised.strategy_wrappers import PNNStrategy
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.base import _group_experiences_by_stream
from avalanche.training.utils import get_last_fc_layer
from tests.training.test_strategy_utils import run_strategy
from tests.unit_tests_utils import get_fast_benchmark, get_device


def iterate_optimizers(model, *optimizers):
    for opt_class in optimizers:
        if opt_class == "SGD":
            yield torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        if opt_class == "Adam":
            yield torch.optim.Adam(model.parameters(), lr=0.001)
        if opt_class == "AdamW":
            yield torch.optim.AdamW(model.parameters(), lr=0.001)


def _check_id_in_(param, param_list):
    for p in param_list:
        if id(p) == id(param):
            return True
    return False


class TestOptimizerUpdate(unittest.TestCase):
    if "USE_GPU" in os.environ:
        use_gpu = os.environ["USE_GPU"].lower() in ["true"]
    else:
        use_gpu = False

    print("Test on GPU:", use_gpu)

    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    def init_model(self, multi_task=False):
        model = self.get_model(multi_task=multi_task)
        criterion = CrossEntropyLoss()
        return model, criterion

    def test_optimizer(self):
        # SIT scenario
        model, criterion = self.init_model(multi_task=True)
        for optimizer in iterate_optimizers(model, "Adam", "SGD", "AdamW"):
            strategy = Naive(
                model,
                optimizer,
                criterion,
                train_mb_size=64,
                device=self.device,
                eval_mb_size=50,
                train_epochs=2,
            )
            self._test_optimizer(strategy)

    def _test_optimizer(self, strategy):
        # Add a parameter
        module = torch.nn.Linear(10, 10)
        param1 = list(module.parameters())[0]
        strategy.make_optimizer()
        self.assertFalse(_check_id_in_(param1, 
                         strategy.optimizer.param_groups[0]["params"]))
        strategy.model.add_module("new_module", module)
        strategy.make_optimizer()
        self.assertTrue(_check_id_in_(param1, 
                        strategy.optimizer.param_groups[0]["params"]))
        # Remove a parameter
        del strategy.model.new_module
        strategy.make_optimizer()
        self.assertFalse(_check_id_in_(param1, 
                         strategy.optimizer.param_groups[0]["params"]))

    def get_model(self, multi_task=False):
        if multi_task:
            model = MTSimpleMLP(input_size=6, hidden_size=10)
        else:
            model = SimpleMLP(input_size=6, hidden_size=10)
        return model


if __name__ == "__main__":
    unittest.main()
