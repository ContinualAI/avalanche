import os
import sys
import unittest

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.models import SimpleMLP, MTSimpleMLP, IncrementalClassifier
from avalanche.training.supervised import Naive
from avalanche.training.checkpoint import save_checkpoint, maybe_load_checkpoint
from tests.unit_tests_utils import get_fast_benchmark, get_device


def iterate_optimizers(model, *optimizers):
    for opt_class in optimizers:
        if opt_class == "SGDmom":
            yield torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        if opt_class == "SGD":
            yield torch.optim.SGD(model.parameters(), lr=0.1)
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

    def load_benchmark(self, use_task_labels=False):
        """
        Returns a NC benchmark from a fake dataset of 10 classes, 5 experiences,
        2 classes per experience.

        :param fast_test: if True loads fake data, MNIST otherwise.
        """
        return get_fast_benchmark(use_task_labels=use_task_labels)

    def init_scenario(self, multi_task=False):
        model = self.get_model(multi_task=multi_task)
        criterion = CrossEntropyLoss()
        benchmark = self.load_benchmark(use_task_labels=multi_task)
        return model, criterion, benchmark

    def test_optimizers(self):
        # SIT scenario
        model, criterion, benchmark = self.init_scenario(multi_task=True)
        for optimizer in iterate_optimizers(
                model, "SGDmom", "Adam", "SGD", "AdamW"):
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

    # Needs torch 2.0 ?
    def test_checkpointing(self):
        model, criterion, benchmark = self.init_scenario(multi_task=True)
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=64,
            device=self.device,
            eval_mb_size=50,
            train_epochs=2,
        )
        experience_0 = benchmark.train_stream[0]
        strategy.train(experience_0)
        save_checkpoint(strategy, "./checkpoint.pt")

        del strategy

        model, criterion, benchmark = self.init_scenario(multi_task=True)
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=64,
            device=self.device,
            eval_mb_size=50,
            train_epochs=2,
        )
        strategy, exp_counter = maybe_load_checkpoint(
            strategy, "./checkpoint.pt", strategy.device
        )
        experience_1 = benchmark.train_stream[1]
        strategy.train(experience_1)

    def test_mh_classifier(self):
        model, criterion, benchmark = self.init_scenario(multi_task=True)
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=64,
            device=self.device,
            eval_mb_size=50,
            train_epochs=2,
        )
        strategy.train(benchmark.train_stream)

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
