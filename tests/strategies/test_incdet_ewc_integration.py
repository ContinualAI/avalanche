import unittest
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, Linear
from torch.utils.data import TensorDataset

from avalanche.training.supervised.strategy_wrappers import Naive
from avalanche.training.plugins import HuberEWCPlugin
from avalanche.training.plugins.pseudo_annotation import PseudoAnnotationPlugin
from avalanche.benchmarks.utils.classification_dataset import _make_taskaware_tensor_classification_dataset
from avalanche.benchmarks.scenarios.deprecated.generators import dataset_benchmark


class IncDetEWCIntegrationTest(unittest.TestCase):
    def test_tiny_synthetic_two_exps(self):
        device = torch.device("cpu")
        # two tiny experiences of different classes
        x0 = torch.randn(20, 10)
        y0 = torch.zeros(20, dtype=torch.long)
        x1 = torch.randn(20, 10)
        y1 = torch.ones(20, dtype=torch.long)

        ds0 = _make_taskaware_tensor_classification_dataset(x0, y0, task_labels=0)
        ds1 = _make_taskaware_tensor_classification_dataset(x1, y1, task_labels=0)

        bench = dataset_benchmark([ds0, ds1], [ds0, ds1])

        model = Linear(10, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        huber_plugin = HuberEWCPlugin(ewc_lambda=1.0, beta=1.0)
        pseudo_plugin = PseudoAnnotationPlugin(confidence_thresh=0.99)

        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=8,
            train_epochs=1,
            eval_mb_size=8,
            device=device,
            plugins=[huber_plugin, pseudo_plugin],
        )

        for i, exp in enumerate(bench.train_stream):
            strategy.train(exp, num_workers=0, pin_memory=False)
            if i == 1:
                break

        res = strategy.eval(bench.test_stream[:1], num_workers=0, pin_memory=False)
        self.assertTrue(isinstance(res, dict))
        self.assertTrue(len(res) > 0)


if __name__ == "__main__":
    unittest.main()
