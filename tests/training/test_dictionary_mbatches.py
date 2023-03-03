import unittest

from avalanche.benchmarks.utils import DataAttribute, ConstantSequence
from avalanche.models import SimpleMLP
from avalanche.training.plugins import ReplayPlugin

import torch

import avalanche
import torch.nn

from avalanche.benchmarks import CLScenario, CLStream, CLExperience
from avalanche.evaluation.metrics import accuracy_metrics
import avalanche.training.templates.base
from avalanche.benchmarks.utils import AvalancheDataset


def collate_dictionaries(dicts):
    """
    Collate a list of dictionaries into a single dictionary.
    """
    if len(dicts) == 0:
        return

    res = {}
    for key in dicts[0].keys():
        els = [d[key] for d in dicts]
        if isinstance(els[0], torch.Tensor):
            res[key] = torch.stack(els)
        elif isinstance(els[0], int):
            res[key] = torch.tensor(els)
        else:
            res[key] = els
    return res


class NaiveDictData(avalanche.training.Naive):
    @property
    def mb_x(self):
        return self.mbatch["x"]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        return self.mbatch["y"]

    def mb_task_id(self):
        return self.mbatch["targets_task_labels"]

    def _unpack_minibatch(self):
        """minibatches are dictionaries of tensors.
        Move tensors to the current device."""
        for k in self.mbatch.keys():
            self.mbatch[k] = self.mbatch[k].to(self.device)


class TestDictionaryDatasets(unittest.TestCase):
    def test_dictionary_datasets(self):
        rand_x = torch.randn(100, 10)
        rand_y = torch.randint(0, 2, (100,))
        data = [{"x": x, "y": y} for x, y in zip(rand_x, rand_y)]
        avl_data = AvalancheDataset([data], collate_fn=collate_dictionaries)

        real_sample = data[0]
        avl_sample = avl_data[0]
        for k, v in real_sample.items():
            assert k in avl_sample
            assert torch.equal(avl_sample[k], v)

    def test_dictionary_train_replay(self):
        rand_x = torch.randn(100, 10)
        rand_y = torch.randint(0, 2, (100,))
        data = [{"x": x, "y": y} for x, y in zip(rand_x, rand_y)]

        train_exps, test_exps = [], []
        for i in range(0, 2):
            tl = ConstantSequence(i, len(data))
            tl = DataAttribute(tl, "targets_task_labels", use_in_getitem=True)
            av_data = AvalancheDataset(
                [data], data_attributes=[tl], collate_fn=collate_dictionaries
            )
            exp = CLExperience()
            exp.dataset = av_data
            train_exps.append(exp)
            test_exps.append(exp)

        benchmark = CLScenario(
            [CLStream("train", train_exps), CLStream("test", test_exps)]
        )
        eval_plugin = avalanche.training.plugins.EvaluationPlugin(
            avalanche.evaluation.metrics.loss_metrics(
                epoch=True, experience=True, stream=True
            ),
            loggers=[avalanche.logging.InteractiveLogger()],
            strict_checks=False,
        )

        plugins = [ReplayPlugin(mem_size=200)]
        model = SimpleMLP(input_size=10, num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=2)
        strategy = NaiveDictData(
            model,
            optimizer,
            evaluator=eval_plugin,
            train_mb_size=10,
            plugins=plugins,
        )
        for experience in benchmark.train_stream:
            strategy.train(experience)
