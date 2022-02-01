import unittest
from typing import List, Dict
from unittest.mock import MagicMock

import torch
from torch import tensor, Tensor, zeros
from torch.nn import CrossEntropyLoss, Module, Identity
from torch.optim import SGD

from avalanche.benchmarks.utils import (
    AvalancheDataset,
    AvalancheDatasetType,
    AvalancheTensorDataset,
)
from avalanche.models import SimpleMLP
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import (
    ExperienceBalancedBuffer,
    ClassBalancedBuffer,
    ExemplarsSelectionStrategy,
    HerdingSelectionStrategy,
    ClosestToCenterSelectionStrategy,
    ParametricBuffer,
)
from avalanche.training.supervised import Naive
from avalanche.training.templates.supervised import SupervisedTemplate
from tests.unit_tests_utils import get_fast_benchmark


class ReplayTest(unittest.TestCase):
    def test_replay_balanced_memory(self):
        mem_size = 25
        policies = [
            None,
            ExperienceBalancedBuffer(max_size=mem_size),
            ClassBalancedBuffer(max_size=mem_size),
        ]
        for policy in policies:
            self._test_replay_balanced_memory(policy, mem_size)

    def _test_replay_balanced_memory(self, storage_policy, mem_size):
        benchmark = get_fast_benchmark(use_task_labels=True)
        model = SimpleMLP(input_size=6, hidden_size=10)
        replayPlugin = ReplayPlugin(
            mem_size=mem_size, storage_policy=storage_policy
        )
        cl_strategy = Naive(
            model,
            SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001),
            CrossEntropyLoss(),
            train_mb_size=32,
            train_epochs=1,
            eval_mb_size=100,
            plugins=[replayPlugin],
        )

        n_seen_data = 0
        for step in benchmark.train_stream:
            n_seen_data += len(step.dataset)
            mem_fill = min(mem_size, n_seen_data)
            cl_strategy.train(step)
            lengths = []
            for d in replayPlugin.storage_policy.buffer_datasets:
                lengths.append(len(d))
            self.assertEqual(sum(lengths), mem_fill)  # Always fully filled

    def test_balancing(self):
        p1 = ExperienceBalancedBuffer(100, adaptive_size=True)
        p2 = ClassBalancedBuffer(100, adaptive_size=True)

        for policy in [p1, p2]:
            self.assert_balancing(policy)

    def assert_balancing(self, policy):
        benchmark = get_fast_benchmark(use_task_labels=True)
        replay = ReplayPlugin(mem_size=100, storage_policy=policy)
        model = SimpleMLP(num_classes=benchmark.n_classes)

        # CREATE THE STRATEGY INSTANCE (NAIVE)
        cl_strategy = Naive(
            model,
            SGD(model.parameters(), lr=0.001),
            CrossEntropyLoss(),
            train_mb_size=100,
            train_epochs=0,
            eval_mb_size=100,
            plugins=[replay],
            evaluator=None,
        )

        for exp in benchmark.train_stream:
            cl_strategy.train(exp)

            ext_mem = policy.buffer_groups
            ext_mem_data = policy.buffer_datasets
            print(list(ext_mem.keys()), [len(el) for el in ext_mem_data])

            # buffer size should equal self.mem_size if data is large enough
            len_tot = sum([len(el) for el in ext_mem_data])
            assert len_tot == policy.max_size


class ParametricBufferTest(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark = get_fast_benchmark(use_task_labels=True)

    def test_groupings(self):
        policies = [
            ParametricBuffer(max_size=10, groupby=None),
            ParametricBuffer(max_size=10, groupby="class"),
            ParametricBuffer(max_size=10, groupby="task"),
            ParametricBuffer(max_size=10, groupby="experience"),
        ]
        for p in policies:
            self._test_policy(p)

    def _test_policy(self, policy):
        for exp in self.benchmark.train_stream:
            policy.update(MagicMock(experience=exp))
            groups_lens = [
                (g_id, len(g_data.buffer))
                for g_id, g_data in policy.buffer_groups.items()
            ]
            print(groups_lens)
        print("DONE.")


class SelectionStrategyTest(unittest.TestCase):
    def test(self):
        # Given
        model = AbsModel()
        herding = HerdingSelectionStrategy(model, "features")
        closest_to_center = ClosestToCenterSelectionStrategy(model, "features")

        # When
        # Features are [[0], [4], [5]]
        # Center is [3]
        dataset = AvalancheTensorDataset(
            tensor([0, -4, 5]).float(),
            zeros(3),
            dataset_type=AvalancheDatasetType.CLASSIFICATION,
        )
        strategy = MagicMock(device="cpu", eval_mb_size=8)

        # Then

        # Herding:

        # 1. At first pass, we select the -4 (at index 1)
        #  because it is the closest ([4]) to the center in feature space
        # 2. At second pass, we select 0 (of index 0)
        #  because the center will be [2], closest to [3] than the center
        #  obtained if we were to select 5 ([4.5])
        # 3. Finally we select the last remaining exemplar
        self.assertSequenceEqual(
            [1, 0, 2], herding.make_sorted_indices(strategy, dataset)
        )
        # Closest to center

        # -4 (index 1) is the closest to the center in feature space.
        # Then 5 (index 2) is closest than 0 (index 0)
        self.assertSequenceEqual(
            [1, 2, 0], closest_to_center.make_sorted_indices(strategy, dataset)
        )


class AbsModel(Module):
    """Fake model, that simply compute the absolute value of the inputs"""

    def __init__(self):
        super().__init__()
        self.features = AbsLayer()
        self.classifier = Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class AbsLayer(Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.abs(x).reshape((-1, 1))


class FixedSelectionStrategy(ExemplarsSelectionStrategy):
    """This is a fake strategy used for testing the policy behavior"""

    def __init__(self, indices: List[int]):
        self.indices = indices

    def make_sorted_indices(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        return self.indices
