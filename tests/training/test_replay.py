import unittest
from typing import List, Dict
from unittest.mock import MagicMock

import torch
from torch import tensor, Tensor, zeros
from torch.nn import CrossEntropyLoss, Module, Identity
from torch.optim import SGD
from torch.utils.data import TensorDataset

from avalanche.benchmarks.utils import AvalancheDataset, AvalancheDatasetType
from avalanche.models import SimpleMLP
from avalanche.training.plugins import ExperienceBalancedStoragePolicy, \
    ClassBalancedStoragePolicy, ReplayPlugin
from avalanche.training.plugins.replay import ClassExemplarsSelectionStrategy, \
    HerdingSelectionStrategy, ClosestToCenterSelectionStrategy
from avalanche.training.strategies import Naive, BaseStrategy
from tests.unit_tests_utils import get_fast_scenario


class ReplayTest(unittest.TestCase):
    def test_replay_balanced_memory(self):
        mem_size = 25
        policies = [None,
                    ExperienceBalancedStoragePolicy({}, mem_size=mem_size),
                    ClassBalancedStoragePolicy({}, mem_size=mem_size)]
        for policy in policies:
            self._test_replay_balanced_memory(policy, mem_size)

    def _test_replay_balanced_memory(self, storage_policy, mem_size):
        scenario = get_fast_scenario(use_task_labels=True)
        model = SimpleMLP(input_size=6, hidden_size=10)
        replayPlugin = ReplayPlugin(mem_size=mem_size,
                                    storage_policy=storage_policy)
        cl_strategy = Naive(
            model,
            SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001),
            CrossEntropyLoss(), train_mb_size=32, train_epochs=1,
            eval_mb_size=100, plugins=[replayPlugin]
        )

        n_seen_data = 0
        for step in scenario.train_stream:
            n_seen_data += len(step.dataset)
            mem_fill = min(mem_size, n_seen_data)
            cl_strategy.train(step)
            ext_mem = replayPlugin.ext_mem
            lengths = []
            for task_id in ext_mem.keys():
                lengths.append(len(ext_mem[task_id]))
            self.assertEqual(sum(lengths), mem_fill)  # Always fully filled

    def test_balancing(self):
        p1 = ExperienceBalancedStoragePolicy({}, 100, adaptive_size=True)
        p2 = ClassBalancedStoragePolicy({}, 100, adaptive_size=True)

        for policy in [p1, p2]:
            self.assert_balancing(policy)

    def assert_balancing(self, policy):
        ext_mem = policy.ext_mem
        scenario = get_fast_scenario(use_task_labels=True)
        replay = ReplayPlugin(mem_size=100, storage_policy=policy)
        model = SimpleMLP(num_classes=scenario.n_classes)

        # CREATE THE STRATEGY INSTANCE (NAIVE)
        cl_strategy = Naive(model,
                            SGD(model.parameters(), lr=0.001),
                            CrossEntropyLoss(), train_mb_size=100,
                            train_epochs=0,
                            eval_mb_size=100, plugins=[replay], evaluator=None)

        for exp in scenario.train_stream:
            cl_strategy.train(exp)
            print(list(ext_mem.keys()), [len(el) for el in ext_mem.values()])

            # buffer size should equal self.mem_size if data is large enough
            len_tot = sum([len(el) for el in ext_mem.values()])
            assert len_tot == policy.mem_size


class ClassBalancePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = {}
        self.policy = ClassBalancedStoragePolicy(self.memory, mem_size=4)

    def test_store_alone_with_enough_memory(self):
        order = [2, 0, 1, 3]

        self.observe_exemplars({0: list(range(4))}, selection_order=order)

        self.assert_memory_equal({0: order})

    def test_store_alone_without_enough_memory(self):
        order = [6, 4, 2, 0, 1, 3, 5]

        self.observe_exemplars({0: list(range(7))}, selection_order=order)

        self.assert_memory_equal({0: order[:4]})

    def test_store_multiple_with_enough_memory(self):
        self.observe_exemplars({0: [0, 1], 1: [10, 11]}, selection_order=[1, 0])

        self.assert_memory_equal({0: [1, 0], 1: [11, 10]})

    def test_store_multiple_without_enough_memory(self):
        self.observe_exemplars({0: [0, 1, 2], 1: [10, 11, 12]},
                               selection_order=[2, 0, 1])

        self.assert_memory_equal({0: [2, 0], 1: [12, 10]})

    def test_sequence(self):
        # 1st observation
        self.observe_exemplars({0: [0, 1], 1: [10, 11]}, selection_order=[1, 0])
        self.assert_memory_equal({0: [1, 0], 1: [11, 10]})

        # 2nd observation
        self.observe_exemplars({2: [20, 21, 22], 3: [30, 31, 32]},
                               selection_order=[2, 1, 0])
        self.assert_memory_equal({0: [1], 1: [11], 2: [22], 3: [32]})

    def observe_exemplars(self, class2exemplars: Dict[int, List[int]],
                          selection_order: List[int]):
        self.policy.selection_strategy = FixedSelectionStrategy(selection_order)
        x = tensor(
            [i for exemplars in class2exemplars.values() for i in exemplars])
        y = tensor(
            [class_id for class_id, exemplars in class2exemplars.items() for _
             in exemplars]).long()
        dataset = AvalancheDataset(
            TensorDataset(x, y),
            dataset_type=AvalancheDatasetType.CLASSIFICATION)

        self.policy(MagicMock(experience=MagicMock(dataset=dataset)))

    def assert_memory_equal(self, class2exemplars: Dict[int, List[int]]):
        self.assertEqual(class2exemplars,
                         {class_id: [x.tolist() for x, *_ in memory] for
                          class_id, memory in self.memory.items()})


class SelectionStrategyTest(unittest.TestCase):
    def test(self):
        # Given
        model = AbsModel()
        herding = HerdingSelectionStrategy(model, "features")
        closest_to_center = ClosestToCenterSelectionStrategy(model, "features")

        # When
        # Features are [[0], [4], [5]]
        # Center is [3]
        dataset = AvalancheDataset(
            TensorDataset(tensor([0, -4, 5]).float(), zeros(3)),
            dataset_type=AvalancheDatasetType.CLASSIFICATION)
        strategy = MagicMock(device="cpu", eval_mb_size=8)

        # Then

        # Herding:

        # 1. At first pass, we select the -4 (at index 1)
        #  because it is the closest ([4]) to the center in feature space
        # 2. At second pass, we select 0 (of index 0)
        #  because the center will be [2], closest to [3] than the center
        #  obtained if we were to select 5 ([4.5])
        # 3. Finally we select the last remaining exemplar
        self.assertSequenceEqual([1, 0, 2],
                                 herding.make_sorted_indices(strategy, dataset))
        # Closest to center

        # -4 (index 1) is the closest to the center in feature space.
        # Then 5 (index 2) is closest than 0 (index 0)
        self.assertSequenceEqual([1, 2, 0],
                                 closest_to_center.make_sorted_indices(strategy,
                                                                       dataset))


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


class FixedSelectionStrategy(ClassExemplarsSelectionStrategy):
    """This is a fake strategy used for testing the policy behavior"""

    def __init__(self, indices: List[int]):
        self.indices = indices

    def make_sorted_indices(self, strategy: "BaseStrategy",
                            data: AvalancheDataset) -> List[int]:
        return self.indices
