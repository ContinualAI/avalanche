import unittest
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.models import SimpleMLP
from avalanche.training.plugins import ExperienceBalancedStoragePolicy, \
    ClassBalancedStoragePolicy, ReplayPlugin
from avalanche.training.strategies import Naive

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
