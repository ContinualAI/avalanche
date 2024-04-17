import unittest

from torch.optim import SGD
from torch.utils.data import DataLoader

from avalanche.core import Agent
from avalanche.models import SimpleMLP, as_multitask
from avalanche.models.dynamic_optimizers import DynamicOptimizer
from avalanche.training import MaskedCrossEntropy
from tests.unit_tests_utils import get_fast_benchmark


class TestDynamicOptimizers(unittest.TestCase):
    def test_dynamic_optimizer(self):
        bm = get_fast_benchmark(use_task_labels=True)
        agent = Agent()
        agent.loss = MaskedCrossEntropy()
        agent.model = as_multitask(SimpleMLP(input_size=6), "classifier")
        opt = SGD(agent.model.parameters(), lr=0.001)
        agent.opt = DynamicOptimizer(opt)

        for exp in bm.train_stream:
            agent.model.train()
            data = exp.dataset.train()
            agent.pre_adapt(exp)
            for ep in range(1):
                dl = DataLoader(data, batch_size=32, shuffle=True)
                for x, y, t in dl:
                    agent.opt.zero_grad()
                    yp = agent.model(x, t)
                    l = agent.loss(yp, y)
                    l.backward()
                    agent.opt.step()
            agent.post_adapt(exp)
