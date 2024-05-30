import unittest

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from avalanche.core import Agent
from avalanche.models import SimpleMLP, as_multitask, IncrementalClassifier, MTSimpleMLP
from avalanche.models.dynamic_modules import avalanche_model_adaptation
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
        agent.opt = DynamicOptimizer(opt, agent.model, verbose=False)

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

    def init_scenario(self, multi_task=False):
        if multi_task:
            model = MTSimpleMLP(input_size=6, hidden_size=10)
        else:
            model = SimpleMLP(input_size=6, hidden_size=10)
            model.classifier = IncrementalClassifier(10, 1)
        criterion = CrossEntropyLoss()
        benchmark = get_fast_benchmark(use_task_labels=multi_task)
        return model, criterion, benchmark

    def _is_param_in_optimizer(self, param, optimizer):
        for group in optimizer.param_groups:
            for curr_p in group["params"]:
                if hash(curr_p) == hash(param):
                    return True
        return False

    def _is_param_in_optimizer_group(self, param, optimizer):
        for group_idx, group in enumerate(optimizer.param_groups):
            for curr_p in group["params"]:
                if hash(curr_p) == hash(param):
                    return group_idx
        return None

    def test_optimizer_groups_clf_til(self):
        """
        Tests the automatic assignation of new
        MultiHead parameters to the optimizer
        """
        model, criterion, benchmark = self.init_scenario(multi_task=True)

        g1 = []
        g2 = []
        for n, p in model.named_parameters():
            if "classifier" in n:
                g1.append(p)
            else:
                g2.append(p)

        agent = Agent()
        agent.model = model
        optimizer = SGD([{"params": g1, "lr": 0.1}, {"params": g2, "lr": 0.05}])
        agent.optimizer = DynamicOptimizer(optimizer, model=model, verbose=False)

        for experience in benchmark.train_stream:
            avalanche_model_adaptation(model, experience)
            agent.optimizer.pre_adapt(agent, experience)

            for n, p in model.named_parameters():
                assert self._is_param_in_optimizer(p, agent.optimizer.optim)
                if "classifier" in n:
                    self.assertEqual(
                        self._is_param_in_optimizer_group(p, agent.optimizer.optim), 0
                    )
                else:
                    self.assertEqual(
                        self._is_param_in_optimizer_group(p, agent.optimizer.optim), 1
                    )

    def test_optimizer_groups_clf_cil(self):
        """
        Tests the automatic assignation of new
        IncrementalClassifier parameters to the optimizer
        """
        model, criterion, benchmark = self.init_scenario(multi_task=False)

        g1 = []
        g2 = []
        for n, p in model.named_parameters():
            if "classifier" in n:
                g1.append(p)
            else:
                g2.append(p)

        agent = Agent()
        agent.model = model
        optimizer = SGD([{"params": g1, "lr": 0.1}, {"params": g2, "lr": 0.05}])
        agent.optimizer = DynamicOptimizer(optimizer, model)

        for experience in benchmark.train_stream:
            avalanche_model_adaptation(model, experience)
            agent.optimizer.pre_adapt(agent, experience)

            for n, p in model.named_parameters():
                assert self._is_param_in_optimizer(p, agent.optimizer.optim)
                if "classifier" in n:
                    self.assertEqual(
                        self._is_param_in_optimizer_group(p, agent.optimizer.optim), 0
                    )
                else:
                    self.assertEqual(
                        self._is_param_in_optimizer_group(p, agent.optimizer.optim), 1
                    )
