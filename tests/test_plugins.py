import unittest

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks import nc_scenario
from avalanche.benchmarks.datasets import MNIST
from avalanche.benchmarks.scenarios import DatasetPart
from avalanche.evaluation.eval_protocol import EvalProtocol
from avalanche.evaluation.metrics import ACC
from avalanche.extras import SimpleMLP
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import Naive
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop


class MockPlugin(StrategyPlugin):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.activated = [False for _ in range(22)]

    def before_training(self, strategy, **kwargs):
        self.activated[0] = True

    def adapt_train_dataset(self, strategy, **kwargs):
        self.activated[1] = True

    def before_training_epoch(self, strategy, **kwargs):
        self.activated[2] = True

    def before_training_iteration(self, strategy, **kwargs):
        self.activated[3] = True

    def before_forward(self, strategy, **kwargs):
        self.activated[4] = True

    def after_forward(self, strategy, **kwargs):
        self.activated[5] = True

    def before_backward(self, strategy, **kwargs):
        self.activated[6] = True

    def after_backward(self, strategy, **kwargs):
        self.activated[7] = True

    def after_training_iteration(self, strategy, **kwargs):
        self.activated[8] = True

    def before_update(self, strategy, **kwargs):
        self.activated[9] = True

    def after_update(self, strategy, **kwargs):
        self.activated[10] = True

    def after_training_epoch(self, strategy, **kwargs):
        self.activated[11] = True

    def after_training(self, strategy, **kwargs):
        self.activated[12] = True

    def before_test(self, strategy, **kwargs):
        self.activated[13] = True

    def adapt_test_dataset(self, strategy, **kwargs):
        self.activated[14] = True

    def before_test_step(self, strategy, **kwargs):
        self.activated[15] = True

    def after_test_step(self, strategy, **kwargs):
        self.activated[16] = True

    def after_test(self, strategy, **kwargs):
        self.activated[17] = True

    def before_test_iteration(self, strategy, **kwargs):
        self.activated[18] = True

    def before_test_forward(self, strategy, **kwargs):
        self.activated[19] = True

    def after_test_forward(self, strategy, **kwargs):
        self.activated[20] = True

    def after_test_iteration(self, strategy, **kwargs):
        self.activated[21] = True


class FlowTests(unittest.TestCase):
    def test_callback_reachability(self):
        model = SimpleMLP()
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        scenario = self.create_scenario()
        eval_protocol = EvalProtocol(
            metrics=[ACC(num_class=scenario.n_classes)])

        plug = MockPlugin()
        strategy = Naive(model, optimizer, criterion, eval_protocol,
            train_mb_size=100, train_epochs=1, test_mb_size=100,
            device='cpu', plugins=[plug]
        )
        strategy.train(scenario[0], num_workers=4)
        strategy.test(scenario[0], DatasetPart.CURRENT, num_workers=4)
        assert all(plug.activated)

    def create_scenario(self):
        train_transform = transforms.Compose([
            RandomCrop(28, padding=4),
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_train = MNIST('./data/mnist', train=True, download=True,
                            transform=train_transform)
        mnist_test = MNIST('./data/mnist', train=False, download=True,
                           transform=test_transform)
        scenario = nc_scenario(mnist_train, mnist_test, 5, task_labels=False,
                               shuffle=True, seed=1234)
        return scenario


if __name__ == '__main__':
    unittest.main()
