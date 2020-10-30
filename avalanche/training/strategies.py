from typing import Optional, Sequence

from torch.nn import Module
from torch.optim import Optimizer

from avalanche.evaluation import EvalProtocol
from avalanche.training.strategy_flow import StrategyFlow
from avalanche.training.plugins import StrategyPlugin


class Naive(StrategyFlow):
    """
    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(self, model: Module, criterion,
                 optimizer: Optimizer,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 evaluation_protocol: Optional[EvalProtocol] = None,
                 plugins: Optional[Sequence[StrategyPlugin]] = None):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param evaluation_protocol: The evaluation protocol. Defaults to None.
        :param plugins: Plugins to be added. Defaults to None.
        """
        # TODO: ADD multi-head plugin
        super().__init__(
            model, criterion, optimizer, train_mb_size=train_mb_size,
            train_epochs=train_epochs, test_mb_size=test_mb_size,
            evaluation_protocol=evaluation_protocol, device=device,
            plugins=plugins)
