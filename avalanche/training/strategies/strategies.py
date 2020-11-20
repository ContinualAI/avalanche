from typing import Optional, Sequence, List

from torch.nn import Module
from torch.optim import Optimizer

from torch.utils.data import ConcatDataset

from avalanche.evaluation import EvalProtocol
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.plugins import StrategyPlugin, \
    CWRStarPlugin, ReplayPlugin, GDumbPlugin


class Naive(BaseStrategy):
    """
    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 evaluation_protocol: Optional[EvalProtocol] = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[Sequence[StrategyPlugin]] = None):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param evaluation_protocol: The evaluation plugin.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        """
        super().__init__(
            model, optimizer, criterion, evaluation_protocol,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins)


class CWRStar(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 second_last_layer_name, num_classes=50,
                 evaluation_protocol: Optional[EvalProtocol] = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 ):
        """ CWR* Strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param second_last_layer_name: name of the second to last layer
                (layer just before the classifier).
        :param num_classes: total number of classes.
        :param evaluation_protocol: The evaluation plugin.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        """
        cwsp = CWRStarPlugin(model, second_last_layer_name, num_classes)
        if plugins is None:
            plugins = [cwsp]
        else:
            plugins.append(cwsp)
        super().__init__(
            model, optimizer, criterion, evaluation_protocol,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins)


class Replay(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 mem_size: int = 200,
                 evaluation_protocol: Optional[EvalProtocol] = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 ):
        """ Experience replay strategy. See ReplayPlugin for more details.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: replay buffer size.
        :param evaluation_protocol: The evaluation plugin.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        """
        rp = ReplayPlugin(mem_size)
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)
        super().__init__(
            model, optimizer, criterion, evaluation_protocol,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins)


class GDumb(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 mem_size: int = 200,
                 evaluation_protocol: Optional[EvalProtocol] = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 ):
        """ GDumb strategy. See GDumbPlugin for more details.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: replay buffer size.
        :param evaluation_protocol: The evaluation plugin.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        """

        gdumb = GDumbPlugin(mem_size)
        if plugins is None:
            plugins = [gdumb]
        else:
            plugins.append(gdumb)

        super().__init__(
            model, optimizer, criterion, evaluation_protocol,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins)


class Cumulative(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 evaluation_protocol: Optional[EvalProtocol] = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 ):
        """ Cumulative strategy. At each step,
            train model with data from all previous steps and current step.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param evaluation_protocol: The evaluation plugin.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        """

        super().__init__(
            model, optimizer, criterion, evaluation_protocol,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins)

        self.dataset = None  # cumulative dataset

    def adapt_train_dataset(self, **kwargs):

        super().adapt_train_dataset(**kwargs)

        if self.dataset is None:
            self.dataset = self.current_data
        else:
            self.dataset = ConcatDataset([self.dataset, self.current_data])
            self.current_data = self.dataset


__all__ = ['Naive', 'CWRStar', 'Replay', 'GDumb', 'Cumulative']
