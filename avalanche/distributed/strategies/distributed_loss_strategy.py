from torch import Tensor

from avalanche.distributed import DistributedLoss
from avalanche.distributed.strategies import DistributedStrategySupport


class DistributedLossStrategySupport(DistributedStrategySupport):

    def __init__(self):
        super().__init__()
        self._loss = DistributedLoss()
        self._use_local_contexts.append(self.use_local_loss)

    @property
    def loss(self) -> Tensor:
        """ The loss tensor. """
        return self._loss.value

    @loss.setter
    def loss(self, value):
        """ Sets the loss. """
        self._loss.value = value

    @property
    def local_loss(self):
        return self._loss.local_value

    @local_loss.setter
    def local_loss(self, value):
        self._loss.local_value = value

    @property
    def distributed_loss(self):
        return self._loss.distributed_value

    @distributed_loss.setter
    def distributed_loss(self, value):
        self._loss.distributed_value = value

    def reset_distributed_loss(self):
        """ Resets the distributed value of the loss. """
        self._loss.reset_distributed_value()

    def use_local_loss(self, *args, **kwargs):
        return self._loss.use_local_value(*args, **kwargs)


__all__ = [
    'DistributedLossStrategySupport'
]
