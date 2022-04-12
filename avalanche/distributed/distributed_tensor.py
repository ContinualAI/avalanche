from abc import ABC, abstractmethod
from typing import List

import torch
from torch import Tensor

from avalanche.distributed import DistributedHelper
from avalanche.distributed.distributed_value import SwitchableDistributedValue


class DistributedTensor(SwitchableDistributedValue[Tensor, Tensor], ABC):
    """
    A distributed Tensor wrapper.

    This abstract class is in charge of synchronizing Tensors across processes.

    Child classes must override `_merge_tensors` to define how those tensors
    should be merged.
    """
    def _synchronize_distributed_value(self) -> Tensor:
        return self._merge_tensors(
            DistributedHelper.gather_all(self.local_value))

    @abstractmethod
    def _merge_tensors(self, tensors: List[Tensor]) -> Tensor:
        """
        Merge all tensors into one.

        :param tensors: The list of tensors obtained from all processes, in the
            order defined by the rank.
        :return: The merged tensor.
        """
        pass


class ConcatDistributedTensor(DistributedTensor):
    """
    A distributed tensor obtained by concatenating tensors from all processes
    (in the order defined by the rank).

    This also correctly manages tensors with 0-length shapes (like losses).
    """
    def _merge_tensors(self, tensors: List[Tensor]) -> Tensor:
        # Manage tensors without shape (0-length shape)
        for i, t in enumerate(tensors):
            if len(t.shape) == 0:
                # Tensor with 0-length shape
                tensors[i] = torch.reshape(t, (1,))

        return torch.cat(tensors)


class DistributedMeanTensor(ConcatDistributedTensor):
    """
    A distributed 1-item tensor obtained by computing the mean of tensors
    from all processes.
    """
    def _merge_tensors(self, tensors: List[Tensor]) -> Tensor:
        concat_tensor = super()._merge_tensors(tensors)
        return torch.mean(concat_tensor)


__all__ = [
    'DistributedTensor',
    'ConcatDistributedTensor',
    'DistributedMeanTensor'
]
