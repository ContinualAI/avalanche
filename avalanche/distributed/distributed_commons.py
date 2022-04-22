import torch

from avalanche.distributed.distributed_tensor import DistributedMeanTensor


class DistributedLoss(DistributedMeanTensor):
    """
    A distributed value in charge of obtaining the mean loss.

    The mean loss is computed as the mean of losses from all processes, without
    weighting using the mini batch sizes in each process.

    This is current mostly an alias for :class:`DistributedMeanTensor`. However,
    in the future this class may be extended to add loss-specific features.
    """
    def __init__(self, name: str = 'loss'):
        super(DistributedLoss, self).__init__(name, torch.zeros((1,)))

    def _merge_tensors(self, tensors):
        return super(DistributedLoss, self)._merge_tensors(tensors)


__all__ = [
    'DistributedLoss'
]
