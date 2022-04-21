from torch.nn import Module

from avalanche.distributed import DistributedModel


class DistributedModelStrategySupport:

    def __init__(self):
        super().__init__()
        self._model = DistributedModel()

    @property
    def model(self) -> Module:
        """ PyTorch model. """
        # This will return the local model if training locally
        return self._model.value

    @model.setter
    def model(self, value):
        """ Sets the PyTorch model. """
        self._model.value = value

    @property
    def local_model(self):
        return self._model.local_model

    @local_model.setter
    def local_model(self, value):
        self._model.local_model = value

    @property
    def distributed_model(self):
        return self._model.distributed_model

    @distributed_model.setter
    def distributed_model(self, value):
        self._model.distributed_model = value

    def use_local_model(self, *args, **kwargs):
        return self._model.use_local_value(*args, **kwargs)


__all__ = [
    'DistributedModelStrategySupport'
]
