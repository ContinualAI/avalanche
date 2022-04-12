################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1/12/2021                                                              #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from typing import Optional, Union, Tuple

from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from typing_extensions import Type

from avalanche.distributed import OptionalDistributedValue
from avalanche.distributed.distributed_value import DistributedT


class DistributedModel(OptionalDistributedValue[Optional[Module]]):
    """
    Contains the model used in the :class:`BaseTemplate` strategy template.

    Instances of this class can also carry the distributed (that is, wrapped
    in a PyTorch `DistributedDataParallel`) version of a local model. If no
    distributed model is set, then the model returned by the
    `distributed_model` field will be the local one.

    By setting the `distributed_model` field, the model stored in the
    `local_model` field will be discarded (from that moment, retrieving the
    `local_model` will be the same as obtaining the `distributed_model.module`
    field). Setting the `local_model` will discard the current
    `distributed_model`.

    Beware that the setter of this class behaves a bit differently
    from superclasses. When setting the `value`, the class of the new value
    us checked against a list of distributed model classes (by default,
    only :class:`DistributedDataParallel` is considered). If the model
    is an instance of these classes, then the distributed value is set
    instead of the local value.
    """

    def __init__(
            self,
            initial_model: Module = None,
            distributed_model_class: Union[Type, Tuple[Type]] =
            DistributedDataParallel):
        """
        Creates a `ModelInstance`.

        :param initial_model: The initial model to use. Defaults to None.
        :param distributed_model_class: The type(s) of the distributed model.
            Defaults to `DistributedDataParallel`.
        """
        super().__init__('model', initial_local_value=initial_model)
        self.distributed_model_class = distributed_model_class

    @OptionalDistributedValue.value.setter
    def value(self, new_value: Module):
        """
        Sets the local or distributed model, depending on if the model is a
        subclass of DistributedDataParallel.

        This will discard the current distributed value.
        """

        if isinstance(new_value, self.distributed_model_class):
            self.distributed_value = new_value
        else:
            self.local_value = new_value

    @OptionalDistributedValue.local_value.getter
    def local_value(self) -> Module:
        if self._distributed_value is not None:
            return self._distributed_value.module
        return self._local_value

    @OptionalDistributedValue.distributed_value.setter
    def distributed_value(self, new_distributed_value: Module):
        if new_distributed_value is None:
            self.reset_distributed_value()
        else:
            self._distributed_value = new_distributed_value
            self._distributed_value_set = True

            # Prevent alignment and memory issues.
            # The local model will be retrieved from the distributed model.
            self._local_value = None

    def reset_distributed_value(self):
        if self._distributed_value_set:
            if self._distributed_value is not None:
                # Unwrap the DistributedDataParallel to obtain the local value.
                self._local_value = self._distributed_value.module
            self._distributed_value = None
            self._distributed_value_set = False

    def reset_distributed_model(self):
        """
        Discards the distributed model.

        If the distributed model was not set, nothing happens.
        """
        return self.reset_distributed_value()

    def _synchronize_distributed_value(self) -> DistributedT:
        raise RuntimeError(
            'The distributed model needs to be wrapped and set by using the '
            f'following class(es): {self.distributed_model_class}')

    # BEGIN ALIASES for "(local|distributed)value"
    @property
    def model(self):
        """
        The current model.
        """
        return self.value

    @model.setter
    def model(self, new_model: Module):
        """
        Sets the current model.
        """
        self.value = new_model

    @property
    def local_model(self) -> Module:
        """
        The current (local) model.

        If a `distributed_model` was set, then the value of the
        `distributed_model.module` field will be returned.
        """
        return self.local_value

    @local_model.setter
    def local_model(self, new_local_value):
        """
        Sets the local model.

        This will discard the current distributed model.
        """
        self.local_value = new_local_value

    @property
    def distributed_model(self):
        """
        The current (distributed) model.

        If not set (not running a distributed training, or if the wrapped
        model has not been created yet), this is the same as `local_model`.
        """
        return self.distributed_value

    @distributed_model.setter
    def distributed_model(self, new_distributed_value):
        """
        Sets the model wrapped by PyTorch `DistributedDataParallel`.

        Setting this field will release the reference to the current local
        model. In that case, the `local_model` field will return
        `distributed_model.module` instead.
        """
        self.distributed_value = new_distributed_value
    # END ALIASES for "(local|distributed)value"


__all__ = [
    'DistributedModel'
]
