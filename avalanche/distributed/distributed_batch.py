from abc import abstractmethod, ABC
from typing import TypeVar, List, Optional

import torch
from torch import Tensor

from avalanche.distributed import DistributedHelper
from avalanche.distributed.distributed_value import SwitchableDistributedValue

TupleT = TypeVar('TupleT', bound='Tuple')
OptTupleT = Optional[TupleT]
LocalT = TypeVar('LocalT')
DistributedT = TypeVar('DistributedT')


class DistributedObject(SwitchableDistributedValue[LocalT, DistributedT], ABC):
    """
    An intermediate abstract class in charge of synchronizing objects.

    The merge procedure must be implemented in child classes.
    """
    def _synchronize_distributed_value(self) -> DistributedT:
        objects = self._synchronize_objects()
        return self._merge_objects(objects)

    def _synchronize_objects(self) -> List[LocalT]:
        return DistributedHelper.gather_all_objects(
            self._local_value
        )

    @abstractmethod
    def _merge_objects(self, objects: List[LocalT]) -> DistributedT:
        pass


class DistributedBatch(DistributedObject[LocalT, LocalT], ABC):
    """
    An intermediate abstract class in charge of synchronizing data batches.

    This class can handle batches as either tuples of elements (as usual) or
    even single values.

    The merge procedure of single elements must be implemented in child classes.

    NOTE: In the future, this class may be replaced with a version in which only
    the accessed tuple elements are synchronized, instead of the whole batch.
    The current design, in which child classes only have to implement
    `_merge_single_values`, allows for this change to happen without affecting
    child classes.
    """

    def __init__(self, name: str, initial_local_value: LocalT):
        super(DistributedBatch, self).__init__(
            name, initial_local_value
        )
        self._value_is_tuple = False

    def _synchronize_distributed_value(self) -> LocalT:
        if self._local_value is None:
            return None
        else:
            return super()._synchronize_distributed_value()

    def _set_local_value(self, new_local_value):
        self._value_is_tuple = isinstance(new_local_value, (tuple, list))
        super(DistributedBatch, self)._set_local_value(new_local_value)

    def _merge_objects(self, objects: List[LocalT]) -> LocalT:
        if self._value_is_tuple:
            return self._merge_tuples(objects)
        else:
            return self._merge_single_values(objects)

    def _merge_tuples(self, tuples: List[LocalT]):
        merged_elements = []
        n_elements = len(self._local_value)
        for element_idx in range(n_elements):
            to_merge_elements = []
            for tp in tuples:
                to_merge_elements.append(tp[element_idx])

            merged_elements.append(
                self._merge_single_values(to_merge_elements)
            )

        return tuple(merged_elements)

    @abstractmethod
    def _merge_single_values(self, values: List):
        pass


class ClassificationBatch(DistributedBatch[LocalT]):
    """
    An implementation of :class:`DistributedBatch` that assumes that all values
    are Tensors.
    """
    def _merge_single_values(self, values: List[Tensor]):
        return torch.cat(values)


__all__ = [
    'DistributedObject',
    'DistributedBatch',
    'ClassificationBatch'
]
