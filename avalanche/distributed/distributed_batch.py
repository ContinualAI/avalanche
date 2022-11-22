from abc import abstractmethod, ABC
from typing import TypeVar, List, Optional, Callable, Any, Iterable

import torch
from torch import Tensor

from avalanche.distributed import DistributedHelper
from avalanche.distributed.distributed_value import SwitchableDistributedValue

LocalT = TypeVar('LocalT')
DistributedT = TypeVar('DistributedT')


class DistributedObject(SwitchableDistributedValue[LocalT, DistributedT], ABC):
    """
    An intermediate abstract class in charge of synchronizing objects.

    The merge procedure must be implemented in child classes.
    """
    def _synchronize(self) -> DistributedT:
        objects = self._synchronize_objects()
        return self._merge_objects(objects)

    def _synchronize_objects(self) -> List[LocalT]:
        return DistributedHelper.gather_all_objects(
            self._local_value
        )

    @abstractmethod
    def _merge_objects(self, objects: List[LocalT]) -> DistributedT:
        pass


class OnlyTupleSynchronizationSupported(BaseException):
    pass


class DistributedBatch(DistributedObject[LocalT, LocalT], ABC):
    """
    An intermediate abstract class in charge of synchronizing data batches.

    This class can handle batches as either tuples of elements (as usual) or
    even single values.

    The merge procedure of tuples and single elements must be implemented in
    child classes. By default, the tuples will be merged value by value.

    NOTE: In the future, this class may be replaced with a version in which only
    the accessed tuple elements are synchronized, instead of the whole batch.
    The current design, in which child classes have to implement
    `_merge_single_values`, allows for this change to happen without affecting
    child classes.
    """

    def __init__(self, name: str, initial_local_value: LocalT):
        super().__init__(name, initial_local_value)
        self._value_is_tuple = False

    def _synchronize(self) -> LocalT:
        if self._local_value is None:
            return None
        else:
            return super()._synchronize()

    def _set_local(self, new_local_value):
        self._value_is_tuple = isinstance(new_local_value, (tuple, list))
        super()._set_local(new_local_value)

    def _merge_objects(self, objects: List[LocalT]) -> LocalT:
        if not self._value_is_tuple:
            try:
                return self._merge_single_values(objects, 0)
            except OnlyTupleSynchronizationSupported:
                pass

        return self._merge_tuples(objects)

    def _merge_tuples(self, tuples: List[LocalT]):
        try:
            merged_elements = []
            # Note: _local_value is usually a tuple (mb_x, mb_y, ...)
            # which means that n_elements is usually == 2 or 3

            n_elements = len(self._local_value)
            for element_idx in range(n_elements):
                to_merge_elements = []
                for tp in tuples:
                    to_merge_elements.append(tp[element_idx])

                merged_elements.append(
                    self._merge_single_values(to_merge_elements, element_idx)
                )

            return tuple(merged_elements)
        except OnlyTupleSynchronizationSupported:
            raise RuntimeError('[DistributedBatch] No proper collate function set.')

    @abstractmethod
    def _merge_single_values(self, values: List, value_index: int):
        pass


class CollateDistributedBatch(DistributedBatch[LocalT]):
    """
    An implementation of :class:`DistributedBatch` in which the
    `_merge_tuples` mechanism is given as a callable function.

    This assumes that local batches are locally pre-collated and
    will thus unroll them before calling the given function.
    """

    def __init__(self, name: str, initial_local_value: LocalT,
                 tuples_collate_fn: Optional[Callable[[List], LocalT]],
                 single_values_collate_fn: Optional[Callable[[Any, int], Any]]):
        super().__init__(name, initial_local_value)
        self.tuples_collate_fn = tuples_collate_fn
        self.single_values_collate_fn = single_values_collate_fn

    def _unroll_minibatch(self, tuples: List[LocalT]) -> List[LocalT]:
        unrolled_elements = []
        for local_tuple in tuples:
            n_elements = len(local_tuple)
            mb_size = len(local_tuple[0])

            for mb_element_idx in range(mb_size):
                mb_element = []
                for tuple_element_idx in range(n_elements):
                    mb_element.append(local_tuple[tuple_element_idx][mb_element_idx])
                unrolled_elements.append(tuple(mb_element))
        return unrolled_elements

    def _unroll_value(self, collated_values: List[Iterable[Any]]) -> Any:
        unrolled_values = []
        for val_batch in collated_values:
            unrolled_values.extend(val_batch)

        return unrolled_values

    def _merge_tuples(self, tuples: List[LocalT]):
        if self.tuples_collate_fn is not None:
            unrolled_elements = self._unroll_minibatch(tuples)

            return self.tuples_collate_fn(unrolled_elements)

        return super()._merge_tuples(tuples)

    def _merge_single_values(self, values: List, value_index: int):
        if self.single_values_collate_fn is None:
            raise OnlyTupleSynchronizationSupported()

        unrolled_elements = self._unroll_value(values)
        return self.single_values_collate_fn(unrolled_elements, value_index)


def make_classification_distributed_batch(name: str) -> \
        CollateDistributedBatch[Optional[Tensor]]:
    """
    Return a :class:`CollateDistributedBatch` that assumes that all values
    are Tensors. Values are obtained by concatenating these tensors.
    """
    return CollateDistributedBatch(
        name, None, None, lambda x, y: torch.stack(x)
    )


__all__ = [
    'DistributedObject',
    'DistributedBatch',
    'CollateDistributedBatch',
    'make_classification_distributed_batch'
]
