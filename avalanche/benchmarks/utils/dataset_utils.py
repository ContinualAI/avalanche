################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import bisect

from torch.utils.data import Subset, ConcatDataset

from .dataset_definitions import (
    IDatasetWithTargets,
    ISupportedClassificationDataset,
)

try:
    from typing import (
        Protocol,
        Sequence,
        List,
        Any,
        Iterable,
        Union,
        Optional,
        SupportsInt,
        TypeVar,
        Tuple,
        Callable,
        Generic,
    )
except ImportError:
    from typing import (
        Sequence,
        List,
        Any,
        Iterable,
        Union,
        Optional,
        SupportsInt,
        TypeVar,
        Tuple,
        Callable,
        Generic,
    )
    from typing_extensions import Protocol

T_co = TypeVar("T_co", covariant=True)
TTargetType = TypeVar("TTargetType")


class SubSequence(Sequence[TTargetType]):
    """
    A utility class used to define a lazily evaluated sub-sequence.
    """

    def __init__(
        self,
        targets: Sequence[TTargetType],
        *,
        indices: Union[Sequence[int], None] = None,
        converter: Optional[Callable[[Any], TTargetType]] = None
    ):
        self._targets = targets
        self._indices = indices
        self.converter = converter

    def __len__(self):
        if self._indices is None:
            return len(self._targets)
        return len(self._indices)

    def __getitem__(self, item_idx) -> TTargetType:
        if self._indices is not None:
            subset_idx = self._indices[item_idx]
        else:
            subset_idx = item_idx

        element = self._targets[subset_idx]

        if self.converter is not None:
            return self.converter(element)

        return element

    def __str__(self):
        return (
            "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"
        )


class LazyClassMapping(SubSequence[int]):
    """
    This class is used when in need of lazy populating a targets field whose
    elements need to be filtered out (when subsetting, see
    :class:`torch.utils.data.Subset`) and/or transformed (remapped). This will
    allow for a more efficient memory usage as the conversion is done on the fly
    instead of actually allocating a new targets list.

    This class should be used only when mapping int targets (classification).
    """

    def __init__(
        self,
        targets: Sequence[SupportsInt],
        indices: Union[Sequence[int], None],
        mapping: Optional[Sequence[int]] = None,
    ):
        super().__init__(targets, indices=indices, converter=int)
        self._mapping = mapping

    def __getitem__(self, item_idx) -> int:
        target_value = super().__getitem__(item_idx)

        if self._mapping is not None:
            return self._mapping[target_value]

        return target_value


class LazyConcatTargets(Sequence[TTargetType]):
    """
    Defines a lazy targets concatenation.

    This class is used when in need of lazy populating a targets created
    as the concatenation of the targets field of multiple datasets.
    This will allow for a more efficient memory usage as the concatenation is
    done on the fly instead of actually allocating a new targets list.
    """

    def __init__(
        self,
        targets_list: Sequence[Sequence[TTargetType]],
        converter: Optional[Callable[[Any], TTargetType]] = None,
    ):
        self._targets_list = targets_list
        self._targets_lengths = [len(targets) for targets in targets_list]
        self._overall_length = sum(self._targets_lengths)
        self._targets_cumulative_lengths = ConcatDataset.cumsum(targets_list)
        self.converter = converter

    def __len__(self):
        return self._overall_length

    def __getitem__(self, item_idx) -> TTargetType:
        targets_idx, internal_idx = find_list_from_index(
            item_idx,
            self._targets_lengths,
            self._overall_length,
            self._targets_cumulative_lengths,
        )

        target = self._targets_list[targets_idx][internal_idx]

        if self.converter is None:
            return target
        return self.converter(target)

    def __iter__(self):
        if self.converter is None:
            for x in self._targets_list:
                for y in x:
                    yield y
        else:
            for x in self._targets_list:
                for y in x:
                    yield self.converter(y)

    def __str__(self):
        return (
            "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"
        )


class LazyConcatIntTargets(LazyConcatTargets[int]):
    """
    Defines a lazy targets concatenation.

    This class is used when in need of lazy populating a targets created
    as the concatenation of the targets field of multiple datasets.
    This will allow for a more efficient memory usage as the concatenation is
    done on the fly instead of actually allocating a new targets list.

    Elements returned by `__getitem__` will be int values.
    """

    def __init__(self, targets_list: Sequence[Sequence[SupportsInt]]):
        super().__init__(targets_list, converter=int)


class ConstantSequence(Sequence[int]):
    """
    Defines a constant sequence given an int value and the length.
    """

    def __init__(self, constant_value: int, size: int):
        self._constant_value = constant_value
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, item_idx) -> int:
        if item_idx >= len(self):
            raise IndexError()

        return self._constant_value

    def __str__(self):
        return (
            "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"
        )


class SubsetWithTargets(Generic[T_co, TTargetType], Subset[T_co]):
    """
    A Dataset that behaves like a PyTorch :class:`torch.utils.data.Subset`.
    However, this dataset also supports the targets field.
    """

    def __init__(
        self,
        dataset: IDatasetWithTargets[T_co, TTargetType],
        indices: Union[Sequence[int], None],
    ):
        if indices is None:
            indices = range(len(dataset))
        super().__init__(dataset, indices)
        self.targets: Sequence[TTargetType] = SubSequence(
            dataset.targets, indices=indices
        )

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)


class ClassificationSubset(SubsetWithTargets[T_co, int]):
    """
    A Dataset that behaves like a PyTorch :class:`torch.utils.data.Subset`.
    However, this dataset also supports the targets field and class mapping.

    Targets will be converted to int.
    """

    def __init__(
        self,
        dataset: ISupportedClassificationDataset[T_co],
        indices: Union[Sequence[int], None],
        class_mapping: Optional[Sequence[int]] = None,
    ):
        super().__init__(dataset, indices)
        self.class_mapping = class_mapping
        self.targets = LazyClassMapping(
            dataset.targets, indices, mapping=class_mapping
        )

    def __getitem__(self, idx):
        result = super().__getitem__(idx)

        if self.class_mapping is not None:
            return make_tuple(
                (result[0], self.class_mapping[result[1]], *result[2:]), result
            )

        return result


class SequenceDataset(IDatasetWithTargets[T_co, TTargetType]):
    """
    A Dataset that wraps existing ndarrays, Tensors, lists... to provide
    basic Dataset functionalities. Very similar to TensorDataset.
    """

    def __init__(
        self,
        *sequences: Sequence,
        targets: Union[int, Sequence[TTargetType]] = 1
    ):
        """
        Creates a ``SequenceDataset`` instance.

        Beware that the second sequence, will be used to fill the targets
        field without running any kind of type conversion.

        :param sequences: A sequence of sequences, Tensors or ndarrays
            representing the patterns.
        :param targets: A sequence representing the targets field of the
            dataset. Can either be 1) a sequence of values containing as many
            elements as the number of contained patterns, or 2) the index
            of the sequence to use as the targets field. Defaults to 1, which
            means that the second sequence (usually containing the "y" values)
            will be used for the targets field.
        """
        if len(sequences) < 1:
            raise ValueError("At least one sequence must be passed")

        common_size = len(sequences[0])
        for seq in sequences:
            if len(seq) != common_size:
                raise ValueError(
                    "Sequences must contain the same " "amount of elements"
                )

        self._sequences = sequences
        if isinstance(targets, int):
            targets = sequences[targets]

        self.targets: Sequence[TTargetType] = targets

    def __getitem__(self, idx):
        return tuple(seq[idx] for seq in self._sequences)

    def __len__(self) -> int:
        return len(self._sequences[0])


def find_list_from_index(
    pattern_idx: int,
    list_sizes: Sequence[int],
    max_size: int,
    cumulative_sizes=None,
):
    if pattern_idx >= max_size:
        raise IndexError()

    if cumulative_sizes is None:
        r, s = [], 0
        for list_len in list_sizes:
            r.append(list_len + s)
            s += list_len
        cumulative_sizes = r

    list_idx = bisect.bisect_right(cumulative_sizes, pattern_idx)
    if list_idx != 0:
        pattern_idx = pattern_idx - cumulative_sizes[list_idx - 1]

    if pattern_idx >= list_sizes[list_idx]:
        raise ValueError("Index out of bounds, wrong max_size parameter")
    return list_idx, pattern_idx


def manage_advanced_indexing(
    idx, single_element_getter, max_length, collate_fn
):
    """
    Utility function used to manage the advanced indexing and slicing.

    If more than a pattern is selected, the X and Y values will be merged
    in two separate torch Tensor objects using the stack operation.

    :param idx: Either an in, a slice object or a list (including ndarrays and
        torch Tensors) of indexes.
    :param single_element_getter: A callable used to obtain a single element
        given its int index.
    :param max_length: The maximum sequence length.
    :param collate_fn: The function to use to create a batch of data from
        single elements.
    :return: A tuple consisting of two tensors containing the X and Y values
        of the patterns addressed by the idx parameter.
    """
    indexes_iterator: Iterable[int]

    # Makes dataset sliceable
    if isinstance(idx, slice):
        indexes_iterator = range(*idx.indices(max_length))
    elif isinstance(idx, int):
        indexes_iterator = [idx]
    elif hasattr(idx, "shape") and len(getattr(idx, "shape")) == 0:
        # Manages 0-d ndarray / Tensor
        indexes_iterator = [int(idx)]
    else:
        indexes_iterator = idx

    elements = []
    for single_idx in indexes_iterator:
        single_element = single_element_getter(int(single_idx))
        elements.append(single_element)

    if len(elements) == 1:
        return elements[0]

    return collate_fn(elements)


class LazySubsequence(Sequence[int]):
    """
    LazySubsequence can be used to define a Sequence based on another sequence
    and a pair of start and end indices.
    """

    def __init__(self, sequence: Sequence[int], start_idx: int, end_idx: int):
        self._sequence = sequence
        self._start_idx = start_idx
        self._end_idx = end_idx

    def __len__(self):
        return self._end_idx - self._start_idx

    def __getitem__(self, item_idx) -> int:
        if item_idx >= len(self):
            raise IndexError()

        return self._sequence[self._start_idx + item_idx]

    def __str__(self):
        return (
            "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"
        )


def optimize_sequence(sequence: Sequence[TTargetType]) -> Sequence[TTargetType]:
    if len(sequence) == 0 or isinstance(sequence, ConstantSequence):
        return sequence

    if isinstance(sequence, list):
        return sequence

    return list(sequence)


class TupleTLabel(tuple):
    """
    A simple tuple class used to describe a value returned from a dataset
    in which the task label is contained.

    Being a vanilla subclass of tuple, this class can be used to describe both a
    single instance and a batch.
    """

    def __new__(cls, *data, **kwargs):
        return super(TupleTLabel, cls).__new__(cls, *data, **kwargs)


def make_tuple(new_tuple: Iterable[T_co], prev_tuple: tuple):
    if isinstance(prev_tuple, TupleTLabel):
        return TupleTLabel(new_tuple)

    return new_tuple


__all__ = [
    "SubSequence",
    "LazyClassMapping",
    "LazyConcatTargets",
    "LazyConcatIntTargets",
    "ConstantSequence",
    "SubsetWithTargets",
    "ClassificationSubset",
    "SequenceDataset",
    "find_list_from_index",
    "manage_advanced_indexing",
    "optimize_sequence",
    "TupleTLabel",
]
