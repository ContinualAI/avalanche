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
from .dataset_definitions import ITensorDataset, ClassificationDataset, \
    IDatasetWithTargets, ISupportedClassificationDataset

try:
    from typing import Protocol, Sequence, List, Any, Iterable, Union, \
        Optional, SupportsInt, TypeVar, Tuple
except ImportError:
    from typing import Sequence, List, Any, Iterable, Union, Optional, \
         SupportsInt, TypeVar, Tuple
    from typing_extensions import Protocol

T_co = TypeVar('T_co', covariant=True)
TTargetType = TypeVar('TTargetType')


class SubSequence(Sequence[TTargetType]):
    """
    A utility class used to define a lazily evaluated sub-sequence.
    """
    def __init__(self,
                 targets: Sequence[TTargetType],
                 indices: Union[Sequence[int], None]):
        self._targets = targets
        self._indices = indices

    def __len__(self):
        if self._indices is None:
            return len(self._targets)
        return len(self._indices)

    def __getitem__(self, item_idx) -> TTargetType:
        if self._indices is not None:
            subset_idx = self._indices[item_idx]
        else:
            subset_idx = item_idx

        return self._targets[subset_idx]

    def __str__(self):
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'


class LazyClassMapping(SubSequence[int]):
    """
    This class is used when in need of lazy populating a targets field whose
    elements need to be filtered out (when subsetting, see
    :class:`torch.utils.data.Subset`) and/or transformed (remapped). This will
    allow for a more efficient memory usage as the conversion is done on the fly
    instead of actually allocating a new targets list.

    This class should be used only when mapping int targets (classification).
    """
    def __init__(self,
                 targets: Sequence[SupportsInt],
                 indices: Union[Sequence[int], None],
                 mapping: Optional[Sequence[int]] = None):
        super().__init__(targets, indices)
        self._mapping = mapping

    def __getitem__(self, item_idx) -> int:
        target_value = int(super().__getitem__(item_idx))

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
    def __init__(self, targets_list: Sequence[Sequence[TTargetType]]):
        self._targets_list = targets_list
        self._targets_lengths = [len(targets) for targets in targets_list]
        self._overall_length = sum(self._targets_lengths)

    def __len__(self):
        return self._overall_length

    def __getitem__(self, item_idx) -> TTargetType:
        targets_idx, internal_idx = find_list_from_index(
            item_idx, self._targets_lengths, self._overall_length)
        return self._targets_list[targets_idx][internal_idx]

    def __str__(self):
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'


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
        super().__init__(targets_list)

    def __getitem__(self, item_idx) -> int:
        return int(super().__getitem__(item_idx))


class LazyTargetsConversion(Sequence[int]):
    """
    Defines a lazy conversion of targets defined in some other format.

    To be used when transforming targets to int (classification).
    """
    def __init__(self, targets: Sequence[SupportsInt]):
        self._targets = targets

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, item_idx) -> int:
        return int(self._targets[item_idx])

    def __str__(self):
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'


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
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'


class SubsetWithTargets(IDatasetWithTargets[T_co, TTargetType]):
    """
    A Dataset that behaves like a PyTorch :class:`torch.utils.data.Subset`.
    However, this dataset also supports the targets field.
    """
    def __init__(self,
                 dataset: IDatasetWithTargets[T_co, TTargetType],
                 indices: Union[Sequence[int], None]):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.targets = SubSequence(dataset.targets, indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            result = self.dataset[self.indices[idx]]
        else:
            result = self.dataset[idx]

        return result

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self.dataset)


class ClassificationSubset(SubsetWithTargets[T_co, int]):
    """
    A Dataset that behaves like a PyTorch :class:`torch.utils.data.Subset`.
    However, this dataset also supports the targets field and class mapping.
    """
    def __init__(self,
                 dataset: ISupportedClassificationDataset[T_co],
                 indices: Union[Sequence[int], None],
                 class_mapping: Optional[Sequence[int]] = None):
        super().__init__(dataset, indices)
        self.class_mapping = class_mapping
        self.targets = LazyClassMapping(dataset.targets, indices,
                                        mapping=class_mapping)

    def __getitem__(self, idx):
        result = super().__getitem__(idx)

        if self.class_mapping is not None:
            return (result[0], self.class_mapping[result[1]], *result[2:])

        return result


class ConcatDatasetWithTargets(IDatasetWithTargets[T_co, TTargetType]):
    """
    A Dataset that behaves like a PyTorch
    :class:`torch.utils.data.ConcatDataset`. In addition, this dataset also
    supports the concatenation of the targets field.
    """
    def __init__(self,
                 datasets: Sequence[IDatasetWithTargets[T_co, TTargetType]]):
        super().__init__()
        self.datasets = datasets
        self._datasets_lengths = [len(dataset) for dataset in datasets]
        self._overall_length = sum(self._datasets_lengths)
        self.targets = LazyConcatTargets(
            [dataset.targets for dataset in datasets])

    def __getitem__(self, idx):
        dataset_idx, internal_idx = find_list_from_index(
            idx, self._datasets_lengths, self._overall_length)
        return self.datasets[dataset_idx][internal_idx]

    def __len__(self):
        return self._overall_length


class SequenceDataset(IDatasetWithTargets[T_co, TTargetType]):
    """
    A Dataset that wraps existing ndarrays, Tensors, lists... to provide
    basic Dataset functionalities. Very similar to TensorDataset.
    """
    def __init__(self,
                 *sequences: Sequence):
        """
        Creates a ``SequenceDataset`` instance.

        Beware that the second sequence, will be used to fill the targets
        field without running any kind of type conversion.

        :param sequences: A sequence of sequences, Tensors or ndarrays
            representing the patterns.
        """
        super().__init__()
        if len(sequences) < 2:
            raise ValueError('At least 2 sequences are required')

        common_size = len(sequences[0])
        for seq in sequences:
            if len(seq) != common_size:
                raise ValueError('Sequences must contain the same '
                                 'amount of elements')

        self._sequences = sequences
        self.targets = sequences[1]

    def __getitem__(self, idx):
        return tuple(seq[idx] for seq in self._sequences)

    def __len__(self) -> int:
        return len(self._sequences[0])


class TensorDatasetWrapper(ClassificationDataset[T_co]):
    """
    A Dataset that wraps a Tensor Dataset to provide the targets field.

    A Tensor Dataset is any dataset with a "tensors" field. The tensors
    field must be a sequence of Tensor. To provide a valid targets field,
    the "tensors" field must contain at least 2 tensors. The second tensor
    must contain elements that can be converted to int.

    Beware that the second element obtained from the wrapped dataset using
    __getitem__ will always be converted to int, This differs from the
    behaviour of PyTorch TensorDataset. This is required to keep a better
    compatibility with torchvision datasets.
    """
    def __init__(self, tensor_dataset: ITensorDataset[T_co]):
        """
        Creates a ``TensorDatasetWrapper`` instance.

        :param tensor_dataset: An instance of a TensorDataset. See class
            description for more details.
        """
        super().__init__()
        if len(tensor_dataset.tensors) < 2:
            raise ValueError('Tensor dataset has not enough tensors: '
                             'at least 2 are required.')

        self.dataset = tensor_dataset
        self.targets = LazyTargetsConversion(tensor_dataset.tensors[1])

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


def find_list_from_index(pattern_idx: int,
                         list_sizes: Sequence[int],
                         max_size: int):
    if pattern_idx >= max_size:
        raise IndexError()

    cumulative_length = 0
    for list_idx, list_length in enumerate(list_sizes):
        dataset_span = cumulative_length + list_length
        if pattern_idx < dataset_span:
            return list_idx, \
                   pattern_idx - cumulative_length

        cumulative_length = dataset_span
    raise ValueError('Index out of bounds, wrong max_size parameter')


def manage_advanced_indexing(idx, single_element_getter, max_length,
                             collate_fn):
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
    elif hasattr(idx, 'shape') and len(getattr(idx, 'shape')) == 0:
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
    def __init__(self,
                 sequence: Sequence[int],
                 start_idx: int,
                 end_idx: int):
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
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'


def optimize_sequence(sequence: Sequence[TTargetType]) -> Sequence[TTargetType]:
    if len(sequence) == 0 or isinstance(sequence, ConstantSequence):
        return sequence

    if not isinstance(sequence[0], int):
        # Can only optimize sequence of ints
        return sequence

    concat_ranges = []
    streak_value = None
    start_idx = -1
    streak_start_idx = 0

    for i, x in enumerate(sequence):
        if i - streak_start_idx == 50 and streak_start_idx != start_idx:
            concat_ranges.append(LazySubsequence(sequence, start_idx,
                                                 streak_start_idx))
            start_idx = streak_start_idx

        if i == 0:
            streak_start_idx = i
            start_idx = i
            streak_value = x
        elif x != streak_value:
            if i - streak_start_idx >= 50:
                concat_ranges.append(ConstantSequence(streak_value,
                                                      i - streak_start_idx))
                start_idx = i

            streak_start_idx = i
            streak_value = x
        else:  # x == last_value
            pass

    i = len(sequence)
    if i - streak_start_idx < 50:
        concat_ranges.append(LazySubsequence(sequence, start_idx, i))
    else:
        if streak_start_idx != start_idx:
            concat_ranges.append(LazySubsequence(sequence, start_idx,
                                                 streak_start_idx))

        concat_ranges.append(ConstantSequence(streak_value,
                                              i - streak_start_idx))

    if len(concat_ranges) == 1:
        if isinstance(concat_ranges[0], LazySubsequence):
            # Couldn't optimize
            return sequence
        return concat_ranges[0]  # Best situation ever: we got a single range!

    return LazyConcatIntTargets(concat_ranges)


__all__ = [
    'SubSequence',
    'LazyClassMapping',
    'LazyConcatTargets',
    'LazyConcatIntTargets',
    'LazyTargetsConversion',
    'ConstantSequence',
    'SubsetWithTargets',
    'ClassificationSubset',
    'ConcatDatasetWithTargets',
    'SequenceDataset',
    'TensorDatasetWrapper',
    'find_list_from_index',
    'manage_advanced_indexing',
    'optimize_sequence'
]
