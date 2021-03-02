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

try:
    from typing import Protocol, Sequence, List, Any, Iterable, Union, \
        Optional, SupportsInt, TypeVar, Tuple
except ImportError:
    from typing import Sequence, List, Any, Iterable, Union, Optional, \
         SupportsInt, TypeVar, Tuple
    from typing_extensions import Protocol

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

T_co = TypeVar('T_co', covariant=True)


# General rule: consume IDatasetWithTargets, produce DatasetWithTargets.
#
# That is, accept IDatasetWithTargets as parameter to functions/constructors
# (when possible), but always expose/return instances of DatasetWithTargets to
# the, user (no matter what). The main difference is that DatasetWithTargets is
# a subclass of the PyTorch Dataset while IDatasetWithTargets is just a
# Protocol. This will allow the user to pass any custom dataset while
# receiving Dataset subclasses as outputs at the same time. This will allow
# popular IDEs (like PyCharm) to properly execute type checks and warn the user
# if something is wrong.

TTargetType = TypeVar('TTargetType', bound=SupportsInt)


class IDataset(Protocol[T_co]):
    """
    Protocol definition of a Dataset.

    Note: no __add__ method is defined.
    """

    def __getitem__(self, index: int) -> Union[Tuple[T_co, int],
                                               Tuple[T_co, int, int]]:
        ...

    def __len__(self) -> int:
        ...


class IDatasetWithTargets(IDataset[T_co], Protocol):
    """
    Protocol definition of a Dataset that has a valid targets field (like the
    Datasets in the torchvision package).

    Note: no __add__ method is defined.
    """

    targets: Sequence[SupportsInt]
    """
    A sequence of ints or a PyTorch Tensor or a NumPy ndarray describing the
    label of each pattern contained in the dataset.
    """

    def __getitem__(self, index: int) -> Union[Tuple[T_co, int],
                                               Tuple[T_co, int, int]]:
        ...

    def __len__(self) -> int:
        ...


class ITensorDataset(IDataset[T_co], Protocol):
    """
    Protocol definition of a Dataset that has a tensors field (like
    TensorDataset).

    Note: no __add__ method is defined.
    """

    tensors: Sequence[T_co]
    """
    A sequence of PyTorch Tensors describing the contents of the Dataset.
    """

    def __getitem__(self, index: int) -> Union[Tuple[T_co, int],
                                               Tuple[T_co, int, int]]:
        ...

    def __len__(self) -> int:
        ...


class IDatasetWithIntTargets(IDatasetWithTargets[T_co], Protocol):
    """
    Protocol definition of a Dataset that has a valid targets field (like the
    Datasets in the torchvision package) where the targets field is a sequence
    of native ints.
    """

    targets: Sequence[int]
    """
    A sequence of ints describing the label of each pattern contained in the
    dataset.
    """

    def __getitem__(self, index: int) -> Union[Tuple[T_co, int],
                                               Tuple[T_co, int, int]]:
        ...

    def __len__(self) -> int:
        ...


class DatasetWithTargets(IDatasetWithIntTargets[T_co], Dataset):
    """
    Dataset that has a valid targets field (like the Datasets in the
    torchvision package) where the targets field is a sequence of native ints.

    The actual value of the targets field should be set by the child class.
    """
    def __init__(self):
        self.targets = []
        """
        A sequence of ints describing the label of each pattern contained in the
        dataset.
        """


class LazyClassMapping(Sequence[int]):
    """
    This class is used when in need of lazy populating a targets field whose
    elements need to be filtered out (when subsetting, see
    :class:`torch.utils.data.Subset`) and/or transformed (remapped). This will
    allow for a more efficient memory usage as the conversion is done on the fly
    instead of actually allocating a new targets list.
    """
    def __init__(self,
                 targets: Sequence[SupportsInt],
                 indices: Union[Sequence[int], None],
                 mapping: Optional[Sequence[int]] = None):
        self._targets = targets
        self._mapping = mapping
        self._indices = indices

    def __len__(self):
        if self._indices is None:
            return len(self._targets)
        return len(self._indices)

    def __getitem__(self, item_idx) -> int:
        if self._indices is not None:
            subset_idx = self._indices[item_idx]
        else:
            subset_idx = item_idx

        target_value = int(self._targets[subset_idx])

        if self._mapping is not None:
            return self._mapping[target_value]

        return target_value

    def __str__(self):
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'


class LazyConcatTargets(Sequence[int]):
    """
    Defines a lazy targets concatenation.

    This class is used when in need of lazy populating a targets created
    as the concatenation of the targets field of multiple datasets.
    This will allow for a more efficient memory usage as the concatenation is
    done on the fly instead of actually allocating a new targets list.
    """
    def __init__(self, targets_list: Sequence[Sequence[SupportsInt]]):
        self._targets_list = targets_list
        self._targets_lengths = [len(targets) for targets in targets_list]
        self._overall_length = sum(self._targets_lengths)

    def __len__(self):
        return self._overall_length

    def __getitem__(self, item_idx) -> int:
        targets_idx, internal_idx = find_list_from_index(
            item_idx, self._targets_lengths, self._overall_length)
        return int(self._targets_list[targets_idx][internal_idx])

    def __str__(self):
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'


class LazyTargetsConversion(Sequence[int]):
    """
    Defines a lazy conversion of targets defined in some other format.
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
    Defines a lazy conversion of targets defined in some other format.
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


class SubsetWithTargets(DatasetWithTargets[T_co]):
    """
    A Dataset that behaves like a PyTorch :class:`torch.utils.data.Subset`.
    However, this dataset also supports the targets field and class mapping.
    """
    def __init__(self,
                 dataset: IDatasetWithTargets[T_co],
                 indices: Union[Sequence[int], None],
                 class_mapping: Optional[Sequence[int]] = None):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.class_mapping = class_mapping
        self.targets = LazyClassMapping(dataset.targets, indices,
                                        mapping=class_mapping)

    def __getitem__(self, idx):
        if self.indices is not None:
            result = self.dataset[self.indices[idx]]
        else:
            result = self.dataset[idx]

        if self.class_mapping is not None:
            return result[0], self.class_mapping[result[1]]

        return result

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self.dataset)


class ConcatDatasetWithTargets(DatasetWithTargets[T_co]):
    """
    A Dataset that behaves like a PyTorch
    :class:`torch.utils.data.ConcatDataset`. However, this dataset also
    supports the targets field.
    """
    def __init__(self, datasets: Sequence[IDatasetWithTargets[T_co]]):
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

    def __len__(self) -> int:
        return self._overall_length


class SequenceDataset(DatasetWithTargets[T_co]):
    """
    A Dataset that wraps existing ndarrays, Tensors, lists... to provide
    basic Dataset functionalities. Very similar to TensorDataset.
    """
    def __init__(self,
                 dataset_x: Sequence[T_co],
                 dataset_y: Sequence[SupportsInt]):
        """
        Creates a ``SequenceDataset`` instance.

        :param dataset_x: An sequence, Tensor or ndarray representing the X
            values of the patterns.
        :param dataset_y: An sequence, Tensor int or ndarray of integers
            representing the Y values of the patterns.
        """
        super().__init__()
        if len(dataset_x) != len(dataset_y):
            raise ValueError('dataset_x and dataset_y must contain the same '
                             'amount of elements')

        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.targets = LazyTargetsConversion(dataset_y)

    def __getitem__(self, idx):
        return self.dataset_x[idx], self.targets[idx]

    def __len__(self) -> int:
        return len(self.dataset_x)


class TensorDatasetWrapper(DatasetWithTargets[T_co]):
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
        obtained_item = list(self.dataset[idx])
        obtained_item[1] = int(obtained_item[1])
        return tuple(obtained_item)

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


def manage_advanced_indexing(idx, single_element_getter, max_length):
    """
    Utility function used to manage the advanced indexing and slicing.

    If more than a pattern is selected, the X and Y values will be merged
    in two separate torch Tensor objects using the stack operation.

    :param idx: Either an in, a slice object or a list (including ndarrays and
        torch Tensors) of indexes.
    :param single_element_getter: A callable used to obtain a single element
        given its int index.
    :param max_length: The maximum sequence length.
    :return: A tuple consisting of two tensors containing the X and Y values
        of the patterns addressed by the idx parameter.
    """
    patterns: List[Any] = []
    labels: List[Tensor] = []
    task_labels: List[int] = []
    has_task_labels = False
    indexes_iterator: Iterable[int]

    treat_as_tensors: bool = True

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

    for single_idx in indexes_iterator:
        single_element = single_element_getter(int(single_idx))
        task_label = 0
        if len(single_element) > 2:
            has_task_labels = True
            pattern, label, task_label = single_element
        else:
            pattern, label = single_element

        if not isinstance(pattern, Tensor):
            treat_as_tensors = False

        patterns.append(pattern)
        labels.append(label)
        if has_task_labels:
            task_labels.append(task_label)

    if len(patterns) == 1:
        if has_task_labels:
            return patterns[0], labels[0], task_labels[0]
        else:
            return patterns[0], labels[0]

    task_labels_cat = None
    labels_cat = torch.tensor(labels)
    if has_task_labels:
        task_labels_cat = torch.tensor(task_labels)
    patterns_cat = patterns

    if treat_as_tensors:
        patterns_cat = torch.stack(patterns)
    if has_task_labels:
        return patterns_cat, labels_cat, task_labels_cat
    else:
        return patterns_cat, labels_cat


class LazySubsequence(Sequence[int]):
    """
    TODO: doc
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


def optimize_sequence(sequence: Sequence[int]) -> Sequence[int]:
    if len(sequence) == 0 or isinstance(sequence, ConstantSequence):
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

    return LazyConcatTargets(concat_ranges)


__all__ = [
    'IDataset',
    'IDatasetWithTargets',
    'ITensorDataset',
    'IDatasetWithIntTargets',
    'DatasetWithTargets',
    'LazyClassMapping',
    'LazyConcatTargets',
    'LazyTargetsConversion',
    'ConstantSequence',
    'SubsetWithTargets',
    'ConcatDatasetWithTargets',
    'SequenceDataset',
    'TensorDatasetWrapper',
    'find_list_from_index',
    'manage_advanced_indexing',
    'optimize_sequence'
]
