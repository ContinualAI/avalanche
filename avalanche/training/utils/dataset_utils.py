
from typing import Protocol, Sequence, List, Any, Iterable, Union, Optional, \
    SupportsInt, TypeVar, Tuple

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

    def __getitem__(self, index: int) -> Tuple[T_co, int]:
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

    def __getitem__(self, index: int) -> Tuple[T_co, int]:
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

    def __getitem__(self, index: int) -> Tuple[T_co, int]:
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
    def __init__(self, targets: Sequence[SupportsInt],
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


class SubsetWithTargets(DatasetWithTargets[T_co]):
    """
    A Dataset that behaves like a PyTorch :class:`torch.utils.data.Subset`.
    However, this dataset also supports the targets field and class mapping.
    """
    def __init__(self, dataset: IDatasetWithTargets[T_co],
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


def concat_datasets_sequentially(
        train_dataset_list: Sequence[IDatasetWithTargets[T_co]],
        test_dataset_list: Sequence[IDatasetWithTargets[T_co]]) -> \
        Tuple[IDatasetWithIntTargets[T_co], IDatasetWithIntTargets[T_co],
              List[list]]:
    """
    Concatenates a list of datasets. This is completely different from
    :class:`ConcatDataset`, in which datasets are merged together without
    other processing. Instead, this function re-maps the datasets class IDs.
    For instance:
    let the dataset[0] contain patterns of 3 different classes,
    let the dataset[1] contain patterns of 2 different classes, then class IDs
    will be mapped as follows:

    dataset[0] class "0" -> new class ID is "0"

    dataset[0] class "1" -> new class ID is "1"

    dataset[0] class "2" -> new class ID is "2"

    dataset[1] class "0" -> new class ID is "3"

    dataset[1] class "1" -> new class ID is "4"

    ... -> ...

    dataset[-1] class "C-1" -> new class ID is "overall_n_classes-1"

    In contrast, using PyTorch ConcatDataset:

    dataset[0] class "0" -> ID is "0"

    dataset[0] class "1" -> ID is "1"

    dataset[0] class "2" -> ID is "2"

    dataset[1] class "0" -> ID is "0"

    dataset[1] class "1" -> ID is "1"

    Note: ``train_dataset_list`` and ``test_dataset_list`` must have the same
    number of datasets.

    :param train_dataset_list: A list of training datasets
    :param test_dataset_list: A list of test datasets

    :returns: A concatenated dataset.
    """
    remapped_train_datasets = []
    remapped_test_datasets = []
    next_remapped_idx = 0

    # Obtain the number of classes of each dataset
    classes_per_dataset = [
        len(torch.unique(
            torch.cat((torch.as_tensor(train_dataset_list[dataset_idx].targets),
                      torch.as_tensor(test_dataset_list[dataset_idx].targets)))
            )) for dataset_idx in range(len(train_dataset_list))
    ]

    new_class_ids_per_dataset = []
    for dataset_idx in range(len(train_dataset_list)):
        # The class IDs for this dataset will be in range
        # [n_classes_in_previous_datasets,
        #       n_classes_in_previous_datasets + classes_in_this_dataset)
        class_mapping = list(
            range(next_remapped_idx,
                  next_remapped_idx + classes_per_dataset[dataset_idx]))
        new_class_ids_per_dataset.append(class_mapping)

        train_set = train_dataset_list[dataset_idx]
        test_set = test_dataset_list[dataset_idx]

        # TransformationSubset is used to apply the class IDs transformation.
        # Remember, the class_mapping parameter must be a list in which:
        # new_class_id = class_mapping[original_class_id]
        remapped_train_datasets.append(
            SubsetWithTargets(train_set, None, class_mapping=class_mapping))
        remapped_test_datasets.append(
            SubsetWithTargets(test_set, None, class_mapping=class_mapping))
        next_remapped_idx += classes_per_dataset[dataset_idx]

    return ConcatDatasetWithTargets(remapped_train_datasets), \
        ConcatDatasetWithTargets(remapped_test_datasets), \
        new_class_ids_per_dataset


class SequenceDataset(DatasetWithTargets[T_co]):
    """
    A Dataset that wraps existing ndarrays, Tensors, lists... to provide
    basic Dataset functionalities. Very similar to TensorDataset.
    """
    def __init__(self, dataset_x: Sequence[T_co],
                 dataset_y: Sequence[SupportsInt]):
        """
        Creates a ``SequenceDataset`` instance.

        :param dataset_x: An sequence, Tensor or ndarray representing the X
            values of the patterns.
        :param dataset_y: An sequence, Tensor int or ndarray of integers
            representing the Y values of the patterns.
        """
        super(SequenceDataset, self).__init__()
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
        pattern, label = single_element_getter(int(single_idx))
        if not isinstance(pattern, Tensor):
            treat_as_tensors = False

        patterns.append(pattern)
        labels.append(label)

    if len(patterns) == 1:
        return patterns[0], labels[0]

    labels_cat = torch.tensor(labels)
    patterns_cat = patterns

    if treat_as_tensors:
        patterns_cat = torch.stack(patterns)

    return patterns_cat, labels_cat


__all__ = ['IDataset', 'IDatasetWithTargets', 'IDatasetWithIntTargets',
           'DatasetWithTargets', 'LazyClassMapping', 'LazyConcatTargets',
           'LazyTargetsConversion', 'SubsetWithTargets',
           'ConcatDatasetWithTargets', 'concat_datasets_sequentially',
           'SequenceDataset', 'find_list_from_index',
           'manage_advanced_indexing']
