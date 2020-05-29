################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

from typing import List, Any, Iterable, Sequence, Union, Optional, TypeVar, \
    Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

T_co = TypeVar('T_co', covariant=True)


class IDataset(Protocol[T_co]):
    """
    Protocol definition of a Dataset.

    Note: no __add__ method is defined.
    """

    def __getitem__(self, index: int) -> T_co:
        ...

    def __len__(self) -> int:
        ...


# General rule: consume IDatasetWithTargets, produce DatasetWithTargets.
#
# That is, accept IDatasetWithTargets as parameter to functions/constructors
# (when possible), but always expose/return instances of DatasetWithTargets to
# the, user (no matter what). The main difference is that DatasetWithTargets is
# a  subclass of the PyTorch Dataset while IDatasetWithTargets is just a
# Protocol. This will allow the user to pass any custom dataset while
# receiving Dataset subclasses as outputs at the same time. This will allow
# popular IDEs (like PyCharm) to properly execute type checks and warn the user
# if something is wrong.
class IDatasetWithTargets(IDataset[T_co], Protocol):
    """
    Protocol definition of a Dataset that has a valid targets field (like the
    Datasets in the torchvision package)
    """
    targets: Sequence[int]


class DatasetWithTargets(IDatasetWithTargets[T_co], Dataset):
    """
    Dataset that has a valid targets field (like the Datasets in the
    torchvision package)

    The actual value of the targets field should be set by the child class.
    """
    def __init__(self):
        self.targets: Sequence[int] = []


def _manage_advanced_indexing(idx, single_element_getter, max_length):
    patterns: List[Any] = []
    labels: List[Tensor] = []
    indexes_iterator: Iterable[int]

    treat_as_tensors: bool = True

    # Makes dataset sliceable
    if isinstance(idx, slice):
        indexes_iterator = range(*idx.indices(max_length))
    elif isinstance(idx, int):
        indexes_iterator = [idx]
    else:  # Should handle other types (ndarray, Tensor, Sequence, ...)
        if hasattr(idx, 'shape') and \
                len(getattr(idx, 'shape')) == 0:
            # Manages 0-d ndarray / Tensor
            indexes_iterator = [int(idx)]
        else:
            indexes_iterator = idx

    for single_idx in indexes_iterator:
        pattern, label = single_element_getter(single_idx)
        if not isinstance(pattern, Tensor):
            treat_as_tensors = False

        label = torch.as_tensor(label)

        patterns.append(pattern)
        labels.append(label)

    if len(patterns) == 1:
        return patterns[0], labels[0]
    else:
        labels_cat = torch.stack(labels)
        patterns_cat = patterns

        if treat_as_tensors:
            patterns_cat = torch.stack(patterns)

        return patterns_cat, labels_cat


class TransformationDataset(DatasetWithTargets):
    """
    A Dataset that applies transformations before returning patterns/targets.
    Also, this Dataset supports basic slicing and advanced indexing.
    """

    def __init__(self, dataset: IDatasetWithTargets, transform=None,
                 target_transform=None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        self.targets = dataset.targets

    def __getitem__(self, idx):
        return _manage_advanced_indexing(idx, self.__get_single_item,
                                         len(self.dataset))

    def __len__(self) -> int:
        return len(self.dataset)

    def __get_single_item(self, idx: int):
        pattern, label = self.dataset[idx]
        if self.transform is not None:
            pattern = self.transform(pattern)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return pattern, label


class LazyClassMapping(Sequence[int]):
    """
    Defines a lazy targets class_list_per_batch.

    This class is used when in need of lazy populating a targets field whose
    elements need to be filtered out (when subsetting, see
    :class:`torch.utils.data.Subset`) and/or transformed (based on some
    class_list_per_batch). This will allow for a more efficient memory usage as
    the class_list_per_batch is done on the fly instead of actually allocating a
    new list.
    """
    def __init__(self, targets: Sequence[int],
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

        if self._mapping is not None:
            return self._mapping[self._targets[subset_idx]]
        else:
            return self._targets[subset_idx]

    def __str__(self):
        return '[' + \
               ', '.join([str(self[idx]) for idx in range(len(self))]) + \
               ']'


class TransformationSubset(DatasetWithTargets):
    """
    A Dataset that behaves like a pytorch :class:`torch.utils.data.Subset`,
    with all the goodness of :class:`TransformationDataset`
    """

    def __init__(self, dataset: IDatasetWithTargets,
                 indices: Union[Sequence[int], None],
                 transform=None, target_transform=None,
                 class_mapping: Optional[Sequence[int]] = None):
        super().__init__()
        self.dataset = TransformationDataset(dataset, transform=transform,
                                             target_transform=target_transform)
        self.indices = indices
        self.class_mapping = class_mapping
        self.targets = LazyClassMapping(dataset.targets, indices,
                                        mapping=class_mapping)

    def __getitem__(self, idx):
        return _manage_advanced_indexing(idx, self.__get_single_item, len(self))

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self.dataset)

    def __get_single_item(self, idx: int):
        if self.indices is not None:
            result = self.dataset[self.indices[idx]]
        else:
            result = self.dataset[idx]

        if self.class_mapping is not None:
            return result[0], self.class_mapping[result[1]]
        else:
            return result


def find_correct_list(pattern_idx, list_sizes, max_size):
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


class LazyConcatTargets(Sequence[int]):
    """
    Defines a lazy targets concatenation.

    This class is used when in need of lazy populating a targets created
    as the concatenation of the targets field of multiple datasets.
    This will allow for a more efficient memory usage as the concatenation is
    done on the fly instead of actually allocating a new list.
    """
    def __init__(self, targets_list: Sequence[Sequence[int]]):
        self._targets_list = targets_list
        self._targets_lengths = [len(targets) for targets in targets_list]
        self._overall_length = sum(self._targets_lengths)

    def __len__(self):
        return self._overall_length

    def __getitem__(self, item_idx) -> int:
        targets_idx, internal_idx = find_correct_list(
            item_idx, self._targets_lengths, self._overall_length)
        return self._targets_list[targets_idx][internal_idx]


class ConcatDatasetWithTargets(DatasetWithTargets):
    """
    A Dataset that behaves like a pytorch
    :class:`torch.utils.data.ConcatDataset`. However, this dataset also
    supports basic slicing and advanced indexing and also has a targets field.
    """

    def __init__(self, datasets: Sequence[IDatasetWithTargets]):
        super().__init__()
        self.datasets = datasets
        self._datasets_lengths = [len(dataset) for dataset in datasets]
        self._overall_length = sum(self._datasets_lengths)
        self.targets = LazyConcatTargets(
            [dataset.targets for dataset in datasets])

    def __getitem__(self, idx):
        return _manage_advanced_indexing(idx, self.__get_single_item,
                                         self._overall_length)

    def __len__(self) -> int:
        return self._overall_length

    def __get_single_item(self, idx: int):
        dataset_idx, internal_idx = find_correct_list(
            idx, self._datasets_lengths, self._overall_length)
        return self.datasets[dataset_idx][internal_idx]


def concat_datasets_sequentially(
        train_dataset_list: Sequence[IDatasetWithTargets],
        test_dataset_list: Sequence[IDatasetWithTargets]):
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

    dataset[1] class "1" -> new classID is "4"

    ... -> ...

    dataset[N-1] class "C-1" -> new class ID is "overall_n_classes-1"

    In contract, using PyTorch ConcatDataset:

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
            TransformationSubset(train_set, None,
                                 class_mapping=class_mapping))
        remapped_test_datasets.append(
            TransformationSubset(test_set, None,
                                 class_mapping=class_mapping))
        next_remapped_idx += classes_per_dataset[dataset_idx]

    return ConcatDatasetWithTargets(remapped_train_datasets), \
        ConcatDatasetWithTargets(remapped_test_datasets), \
        new_class_ids_per_dataset


__all__ = ['IDataset', 'IDatasetWithTargets', 'DatasetWithTargets',
           'TransformationDataset', 'TransformationSubset',
           'ConcatDatasetWithTargets', 'concat_datasets_sequentially']
