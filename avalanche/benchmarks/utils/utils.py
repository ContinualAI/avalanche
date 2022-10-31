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

""" Common benchmarks/environments utils. """

from collections import OrderedDict
from typing import List, Iterable, Sequence, Union, Dict, Tuple, SupportsInt

import torch
from torch import Tensor

from avalanche.benchmarks.utils.classification_dataset import (
    classification_subset,
    concat_classification_datasets,
)
from avalanche.benchmarks.utils.classification_dataset import (
    T_co,
    ClassificationDataset,
)
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.dataset_definitions import (
    ISupportedClassificationDataset,
)


def tensor_as_list(sequence) -> List:
    # Numpy: list(np.array([1, 2, 3])) returns [1, 2, 3]
    # whereas: list(torch.tensor([1, 2, 3])) returns ->
    # -> [tensor(1), tensor(2), tensor(3)]
    #
    # This is why we have to handle Tensor in a different way
    if isinstance(sequence, list):
        return sequence
    if not isinstance(sequence, Iterable):
        return [sequence]
    if isinstance(sequence, Tensor):
        return sequence.tolist()
    return list(sequence)


def _indexes_grouped_by_classes(
    targets: Sequence[int],
    patterns_indexes: Union[None, Sequence[int]],
    sort_indexes: bool = True,
    sort_classes: bool = True,
) -> Union[List[int], None]:
    result_per_class: Dict[int, List[int]] = OrderedDict()
    result: List[int] = []

    indexes_was_none = patterns_indexes is None

    if patterns_indexes is not None:
        patterns_indexes = tensor_as_list(patterns_indexes)
    else:
        patterns_indexes = list(range(len(targets)))

    targets = tensor_as_list(targets)

    # Consider that result_per_class is an OrderedDict
    # This means that, if sort_classes is True, the next for statement
    # will initialize "result_per_class" in sorted order which in turn means
    # that patterns will be ordered by ascending class ID.
    classes = torch.unique(
        torch.as_tensor(targets), sorted=sort_classes
    ).tolist()

    for class_id in classes:
        result_per_class[class_id] = []

    # Stores each pattern index in the appropriate class list
    for idx in patterns_indexes:
        result_per_class[targets[idx]].append(idx)

    # Concatenate all the pattern indexes
    for class_id in classes:
        if sort_indexes:
            result_per_class[class_id].sort()
        result.extend(result_per_class[class_id])

    if result == patterns_indexes and indexes_was_none:
        # Result is [0, 1, 2, ..., N] and patterns_indexes was originally None
        # This means that the user tried to obtain a full Dataset
        # (indexes_was_none) only ordered according to the sort_indexes and
        # sort_classes parameters. However, sort_indexes+sort_classes returned
        # the plain pattern sequence as it already is. So the original Dataset
        # already satisfies the sort_indexes+sort_classes constraints.
        # By returning None, we communicate that the Dataset can be taken as-is.
        return None

    return result


def grouped_and_ordered_indexes(
    targets: Sequence[int],
    patterns_indexes: Union[None, Sequence[int]],
    bucket_classes: bool = True,
    sort_classes: bool = False,
    sort_indexes: bool = False,
) -> Union[List[int], None]:
    """
    Given the targets list of a dataset and the patterns to include, returns the
    pattern indexes sorted according to the ``bucket_classes``,
    ``sort_classes`` and ``sort_indexes`` parameters.

    :param targets: The list of pattern targets, as a list.
    :param patterns_indexes: A list of pattern indexes to include in the set.
        If None, all patterns will be included.
    :param bucket_classes: If True, pattern indexes will be returned so that
        patterns will be grouped by class. Defaults to True.
    :param sort_classes: If both ``bucket_classes`` and ``sort_classes`` are
        True, class groups will be sorted by class index. Ignored if
        ``bucket_classes`` is False. Defaults to False.
    :param sort_indexes: If True, patterns indexes will be sorted. When
        bucketing by class, patterns will be sorted inside their buckets.
        Defaults to False.

    :returns: The list of pattern indexes sorted according to the
        ``bucket_classes``, ``sort_classes`` and ``sort_indexes`` parameters or
        None if the patterns_indexes is None and the whole dataset can be taken
        using the existing patterns order.
    """
    if bucket_classes:
        return _indexes_grouped_by_classes(
            targets,
            patterns_indexes,
            sort_indexes=sort_indexes,
            sort_classes=sort_classes,
        )

    if patterns_indexes is None:
        # No grouping and sub-set creation required... just return None
        return None
    if not sort_indexes:
        # No sorting required, just return patterns_indexes
        return tensor_as_list(patterns_indexes)

    # We are here only because patterns_indexes != None and sort_indexes is True
    patterns_indexes = tensor_as_list(patterns_indexes)
    result = list(patterns_indexes)  # Make sure we're working on a copy
    result.sort()
    return result


def concat_datasets_sequentially(
    train_dataset_list: Sequence[ISupportedClassificationDataset],
    test_dataset_list: Sequence[ISupportedClassificationDataset],
) -> Tuple[ClassificationDataset, ClassificationDataset, List[list]]:
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
        _count_unique(
            train_dataset_list[dataset_idx].targets,
            test_dataset_list[dataset_idx].targets,
        )
        for dataset_idx in range(len(train_dataset_list))
    ]

    new_class_ids_per_dataset = []
    for dataset_idx in range(len(train_dataset_list)):

        # Get the train and test sets of the dataset
        train_set = train_dataset_list[dataset_idx]
        test_set = test_dataset_list[dataset_idx]

        # Get the classes in the dataset
        dataset_classes = set(map(int, train_set.targets))

        # The class IDs for this dataset will be in range
        # [n_classes_in_previous_datasets,
        #       n_classes_in_previous_datasets + classes_in_this_dataset)
        new_classes = list(
            range(
                next_remapped_idx,
                next_remapped_idx + classes_per_dataset[dataset_idx],
            )
        )
        new_class_ids_per_dataset.append(new_classes)

        # AvalancheSubset is used to apply the class IDs transformation.
        # Remember, the class_mapping parameter must be a list in which:
        # new_class_id = class_mapping[original_class_id]
        # Hence, a list of size equal to the maximum class index is created
        # Only elements corresponding to the present classes are remapped
        class_mapping = [-1] * (max(dataset_classes) + 1)
        j = 0
        for i in dataset_classes:
            class_mapping[i] = new_classes[j]
            j += 1

        # Create remapped datasets and append them to the final list
        remapped_train_datasets.append(
            classification_subset(train_set, class_mapping=class_mapping)
        )
        remapped_test_datasets.append(
            classification_subset(test_set, class_mapping=class_mapping)
        )
        next_remapped_idx += classes_per_dataset[dataset_idx]

    return (
        concat_datasets(remapped_train_datasets),
        concat_datasets(remapped_test_datasets),
        new_class_ids_per_dataset,
    )


def as_avalanche_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> AvalancheDataset:
    if isinstance(dataset, AvalancheDataset):
        return dataset
    return AvalancheDataset([dataset])


def as_classification_dataset(
    dataset: ISupportedClassificationDataset[T_co],
) -> ClassificationDataset:
    if isinstance(dataset, ClassificationDataset):
        return dataset
    return ClassificationDataset([dataset])


def _count_unique(*sequences: Sequence[SupportsInt]):
    uniques = set()

    for seq in sequences:
        for x in seq:
            uniques.add(int(x))

    return len(uniques)


def concat_datasets(datasets):
    """Concatenates a list of datasets."""
    if len(datasets) == 0:
        return AvalancheDataset([])
    res = datasets[0]
    if not isinstance(res, AvalancheDataset):
        res = AvalancheDataset([res])

    for d in datasets[1:]:
        if not isinstance(d, AvalancheDataset):
            d = AvalancheDataset([d])
        res = res.concat(d)
    return res


__all__ = [
    "tensor_as_list",
    "grouped_and_ordered_indexes",
    "concat_datasets_sequentially",
    "as_avalanche_dataset",
    "as_classification_dataset",
    "concat_datasets"
]
