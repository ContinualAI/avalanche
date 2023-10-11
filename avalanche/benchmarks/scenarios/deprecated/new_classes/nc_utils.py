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

from collections import OrderedDict
from typing import List, Sequence, Dict, Any, Union, SupportsInt

import torch

from avalanche.benchmarks.utils import (
    _taskaware_classification_subset,
    SupportedDataset,
)
from avalanche.benchmarks.utils import tensor_as_list
from avalanche.benchmarks.utils.classification_dataset import (
    TaskAwareClassificationDataset,
)


def _indexes_grouped_by_classes(
    sequence: Sequence[SupportsInt],
    search_elements: Union[None, Sequence[int]],
    sort_indexes: bool = True,
    sort_classes: bool = True,
) -> Union[List[int], None]:
    result_per_class: Dict[int, List[int]] = OrderedDict()
    result: List[int] = []

    # tensor_as_list() handles the situation in which sequence and
    # search_elements are a torch.Tensor
    #
    # Without the tensor_as_list conversion:
    # result_per_class[element].append(idx) -> error
    # because result_per_class[0] won't exist (result_per_class[tensor(0)] will)

    sequence_list: List[int] = tensor_as_list(sequence)
    if search_elements is not None:
        search_elements_list = tensor_as_list(search_elements)
    else:
        search_elements_list = torch.unique(torch.as_tensor(sequence_list)).tolist()

    if sort_classes:
        # Consider that result_per_class is an OrderedDict
        # This means that, if sort_classes is True, the next for statement
        # will initialize the "result_per_class" in sorted order ->
        # -> patterns will be ordered by ascending class ID
        search_elements_list = sorted(search_elements_list)

    for search_element in search_elements_list:
        result_per_class[search_element] = []

    # Set based "in" operator is **much** faster that its list counterpart!
    search_elements_set = set()
    if search_elements_list is not None:
        search_elements_set = set(search_elements_list)

    # Stores each pattern index in the appropriate class list
    for idx, element in enumerate(sequence_list):
        if element in search_elements_set:
            result_per_class[element].append(idx)

    # Concatenate all the pattern indexes
    for search_element in search_elements_list:
        if sort_indexes:
            result_per_class[search_element].sort()
        result.extend(result_per_class[search_element])

    if result == sequence_list:
        # The resulting index order is the same as the input one
        # Return None to flag that the whole sequence can be
        # taken as it already is
        return None

    return result


def _indexes_without_grouping(
    sequence: Sequence[SupportsInt],
    search_elements: Union[None, Sequence[int]],
    sort_indexes: bool = False,
) -> Union[List[int], None]:
    sequence = tensor_as_list(sequence)

    if search_elements is None and not sort_indexes:
        # No-op
        return list(range(len(sequence)))

    if search_elements is not None:
        search_elements = tensor_as_list(search_elements)

    result: List[int]
    if search_elements is None:
        result = list(range(len(sequence)))
    else:
        # Set based "in" operator is **much** faster that its list counterpart!
        search_elements_set = set(search_elements)
        result = []
        for idx, element in enumerate(sequence):
            if element in search_elements_set:
                result.append(idx)

    if sort_indexes:
        result.sort()
    elif not sort_indexes and len(result) == len(sequence):
        # All patterns selected. Also, no sorting is required
        # Return None to flag that the whole sequence can be
        # taken as it already is
        return None
    return result


def _indexes_from_set(
    sequence: Sequence[SupportsInt],
    search_elements: Union[Sequence[int], None],
    bucket_classes: bool = True,
    sort_classes: bool = False,
    sort_indexes: bool = False,
) -> Union[List[int], None]:
    """
    Given the target list of a dataset, returns the indexes of patterns
    belonging to classes listed in the search_elements parameter.

    :param sequence: The list of pattern targets, as a list.
    :param search_elements: A list of classes used to filter the dataset
        patterns. Patterns belonging to one of those classes will be included.
        If None, all patterns will be included.
    :param bucket_classes: If True, pattern indexes will be returned so that
        patterns will be grouped by class. Defaults to True.
    :param sort_classes: If both ``bucket_classes`` and ``sort_classes`` are
        True, class groups will be sorted by class index. Ignored if
        ``bucket_classes`` is False. Defaults to False.
    :param sort_indexes: If True, patterns indexes will be sorted. When
        bucketing by class, patterns will be sorted inside their buckets.
        Defaults to False.

    :returns: The indexes of patterns belonging to the required classes,
        as a list. Can return None, which means that the original pattern
        sequence already satisfies all the constraints.
    """
    if bucket_classes:
        return _indexes_grouped_by_classes(
            sequence,
            search_elements,
            sort_indexes=sort_indexes,
            sort_classes=sort_classes,
        )

    return _indexes_without_grouping(
        sequence, search_elements, sort_indexes=sort_indexes
    )


def make_nc_transformation_subset(
    dataset: SupportedDataset,
    transform: Any,
    target_transform: Any,
    classes: Union[None, Sequence[int]],
    bucket_classes: bool = False,
    sort_classes: bool = False,
    sort_indexes: bool = False,
) -> TaskAwareClassificationDataset:
    """
    Creates a subset given the list of classes the patterns should belong to.

    :param dataset: The original dataset
    :param transform: The transform function for patterns. Can be None.
    :param target_transform: The transform function for targets. Can be None.
    :param classes: A list of classes used to filter the dataset patterns.
        Patterns belonging to one of those classes will be included. If None,
        all patterns will be included.
    :param bucket_classes: If True, the final Dataset will output patterns by
        grouping them by class. Defaults to True.
    :param sort_classes: If ``bucket_classes`` and ``sort_classes`` are both
        True, the final Dataset will output patterns by grouping them by class
        and the class groups will be ordered by class ID (ascending). Ignored
        if ``bucket_classes`` is False. Defaults to False.
    :param sort_indexes: If True, pattern indexes will be sorted (ascending).
        When grouping by class, patterns will be sorted inside their respective
        class buckets. Defaults to False.

    :returns: A :class:`TransformationSubset` that includes only patterns
        belonging to the given classes, in the order controlled by the
        ``bucket_classes``, ``sort_classes`` and ``sort_indexes`` parameters.
    """
    return _taskaware_classification_subset(
        dataset,
        indices=_indexes_from_set(
            getattr(dataset, "targets"),
            classes,
            bucket_classes=bucket_classes,
            sort_classes=sort_classes,
            sort_indexes=sort_indexes,
        ),
        transform=transform,
        target_transform=target_transform,
    )


__all__ = ["make_nc_transformation_subset"]
