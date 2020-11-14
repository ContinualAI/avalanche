################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

from collections import OrderedDict
from typing import List, Sequence, Dict, Any, Union

import torch

from avalanche.training.utils.transform_dataset import TransformationSubset, \
    IDatasetWithTargets
from avalanche.benchmarks.utils import tensor_as_list


def _indexes_grouped_by_classes(targets: Sequence[int],
                                patterns_indexes: Union[None, Sequence[int]],
                                sort_indexes: bool = True,
                                sort_classes: bool = True) \
        -> Union[List[int], None]:
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
    # will initialize "result_per_class" in sorted order ->
    # -> patterns will be ordered by ascending class ID
    classes = torch.unique(torch.as_tensor(targets),
                           sorted=sort_classes).tolist()

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
        return None

    return result


def _indexes_from_set(targets: Sequence[int],
                      patterns_indexes: Union[None, Sequence[int]],
                      bucket_classes: bool = True, sort_classes: bool = False,
                      sort_indexes: bool = False) -> Union[List[int], None]:
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
        return _indexes_grouped_by_classes(targets, patterns_indexes,
                                           sort_indexes=sort_indexes,
                                           sort_classes=sort_classes)

    if patterns_indexes is None:
        return None
    if not sort_indexes:
        return tensor_as_list(patterns_indexes)

    patterns_indexes = tensor_as_list(patterns_indexes)
    result = list(patterns_indexes)
    result.sort()
    return result


def make_ni_transformation_subset(dataset: IDatasetWithTargets,
                                  transform: Any, target_transform: Any,
                                  patterns_indexes: Union[None, Sequence[int]],
                                  bucket_classes: bool = False,
                                  sort_classes: bool = False,
                                  sort_indexes: bool = False) \
        -> TransformationSubset:
    """
    Creates a subset given the list of patterns to include.

    :param dataset: The original dataset
    :param transform: The transform function for patterns. Can be None.
    :param target_transform: The transform function for targets. Can be None.
    :param patterns_indexes: A list of indexes of patterns to include.
        If None, all patterns will be included.
    :param bucket_classes: If True, the final Dataset will output patterns by
        grouping them by class. Defaults to True.
    :param sort_classes: If ``bucket_classes`` and ``sort_classes`` are both
        True, the final Dataset will output patterns by grouping them by class
        and the class groups will be ordered by class ID (ascending). Ignored
        if ``bucket_classes`` is False. Defaults to False.
    :param sort_indexes: If True, pattern indexes will be sorted (ascending).
        When grouping by class, patterns will be sorted inside their respective
        class buckets. Defaults to False.

    :returns: A :class:`TransformationSubset` that includes only the required
        patterns, in the order controlled by the ``bucket_classes``,
        ``sort_classes`` and ``sort_indexes`` parameters.
    """
    return TransformationSubset(dataset,
                                _indexes_from_set(dataset.targets,
                                                  patterns_indexes,
                                                  bucket_classes=bucket_classes,
                                                  sort_classes=sort_classes,
                                                  sort_indexes=sort_indexes),
                                transform=transform,
                                target_transform=target_transform)


__all__ = ['make_ni_transformation_subset']
