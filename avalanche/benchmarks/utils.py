#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

""" Common benchmarks/environments utils. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from collections import OrderedDict
from typing import List, Iterable, Sequence, Union, Dict

import numpy as np
import torch
from torch import Tensor


def remove_some_labels(dataset, labels_set, scale_labels=False):
    """ This method simply remove patterns with labels contained in
        the labels_set. """

    data, labels = dataset
    for label in labels_set:
        # Using fun below copies data
        mask = np.where(labels == label)[0]
        labels = np.delete(labels, mask)
        data = np.delete(data, mask, axis=0)

    if scale_labels:
        # scale labels if they do not start from zero
        min = np.min(labels)
        labels = (labels - min)

    return data, labels


def change_some_labels(dataset, labels_set, change_set):
    """ This method simply change labels contained in
        the labels_set. """

    data, labels = dataset
    for label, change in zip(labels_set, change_set):
        mask = np.where(labels == label)[0]
        labels = np.put(labels, mask, change)

    return data, labels


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
    # will initialize "result_per_class" in sorted order which in turn means
    # that patterns will be ordered by ascending class ID.
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
        targets: Sequence[int], patterns_indexes: Union[None, Sequence[int]],
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
