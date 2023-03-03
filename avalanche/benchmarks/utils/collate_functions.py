################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 21-04-2022                                                             #
# Author(s): Antonio Carta, Lorenzo Pellegrini                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import itertools
from collections import defaultdict

import torch


def classification_collate_mbatches_fn(mbatches):
    """Combines multiple mini-batches together.

    Concatenates each tensor in the mini-batches along dimension 0 (usually
    this is the batch size).

    :param mbatches: sequence of mini-batches.
    :return: a single mini-batch
    """
    batch = []
    for i in range(len(mbatches[0])):
        t = classification_single_values_collate_fn(
            [el[i] for el in mbatches], i
        )
        batch.append(t)
    return batch


def classification_single_values_collate_fn(values_list, index):
    """
    Collate function used to merge the single elements (x or y or t,
    etcetera) of a minibatch of data from a classification dataset.

    This function assumes that all values are tensors of the same shape
    (excluding the first dimension).

    :param values_list: The list of values to merge.
    :param index: The index of the element. 0 for x values, 1 for y values,
        etcetera. In this implementation, this parameter is ignored.
    :return: The merged values.
    """
    return torch.cat(values_list, dim=0)


def detection_collate_fn(batch):
    """
    Collate function used when loading detection datasets using a DataLoader.

    This will merge the single samples of a batch to create a minibatch.
    This collate function follows the torchvision format for detection tasks.
    """
    return tuple(zip(*batch))


def detection_collate_mbatches_fn(mbatches):
    """
    Collate function used when loading detection datasets using a DataLoader.

    This will merge multiple batches to create a concatenated batch.

    Beware that merging multiple batches is different from creating a batch
    from single dataset elements: Batches can be created from a
    list of single dataset elements by using :func:`detection_collate_fn`.
    """
    lists_dict = defaultdict(list)
    for mb in mbatches:
        for mb_elem_idx, mb_elem in enumerate(mb):
            lists_dict[mb_elem_idx].append(mb_elem)

    lists = []
    for mb_elem_idx in range(max(lists_dict.keys()) + 1):
        lists.append(
            list(itertools.chain.from_iterable(lists_dict[mb_elem_idx]))
        )

    return lists


__all__ = [
    "classification_collate_mbatches_fn",
    "classification_single_values_collate_fn",
    "detection_collate_fn",
    "detection_collate_mbatches_fn",
]
