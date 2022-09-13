################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-07-2022                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
    Datasets with optimized concat/subset operations.
"""
import bisect
from typing import Sequence, List

from torch.utils.data import ConcatDataset

from avalanche.benchmarks.utils.dataset_definitions import IDataset


class FlatData(IDataset):
    """FlatData is a dataset optimized for efficient repeated concatenation
    and subset operations.

    Class for internal use only.
    """

    def __init__(self, datasets: Sequence[IDataset],
                 indices: List[int] = None, can_flatten=True):
        self._datasets = datasets
        if can_flatten:
            self._datasets = _flatten_dataset_list(self._datasets)

        self._indices = indices
        self._cumulative_sizes = ConcatDataset.cumsum(self._datasets)
        self._can_flatten = can_flatten
        if indices is not None and can_flatten:
            pass  # flatten indices

    def subset(self, indices):
        assert len(indices) == len(self)
        if self._can_flatten:
            new_indices = [self._indices[x] for x in indices]
            self.__class__(self._datasets, new_indices)
        return self.__class__([self], indices)

    def concat(self, other: "FlatData"):
        if self._can_flatten and other._can_flatten:
            return self.__class__(
                self._datasets + other._datasets,
                self._indices + [len(self) + idx for idx in other._indices])
        elif self._can_flatten:
            return self.__class__(
                self._datasets + [other],
                self._indices + [len(self) + idx for idx in other._indices])
        elif other._can_flatten:
            return self.__class__(
                [self] + other._datasets,
                list(range(len(self))) + [len(self) + idx for idx in other._indices])
        return self.__class__([self, other])

    def _get_idx(self, idx):
        if self._indices is not None:  # subset indexing
            idx = self._indices[idx]
        if len(self._datasets) == 1:
            dataset_idx = 1
        else:  # concat indexing
            dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
            if dataset_idx == 0:
                idx = idx
            else:
                idx = idx - self._cumulative_sizes[dataset_idx - 1]
        return dataset_idx, idx

    def __getitem__(self, idx):
        dataset_idx, idx = self._get_idx(idx)
        return self._datasets[dataset_idx][idx]

    def __len__(self):
        if len(self._cumulative_sizes) == 0:
            return 0
        return self._cumulative_sizes[-1]

    def __add__(self, other: "FlatData") -> "FlatData":
        return self.concat(other)

    def __radd__(self, other: "FlatData") -> "FlatData":
        return other.concat(self)


class ConstantSequence(FlatData):
    """A memory-efficient constant sequence."""

    def __init__(self, constant_value: int, size: int):
        self._constant_value = constant_value
        self._size = size
        super().__init__([self])

    def __len__(self):
        return self._size

    def __getitem__(self, item_idx) -> int:
        if item_idx >= len(self):
            raise IndexError()
        return self._constant_value

    def subset(self, indices):
        return ConstantSequence(self._constant_value, len(indices))

    def concat(self, other: "FlatData"):
        if isinstance(other, ConstantSequence) and \
                self._constant_value == other._constant_value:
            return ConstantSequence(self._constant_value, len(self) + len(other))
        else:
            return super().concat(other)

    def __str__(self):
        return f"ConstantSequence(value={self._constant_value}, len={self._size}"


def _flatten_dataset_list(datasets: List[FlatData]):
    """Flatten dataset tree if possible."""
    # Concat -> Concat branch
    # Flattens by borrowing the list of concatenated datasets
    # from the original datasets.
    flattened_list = []
    for dataset in datasets:
        if len(dataset) == 0:
            continue
        elif isinstance(dataset, FlatData) and dataset._can_flatten:
            flattened_list.extend(dataset._datasets)
        else:
            flattened_list.append(dataset)

    # merge consecutive Subsets if compatible
    new_data_list = []
    for dataset in flattened_list:
        if isinstance(dataset, FlatData) and len(new_data_list) > 0 and \
                isinstance(new_data_list[-1], FlatData):
            merged_ds = _maybe_merge_subsets(new_data_list.pop(), dataset)
            new_data_list.extend(merged_ds)
        else:
            new_data_list.append(dataset)
    return new_data_list


def _maybe_merge_subsets(d1: FlatData, d2: FlatData):
    if (not d1._can_flatten) or (not d2._can_flatten):
        return [d1, d2]

    if len(d1._datasets) == 1 and len(d2._datasets) == 1 and d1._datasets[0] is d2._datasets[0]:
        return [d1.__class__(d1._datasets, d1._indices + d2._indices)]

    return [d1, d2]
