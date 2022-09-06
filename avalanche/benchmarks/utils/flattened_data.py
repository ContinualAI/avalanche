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
from typing import Sequence


class FlatData(Sequence):
    """FlatData is an efficient dataset optimized for repeated concatenations
    and subset operations.

    Class for internal use only.
    """

    def __init__(self, data: Sequence):
        self._data = data

    def get_data(self):
        return self._data

    def subset(self, indices):
        if isinstance(self.get_data(), FlatData):
            return self.get_data().subset(indices)
        else:
            return _FlatDataSubset(self.get_data(), indices)

    def concat(self, other: "FlatData"):
        if isinstance(self.get_data(), FlatData):
            return self.get_data().concat(other)
        else:
            return _FlatDataConcat([self, other])

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)


class _FlatDataSubset(FlatData):
    """FlatData is a pytorch-like Dataset optimized to minimize its tree
    depth.

    Class for internal use only.
    """

    def __init__(self, data: Sequence, indices: Sequence[int]):
        self._indices = list(indices)
        super().__init__(data)

    def subset(self, indices):
        return _FlatDataSubset(self.get_data(),
                               [self._indices[i] for i in indices])

    def concat(self, other: "FlatData"):
        if isinstance(other, _FlatDataSubset):
            if other.get_data() is self.get_data():
                return _FlatDataSubset(self.get_data(),
                                       self._indices + other._indices)
        return super().concat(other)

    def __getitem__(self, item):
        return super().__getitem__(self._indices[item])

    def __len__(self):
        return len(self._indices)


class _FlatDataConcat(FlatData):
    """FlatData is a pytorch-like Dataset optimized to minimize its tree
    depth.

    Class for internal use only.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datas: Sequence[Sequence]):
        assert len(datas) > 0, 'datasets should not be an empty iterable'
        self._datasets = list(datas)
        self.cumulative_sizes = self.cumsum(self._datasets)

    def get_data(self):
        return self

    def concat(self, other: "FlatData"):
        if isinstance(other, _FlatDataConcat):
            return _FlatDataConcat(self._datasets + other._datasets)
        return super().concat(other)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self._datasets[dataset_idx][sample_idx]


class ConstantSequence(FlatData):
    """A memory-efficient constant sequence."""

    def __init__(self, constant_value: int, size: int):
        self._constant_value = constant_value
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, item_idx) -> int:
        if item_idx >= len(self):
            raise IndexError()
        return self._constant_value

    def get_data(self):
        return self

    def subset(self, indices):
        return ConstantSequence(self._data[0], len(indices))

    def concat(self, other: "FlatData"):
        if isinstance(other, ConstantSequence) and \
                self._constant_value == other._constant_value:
            return ConstantSequence(self._constant_value, len(self) + len(other))
        else:
            return _FlatDataConcat([self, other])

    def __str__(self):
        return (
            "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"
        )