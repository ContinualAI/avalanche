"""
    Datasets with optimized concat/subset operations.
"""
import bisect
from typing import Sequence

import torch

from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.benchmarks.utils.dataset_definitions import IDataset


class _FlatData:
    """FlatData is a pytorch-like Dataset optimized to minimize its tree
    depth.

    Class for internal use only.
    """

    def __init__(self, data: IDataset):
        self._data = data

    def get_data(self):
        return self._data

    def subset(self, indices):
        if isinstance(self.get_data(), ConstantSequence):
            # fast path for ConstantSequence
            new_info = ConstantSequence(self._data[0], len(indices))
            return _FlatData(new_info)
        else:
            return _FlatDataSubset(self.get_data(), indices)

    def concat(self, other: "_FlatData"):
        if isinstance(self.get_data(), ConstantSequence) and \
                isinstance(self.get_data(), ConstantSequence):
            # fast path for ConstantSequence
            new_info = ConstantSequence(self._data[0], len(self) + len(other))
            return _FlatData(new_info)
        else:
            return _FlatDataConcat([self, other])

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)


class _FlatDataSubset(_FlatData):
    """FlatData is a pytorch-like Dataset optimized to minimize its tree
    depth.

    Class for internal use only.
    """

    def __init__(self, data: IDataset, indices: Sequence[int]):
        self._indices = list(indices)
        super().__init__(data)

    def subset(self, indices):
        return _FlatDataSubset(self.get_data(),
                               [self._indices[i] for i in indices])

    def concat(self, other: "_FlatData"):
        if isinstance(other, _FlatDataSubset):
            if other.get_data() is self.get_data():
                return _FlatDataSubset(self.get_data(),
                                       self._indices + other._indices)
        return super().concat(other)

    def __getitem__(self, item):
        return super().__getitem__(self._indices[item])

    def __len__(self):
        return len(self._indices)


class _FlatDataConcat(_FlatData):
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

    def __init__(self, datas: Sequence[IDataset]):
        assert len(datas) > 0, 'datasets should not be an empty iterable'
        self._datasets = list(datas)
        self.cumulative_sizes = self.cumsum(self._datasets)

    def get_data(self):
        return self

    def concat(self, other: "_FlatData"):
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
