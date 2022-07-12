import bisect
from typing import Union, Sequence

import torch
from torch.utils.data import Dataset

from avalanche.benchmarks.utils.dataset_utils import ConstantSequence


class FlatData:
    """FlatData is a pytorch-like Dataset optimized to minimize its tree
    depth."""

    def __init__(self, data: Dataset):
        self.data = data

    def subset(self, indices):
        if isinstance(self.data, ConstantSequence):
            new_info = ConstantSequence(self.data[0], len(indices))
        else:
            new_info = self.data[indices]
        return FlatData(new_info)

    def concat(self, other: "FlatData"):
        ta = torch.tensor(self.data)
        tb = torch.tensor(other.data)
        return FlatDataConcat(ta.name, torch.cat([ta, tb], dim=0))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class FlatDataSubset(FlatData):
    """FlatData is a pytorch-like Dataset optimized to minimize its tree
    depth."""

    def __init__(self, data: Dataset, indices: Sequence[int]):
        self.indices = indices
        super().__init__(data)

    def __getitem__(self, item):
        return super().__getitem__(self.indices[item])

    def __len__(self):
        return len(self.indices)


class FlatDataConcat(FlatData):
    """FlatData is a pytorch-like Dataset optimized to minimize its tree
    depth."""

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datas: Sequence[Dataset]):
        self.datasets = list(datas)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        self.cumulative_sizes = self.cumsum(self.datasets)

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
        return self.datasets[dataset_idx][sample_idx]
