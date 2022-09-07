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

from avalanche.benchmarks.utils.dataset_definitions import IDataset


class FlatData(IDataset):
    """FlatData is an efficient dataset optimized for repeated concatenations
    and subset operations.

    Class for internal use only.
    """

    def __init__(self, data: IDataset):
        # IMPLEMENTATION NOTE: if you make a subclass remember to check
        # _can_flatten and _can_merge.
        self._data = data

    def _can_flatten(self):
        """Private method used to check if the dataset can be flattened.

        Child classes of FlatData can override this method to disable flattening
        whenever is necessary, for example, to avoid removing transformations.
        """
        return True

    def _can_merge(self, other):
        """Private method used to check if the dataset can be merged.

        Child classes of FlatData can override this method to disable merging
        whenever is necessary, for example, to avoid removing transformations.

        :param other:
        :return:
        """
        return False

    def get_data(self):
        return self._data

    def subset(self, indices):
        if isinstance(self.get_data(), FlatData) and self._can_flatten():
            return self.get_data().subset(indices)
        else:
            return _FlatDataSubset(self.get_data(), indices)

    def concat(self, other: "FlatData"):
        if isinstance(self.get_data(), FlatData) and self._can_flatten():
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

    def __init__(self, data: IDataset, indices: Sequence[int]):
        self._indices = list(indices)
        super().__init__(data)

    def _can_merge(self, other: FlatData):
        return self.get_data() is other.get_data()

    def subset(self, indices):
        if not self._can_flatten():
            return _FlatDataSubset(self.get_data(), indices)
        else:
            mapped_idxs = [self._indices[i] for i in indices]
            return _FlatDataSubset(self.get_data(), mapped_idxs)

    def concat(self, other: "FlatData"):
        if isinstance(other, _FlatDataSubset) and \
                self._can_flatten() and other._can_flatten() and \
                self._can_merge(other) and other._can_merge(self):
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

    def __init__(self, datas: Sequence[IDataset]):
        assert len(datas) > 0, 'datasets should not be an empty iterable'
        self._datasets = list(datas)
        self.cumulative_sizes = self.cumsum(self._datasets)

    def _can_merge(self, other):
        return True

    def get_data(self):
        return self

    def subset(self, indices):
        if not self._can_flatten():
            return _FlatDataSubset(self, indices)

        # flatten Subset -> Concat -> Subset branches
        # before creating the new dataset
        new_data_list = []
        start_idx = 0
        for k, curr_data in enumerate(self._datasets):
            if not isinstance(curr_data, FlatData):
                new_data_list.append(curr_data)
            else:
                # find permutation indices for current dataset in the concat list
                end_idx = self.cumulative_sizes[k]
                curr_idxs = indices[start_idx:end_idx]
                curr_idxs = [el - start_idx for el in curr_idxs]
                start_idx = end_idx

                if len(curr_idxs) > 0:
                    # we have a recursive call here because
                    # if the curr_data is a subset, we can flatten it
                    new_data_list.append(curr_data.subset(curr_idxs))

        # make a new ConcatDataset with the new data_list
        # attributes and transform are the same as the original
        return _FlatDataConcat(new_data_list)

    def concat(self, other: "FlatData"):
        if isinstance(other, _FlatDataConcat):
            if self._can_flatten() and other._can_flatten():
                return _FlatDataConcat(self._datasets + other._datasets)
        else:
            if self._can_flatten():
                return _FlatDataConcat(self._datasets + [other])
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
        if self._can_flatten() and other._can_flatten() and \
                self._can_merge(other) and other._can_merge(self) and \
                isinstance(other, ConstantSequence) and \
                self._constant_value == other._constant_value:
            return ConstantSequence(self._constant_value, len(self) + len(other))
        else:
            return super().concat(other)

    def __str__(self):
        return (
            "[" + ", ".join([str(self[idx]) for idx in range(len(self))]) + "]"
        )