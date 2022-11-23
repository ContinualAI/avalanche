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
try:
    from collections import Hashable
except ImportError:
    from collections.abc import Hashable

from typing import List

from torch.utils.data import ConcatDataset

from avalanche.benchmarks.utils.dataset_definitions import IDataset


class FlatData(IDataset):
    """FlatData is a dataset optimized for efficient repeated concatenation
    and subset operations.

    The class combines concatentation and subsampling operations in a single
    class.

    Class for internal use only. Users shuold use `AvalancheDataset` for data
    or `DataAttribute` for attributes such as class and task labels.

    *Notes for subclassing*

    Cat/Sub operations are "flattened" if possible, which means that they will
    take the datasets and indices from their arguments and create a new dataset
    with them, avoiding the creation of large trees of dataset (what would
    happen with PyTorch datasets). Flattening is not always possible, for
    example if the data contains additional info (e.g. custom transformation
    groups), so subclasses MUST set `can_flatten` properly in order to avoid
    nasty bugs.
    """

    def __init__(
        self,
        datasets: List[IDataset],
        indices: List[int] = None,
        can_flatten=True,
    ):
        """Constructor

        :param datasets: list of datasets to concatenate.
        :param indices:  list of indices.
        :param can_flatten: if True, enables flattening.
        """
        self._datasets = datasets
        self._indices = indices
        self._can_flatten = can_flatten

        if can_flatten:
            self._datasets = _flatten_dataset_list(self._datasets)
            self._datasets, self._indices = _flatten_datasets_and_reindex(
                self._datasets, self._indices)
        self._cumulative_sizes = ConcatDataset.cumsum(self._datasets)

        # NOTE: check disabled to avoid slowing down OCL scenarios
        # # check indices
        # if self._indices is not None and len(self) > 0:
        #     assert min(self._indices) >= 0
        #     assert max(self._indices) < self._cumulative_sizes[-1], \
        #         f"Max index {max(self._indices)} is greater than datasets " \
        #         f"list size {self._cumulative_sizes[-1]}"

    def _get_indices(self):
        """This method creates indices on-the-fly if self._indices=None.
        Only for internal use. Call may be expensive if self._indices=None.
        """
        if self._indices is not None:
            return self._indices
        else:
            return list(range(len(self)))

    def subset(self, indices: List[int]) -> "FlatData":
        """Subsampling operation.

        :param indices: indices of the new samples
        :return:
        """
        if self._can_flatten:
            if self._indices is None:
                new_indices = indices
            else:
                self_indices = self._get_indices()
                new_indices = [self_indices[x] for x in indices]
            return self.__class__(datasets=self._datasets, indices=new_indices)
        return self.__class__(datasets=[self], indices=indices)

    def concat(self, other: "FlatData") -> "FlatData":
        """Concatenation operation.

        :param other: other dataset.
        :return:
        """
        if (not self._can_flatten) and (not other._can_flatten):
            return self.__class__(datasets=[self, other])

        # Case 1: one is a subset of the other
        if len(self._datasets) == 1 and len(other._datasets) == 1:
            if self._can_flatten and self._datasets[0] is other:
                return other.subset(self._indices + list(range(len(other))))
            elif other._can_flatten and other._datasets[0] is self:
                return self.subset(list(range(len(self))) + other._indices)
            elif (
                self._can_flatten
                and other._can_flatten
                and self._datasets[0] is other._datasets[0]
            ):
                idxs = self._get_indices() + other._get_indices()
                return self.__class__(datasets=self._datasets, indices=idxs)

        # Case 2: at least one of them can be flattened
        if self._can_flatten and other._can_flatten:
            if self._indices is None and other._indices is None:
                new_indices = None
            else:
                if len(self._cumulative_sizes) == 0:
                    base_other = 0
                else:
                    base_other = self._cumulative_sizes[-1]
                new_indices = self._get_indices() + [
                    base_other + idx for idx in other._get_indices()
                ]
            return self.__class__(
                datasets=self._datasets + other._datasets, indices=new_indices
            )
        elif self._can_flatten:
            if self._indices is None and other._indices is None:
                new_indices = None
            else:
                if len(self._cumulative_sizes) == 0:
                    base_other = 0
                else:
                    base_other = self._cumulative_sizes[-1]
                new_indices = self._get_indices() + [
                    base_other + idx for idx in range(len(other))
                ]
            return self.__class__(
                datasets=self._datasets + [other], indices=new_indices
            )
        elif other._can_flatten:
            if self._indices is None and other._indices is None:
                new_indices = None
            else:
                base_other = len(self)
                self_idxs = list(range(len(self)))
                new_indices = self_idxs + [
                    base_other + idx for idx in other._get_indices()
                ]
            return self.__class__(
                datasets=[self] + other._datasets, indices=new_indices
            )
        else:
            assert False, "should never get here"

    def _get_idx(self, idx):
        """Return the index as a tuple <dataset-idx, sample-idx>.

        The first index indicates the dataset to use from `self._datasets`,
        while the second is the index of the sample in
        `self._datasets[dataset-idx]`.

        Private method.
        """
        if self._indices is not None:  # subset indexing
            idx = self._indices[idx]
        if len(self._datasets) == 1:
            dataset_idx = 0
        else:  # concat indexing
            dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
            if dataset_idx == 0:
                idx = idx
            else:
                idx = idx - self._cumulative_sizes[dataset_idx - 1]
        return dataset_idx, int(idx)

    def __getitem__(self, idx):
        dataset_idx, idx = self._get_idx(idx)
        return self._datasets[dataset_idx][idx]

    def __len__(self):
        if len(self._cumulative_sizes) == 0:
            return 0
        elif self._indices is not None:
            return len(self._indices)
        return self._cumulative_sizes[-1]

    def __add__(self, other: "FlatData") -> "FlatData":
        return self.concat(other)

    def __radd__(self, other: "FlatData") -> "FlatData":
        return other.concat(self)

    def __hash__(self):
        return id(self)


class ConstantSequence:
    """A memory-efficient constant sequence."""

    def __init__(self, constant_value: int, size: int):
        """Constructor

        :param constant_value: the fixed value
        :param size: length of the sequence
        """
        self._constant_value = constant_value
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, item_idx) -> int:
        if item_idx >= len(self):
            raise IndexError()
        return self._constant_value

    def subset(self, indices: List[int]) -> "ConstantSequence":
        """Subset

        :param indices: indices of the new data.
        :return:
        """
        return ConstantSequence(self._constant_value, len(indices))

    def concat(self, other: "FlatData"):
        """Concatenation

        :param other: other dataset
        :return:
        """
        if (
            isinstance(other, ConstantSequence)
            and self._constant_value == other._constant_value
        ):
            return ConstantSequence(
                self._constant_value, len(self) + len(other)
            )
        else:
            return FlatData([self, other])

    def __str__(self):
        return (
            f"ConstantSequence(value={self._constant_value}, len={self._size}"
        )

    def __hash__(self):
        return id(self)


def _flatten_dataset_list(datasets: List[FlatData]):
    """Flatten the dataset tree if possible."""
    # Concat -> Concat branch
    # Flattens by borrowing the list of concatenated datasets
    # from the original datasets.
    flattened_list = []
    for dataset in datasets:
        if len(dataset) == 0:
            continue
        elif (
            isinstance(dataset, FlatData)
            and dataset._indices is None
            and dataset._can_flatten
        ):
            flattened_list.extend(dataset._datasets)
        else:
            flattened_list.append(dataset)

    # merge consecutive Subsets if compatible
    new_data_list = []
    for dataset in flattened_list:
        if (
            isinstance(dataset, FlatData)
            and len(new_data_list) > 0
            and isinstance(new_data_list[-1], FlatData)
        ):
            merged_ds = _maybe_merge_subsets(new_data_list.pop(), dataset)
            new_data_list.extend(merged_ds)
        else:
            new_data_list.append(dataset)
    return new_data_list


def _flatten_datasets_and_reindex(datasets, indices):
    """The same dataset may occurr multiple times in the list of datasets.

    Here, we flatten the list of datasets and fix the indices to account for
    the new dataset position in the list.
    """
    # find unique datasets
    if not all(isinstance(d, Hashable) for d in datasets):
        return datasets, indices

    dset_uniques = set(datasets)
    if len(dset_uniques) == len(datasets):  # no duplicates. Nothing to do.
        return datasets, indices

    # split the indices into <dataset-id, sample-id> pairs
    cumulative_sizes = [0] + ConcatDataset.cumsum(datasets)
    data_sample_pairs = []
    if indices is None:
        for ii, dset in enumerate(datasets):
            data_sample_pairs.extend([(ii, jj) for jj in range(len(dset))])
    else:
        for idx in indices:
            d_idx = bisect.bisect_right(cumulative_sizes, idx) - 1
            s_idx = idx - cumulative_sizes[d_idx]
            data_sample_pairs.append((d_idx, s_idx))

    # assign a new position in the list to each unique dataset
    new_datasets = list(dset_uniques)
    new_dpos = {d: i for i, d in enumerate(new_datasets)}
    new_cumsizes = [0] + ConcatDataset.cumsum(new_datasets)
    # reindex the indices to account for the new dataset position
    new_indices = []
    for d_idx, s_idx in data_sample_pairs:
        new_d_idx = new_dpos[datasets[d_idx]]
        new_indices.append(new_cumsizes[new_d_idx] + s_idx)

    # NOTE: check disabled to avoid slowing down OCL scenarios
    # if len(new_indices) > 0 and new_cumsizes[-1] > 0:
    #     assert min(new_indices) >= 0
    #     assert max(new_indices) < new_cumsizes[-1], \
    #         f"Max index {max(new_indices)} is greater than datasets " \
    #         f"list size {new_cumsizes[-1]}"
    return new_datasets, new_indices


def _maybe_merge_subsets(d1: FlatData, d2: FlatData):
    """Check the conditions for merging subsets."""
    if (not d1._can_flatten) or (not d2._can_flatten):
        return [d1, d2]

    if (
        len(d1._datasets) == 1
        and len(d2._datasets) == 1
        and d1._datasets[0] is d2._datasets[0]
    ):
        # return [d1.__class__(d1._datasets, d1._indices + d2._indices)]
        return d1.concat(d2)
    return [d1, d2]


def _flatdata_depth(dataset):
    """Internal debugging method.
    Returns the depth of the dataset tree."""
    if isinstance(dataset, FlatData):
        dchilds = [_flatdata_depth(dd) for dd in dataset._datasets]
        if len(dchilds) == 0:
            return 1
        return 1 + max(dchilds)
    else:
        return 1


def _flatdata_print(dataset, indent=0):
    """Internal debugging method.
    Print the dataset."""
    if isinstance(dataset, FlatData):
        ss = dataset._indices is not None
        cc = len(dataset._datasets) != 1
        cf = dataset._can_flatten
        print(
            "\t" * indent
            + f"{dataset.__class__.__name__} (len={len(dataset)},subset={ss},"
            f"cat={cc},cf={cf})"
        )
        for dd in dataset._datasets:
            _flatdata_print(dd, indent + 1)
    else:
        print(
            "\t" * indent + f"{dataset.__class__.__name__} (len={len(dataset)})"
        )


__all__ = ["FlatData", "ConstantSequence"]
