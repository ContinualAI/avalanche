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
This module contains the implementation of the DataAttribute,
a class designed to managed task and class labels. DataAttributes allow fast
concatenation and subsampling operations and are automatically managed by
AvalancheDatasets.
"""

import torch

from .dataset_definitions import IDataset
from .flat_data import ConstantSequence, FlatData


class DataAttribute:
    """Data attributes manage sample-wise information such as task or
    class labels.

    It provides access to unique values (`self.uniques`) and their indices
    (`self.val_to_idx`). Both fields are initialized lazily.

    Data attributes can be efficiently concatenated and subsampled.
    """

    def __init__(self, data: IDataset, name: str = None, use_in_getitem=False):
        """Data Attribute.

        :param data: a sequence of values, one for each sample.
        :param name: a name that uniquely identifies the attribute.
            It is used by `AvalancheDataset` to dynamically add it to its
            attributes.
        :param use_in_getitem: If True, `AvalancheDataset` will add
            the value at the end of each sample.
        """
        self.name = name
        self.use_in_getitem = use_in_getitem

        self._data = self._normalize_sequence(data)

        self._uniques = None  # set()
        self._val_to_idx = None  # dict()
        self._count = None  # dict()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return str(self.data[:])

    def __str__(self):
        return str(self.data[:])

    @property
    def data(self):
        return self._data

    @property
    def uniques(self):
        """Set of unique values in the attribute."""
        if self._uniques is None:
            self._uniques = set()
            # init. uniques with fast paths for special cases
            if isinstance(self.data, ConstantSequence):
                self.uniques.add(self.data[0])
            elif isinstance(self.data, DataAttribute):
                self.uniques.update(self.data.uniques)
            else:
                for el in self.data:
                    self.uniques.add(el)
        return self._uniques

    @property
    def count(self):
        """Dictionary of value -> count."""
        if self._count is None:
            self._count = {}
            for val in self.uniques:
                self._count[val] = 0
            for val in self.data:
                self._count[val] += 1
        return self._count

    @property
    def val_to_idx(self):
        """Dictionary mapping unique values to indices."""
        if self._val_to_idx is None:
            # init. val-to-idx
            self._val_to_idx = dict()
            if isinstance(self.data, ConstantSequence):
                self._val_to_idx = {self.data[0]: range(len(self.data))}
            else:
                for i, x in enumerate(self.data):
                    if x not in self.val_to_idx:
                        self._val_to_idx[x] = []
                    self._val_to_idx[x].append(i)
        return self._val_to_idx

    def subset(self, indices):
        """Subset operation.

        Return a new `DataAttribute` by keeping only the elements in `indices`.

        :param indices: position of the elements in the new subset
        :return: the new `DataAttribute`
        """
        return DataAttribute(
            self.data.subset(indices),
            self.name,
            use_in_getitem=self.use_in_getitem,
        )

    def concat(self, other: "DataAttribute"):
        """Concatenation operation.

        :param other: the other `DataAttribute`
        :return: the new concatenated `DataAttribute`
        """
        assert self.name == other.name, (
            "Cannot concatenate DataAttributes" + "with different names."
        )
        return DataAttribute(
            self.data.concat(other.data),
            self.name,
            use_in_getitem=self.use_in_getitem,
        )

    @staticmethod
    def _normalize_sequence(seq):
        if isinstance(seq, torch.Tensor):
            # equality doesn't work for tensors
            seq = seq.tolist()
        if not isinstance(seq, FlatData):
            return FlatData([seq])
        return seq


class TaskLabels(DataAttribute):
    """Task labels are `DataAttribute`s that are automatically appended to the
    mini-batch."""

    def __init__(self, task_labels):
        super().__init__(task_labels, "task_labels", use_in_getitem=True)


__all__ = ["DataAttribute", "TaskLabels"]
