import torch

from .dataset_definitions import IDataset
from .dataset_utils import ConstantSequence
from .flattened_data import _FlatDataSubset, _FlatDataConcat


class DataAttribute:
    """Data attributes manage sample-wise information such as task or
    class labels.

    provides access to unique values (`self.uniques`) and their indices (`self.val_to_idx`).
    Both fields are initialized lazily.
    """

    def __init__(self, name: str, data: IDataset):
        self.name = name
        self._data = self._optimize_sequence(data)

        self._uniques = None  # set()
        self._val_to_idx = None  # dict()

        if len(data) == 0:
            return

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    @property
    def uniques(self):
        """Set of unique values in the attribute."""
        if self._uniques is None:
            self._uniques = set()
            # init. uniques with fast paths for special cases
            if isinstance(self._data, ConstantSequence):
                self.uniques.add(self._data[0])
            elif isinstance(self._data, DataAttribute):
                self.uniques.update(self._data.uniques)
            else:
                for el in self._data:
                    self.uniques.add(el)
        return self._uniques

    @property
    def val_to_idx(self):
        """Dictionary mapping unique values to indices."""
        if self._val_to_idx is None:
            # init. val-to-idx
            self._val_to_idx = dict()
            if isinstance(self._data, ConstantSequence):
                self._val_to_idx = {self._data[0]: range(len(self._data))}
            else:
                for i, x in enumerate(self._data):
                    if x not in self.val_to_idx:
                        self._val_to_idx[x] = []
                    self._val_to_idx[x].append(i)
        return self._val_to_idx

    def subset(self, indices):
        return DataAttribute(self.name, _FlatDataSubset(self._data, indices))

    def concat(self, other: "DataAttribute"):
        assert self.name == other.name, "Cannot concatenate DataAttributes" + \
                                        "with different names."
        return DataAttribute(self.name, _FlatDataConcat([self._data, other._data]))

    @staticmethod
    def _optimize_sequence(seq):
        if isinstance(seq, torch.Tensor):
            # equality doesn't work for tensors
            return list(seq)
        return seq


class TaskLabels(DataAttribute):
    def __init__(self, task_labels):
        super().__init__('task_labels', task_labels)
