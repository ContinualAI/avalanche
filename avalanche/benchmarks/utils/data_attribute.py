import bisect
from typing import Union, Sequence

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset

from .dataset_utils import ConstantSequence
from .flattened_data import FlatData


class DataAttribute(FlatData):
    """Data attributes manage sample-wise information such as task or
    class labels."""

    def __init__(self, name: str, data: Union[Sequence, ConstantSequence]):
        self.name = name

        data = self._optimize_sequence(data)
        super().__init__(data)

        self.uniques = set()
        self.val_to_idx = dict()

        if len(self) == 0:
            return

        # init. uniques
        if isinstance(self.data, ConstantSequence):
            self.uniques.add(self.data[0])
        else:
            for el in self:
                self.uniques.add(el)

        # init. val-to-idx
        if isinstance(self.data, ConstantSequence):
            self.val_to_idx = {self.data[0]: range(len(self.data))}
        else:
            for i, x in enumerate(self):
                if x not in self.val_to_idx:
                    self.val_to_idx[x] = []
                self.val_to_idx[x].append(i)

    def subset(self, indices):
        if isinstance(self.data, ConstantSequence):
            new_info = ConstantSequence(self.data[0], len(indices))
        elif isinstance(self.data, list):
            new_info = [self.data[idx] for idx in indices]
        else:
            new_info = self.data[indices]
        return DataAttribute(self.name, new_info)

    def concat(self, other: "DataAttribute"):
        assert self.name == other.name, "Cannot concatenate DataAttributes" + \
                                        "with different names."
        ta = torch.tensor(self.data)
        tb = torch.tensor(other.data)
        return DataAttribute(ta.name, torch.cat([ta, tb], dim=0))

    @staticmethod
    def _optimize_sequence(seq):
        if len(seq) == 0 or isinstance(seq, ConstantSequence):
            return seq
        if isinstance(seq, list):
            return seq
        return list(seq)


class TaskLabels(DataAttribute):
    def __init__(self, task_labels):
        super().__init__('task_labels', task_labels)
