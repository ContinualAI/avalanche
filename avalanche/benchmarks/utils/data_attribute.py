import bisect
from typing import Union, Sequence

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset

from avalanche.benchmarks.utils.dataset_utils import ConstantSequence


class DataAttribute:
    """Data attributes manage sample-wise information such as task or
    class labels."""

    def __init__(self, name: str, info: Union[Sequence, ConstantSequence],
                 append_to_mbatch=False):
        self.name = name
        self.info = info
        self.append_to_mbatch = append_to_mbatch

        self._optimize_sequence()
        self.uniques = set()
        self.val_to_idx = dict()

        if len(self) == 0:
            return

        # init. uniques
        if isinstance(self.info, ConstantSequence):
            self.uniques.add(self.info[0])
        else:
            for el in self:
                self.uniques.add(el)

        # init. val-to-idx
        if isinstance(self.info, ConstantSequence):
            self.val_to_idx = {self.info[0]: range(len(self.info))}
        else:
            for i, x in enumerate(self):
                if x not in self.val_to_idx:
                    self.val_to_idx[x] = []
                self.val_to_idx[x].append(i)

    def subset(self, indices):
        if isinstance(self.info, ConstantSequence):
            new_info = ConstantSequence(self.info[0], len(indices))
        else:
            new_info = self.info[indices]
        return DataAttribute(self.name, new_info)

    def cat(self, other: "DataAttribute"):
        assert self.name == other.name, "Cannot concatenate DataAttributes" + \
                                        "with different names."
        ta = torch.tensor(self.info)
        tb = torch.tensor(other.info)
        return DataAttribute(ta.name, torch.cat([ta, tb], dim=0))

    def _optimize_sequence(self):
        if len(self.info) == 0 or isinstance(self.info, ConstantSequence):
            return
        if isinstance(self.info, list):
            return
        return list(self.info)

    def __getitem__(self, item):
        return self.info[item]

    def __len__(self):
        return len(self.info)


class DataAttributeSubSet(DataAttribute):
    """Data attributes manage sample-wise information such as task or
    class labels."""

    def __init__(self, info: DataAttribute, indices: Sequence[int]):
        self.indices = indices

        super().__init__(
            info.name,
            info.info,
            info.append_to_mbatch
        )

    def __getitem__(self, item):
        return super().__getitem__(self.indices[item])

    def __len__(self):
        return len(self.indices)


class TaskLabels(DataAttribute):
    def __init__(self, task_labels):
        super().__init__('task_labels', task_labels)
