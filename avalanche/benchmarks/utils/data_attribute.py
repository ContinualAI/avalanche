from typing import Union

from torch import Tensor

from avalanche.benchmarks.utils.dataset_utils import ConstantSequence


class DataAttribute:
    """Data attributes manage sample-wise information such as task or
    class labels.
    """
    def __init__(self, name: str, info: Union[Tensor, ConstantSequence]):
        self.name = name
        self.info = info
        self._optimize_sequence()
        self.uniques = set()
        self.task_set = dict()

        if len(self.info) == 0:
            pass

        # init. uniques
        if isinstance(self.info, ConstantSequence):
            self.uniques.add(self.info[0])
        else:
            for el in self.info:
                self.uniques.add(el)

        # init. val-to-idx
        if isinstance(self.info, ConstantSequence):
            self.task_set = {self.info[0]: range(len(self.info))}
        else:
            for i, x in enumerate(self.info):
                if x not in self.task_set:
                    self.task_set[x] = []
                self.task_set[x].append(i)

    def _optimize_sequence(self):
        if len(self.info) == 0 or isinstance(self.info, ConstantSequence):
            return
        if isinstance(self.info, list):
            return
        return list(self.info)

    def __getitem__(self, item):
        return self.info[item]


class TaskLabels(DataAttribute):
    def __init__(self, task_labels):
        super().__init__('task_labels', task_labels)
