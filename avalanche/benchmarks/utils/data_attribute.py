from typing import Union, Dict

from torch import Tensor

from benchmarks.utils.data import AvalancheDataset
from benchmarks.utils.dataset_utils import ConstantSequence


class DataAttribute:
    """Data attributes manage sample-wise information such as task or
    class labels.

    """
    def __init__(self, info: Union[Tensor, ConstantSequence]):
        self.info = info
        self._optimize_sequence()
        self._make_task_set_dict()

    def _optimize_sequence(self):
        if len(self.info) == 0 or isinstance(self.info, ConstantSequence):
            return
        if isinstance(self.info, list):
            return

        return list(self.info)

    def _make_task_set_dict(self) -> Dict[int, "AvalancheDataset"]:
        task_dict = _TaskSubsetDict()
        for task_id in sorted(self.tasks_pattern_indices.keys()):
            task_indices = self.tasks_pattern_indices[task_id]
            task_dict[task_id] = (self, task_indices)

        return task_dict