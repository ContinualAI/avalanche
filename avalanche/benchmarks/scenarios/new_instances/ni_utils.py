################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

from typing import Sequence

import torch

from avalanche.benchmarks.utils import IDatasetWithTargets


def _step_structure_from_assignment(dataset: IDatasetWithTargets,
                                    assignment: Sequence[Sequence[int]],
                                    n_classes: int):
    n_steps = len(assignment)
    step_structure = [[0 for _ in range(n_classes)] for _ in range(n_steps)]

    for step_id in range(n_steps):
        step_targets = [int(dataset.targets[pattern_idx])
                        for pattern_idx in assignment[step_id]]
        cls_ids, cls_counts = torch.unique(torch.as_tensor(
            step_targets), return_counts=True)

        for unique_idx in range(len(cls_ids)):
            step_structure[step_id][int(cls_ids[unique_idx])] += \
                int(cls_counts[unique_idx])

    return step_structure


__all__ = [
    '_step_structure_from_assignment'
]
