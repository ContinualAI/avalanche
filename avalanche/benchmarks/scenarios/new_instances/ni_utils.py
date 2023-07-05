################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import Sequence, Any
import torch
from avalanche.benchmarks.utils.dataset_definitions import (
    ISupportedClassificationDataset,
)


def _exp_structure_from_assignment(
    dataset: ISupportedClassificationDataset[Any],
    assignment: Sequence[Sequence[int]],
    n_classes: int,
):
    n_experiences = len(assignment)
    exp_structure = [[0 for _ in range(n_classes)] for _ in range(n_experiences)]

    for exp_id in range(n_experiences):
        exp_targets = [
            int(dataset.targets[pattern_idx]) for pattern_idx in assignment[exp_id]
        ]
        cls_ids, cls_counts = torch.unique(
            torch.as_tensor(exp_targets), return_counts=True
        )

        for unique_idx in range(len(cls_ids)):
            exp_structure[exp_id][int(cls_ids[unique_idx])] += int(
                cls_counts[unique_idx]
            )

    return exp_structure


__all__ = ["_exp_structure_from_assignment"]
