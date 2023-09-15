################################################################################
# Copyright (c) 2023 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-09-2023                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from typing import Protocol, List


class TaskAwareExperience(Protocol):
    """Task-aware experiences provide task labels.

    The attribute `task_label` is available is an experience has data from
    a single task. Otherwise, `task_labels` must be used, which provides the
    list of task labels for the current experience.
    """

    @property
    def task_label(self) -> int:
        """The experience task label.

        This attribute is accessible only if the experience contains a single
        task. It will raise an error for multi-task experiences.
        """
        return 0

    @property
    def task_labels(self) -> List[int]:
        """The list of task labels in the experience."""
        return [0]
