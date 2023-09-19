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
# TODO: module doc
import warnings
from copy import copy
from typing import Protocol, List

from .generic_scenario import CLScenario, CLStream, EagerCLStream, CLExperience


class TaskAware(Protocol):
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


# TODO: doc, test
# TODO: respect stream generators. Should return a new generators which applies
#  foo_decorate_exp every time a new experience is generated.
def _decorate_generic(obj, exp_decorator):
    """Call `exp_decorator` on each experience in `obj`.

    `obj` can be a scenario, stream, or a single experience.

    `exp_decorator` is a decorator method that adds the desired attributes.

    streams must be eager! internal use only.
    `exp_decorator` will receive a copy of the experience.
    """
    # IMPLEMENTATION NOTE: first, we check the type of `obj`. Then, for
    # benchmarks and streams we call `exp_decorator` on each experience.
    def _decorate_exp(obj, exp_decorator):
        return exp_decorator(copy(obj))

    def _decorate_benchmark(obj, exp_decorator):
        new_streams = []
        for s in obj.streams.values():
            new_streams.append(_decorate_stream(s, exp_decorator))
        return CLScenario(new_streams)

    def _decorate_stream(obj, exp_decorator):
        new_stream = []
        if not isinstance(obj, EagerCLStream):
            warnings.warn("stream generators will be converted to a list.")
        for exp in obj:
            new_stream.append(_decorate_exp(exp, exp_decorator))
        return EagerCLStream(obj.name, new_stream)

    if isinstance(obj, CLScenario):
        return _decorate_benchmark(obj, exp_decorator)
    elif isinstance(obj, CLStream):
        return _decorate_stream(obj, exp_decorator)
    elif isinstance(obj, CLExperience):
        return _decorate_exp(obj, exp_decorator)
    else:
        raise ValueError("Unsupported object type: must be one of {CLScenario, CLStream, CLExperience}")


def with_task_labels(obj):
    """Add `TaskAware` attributes.

    The dataset must already have task labels.

    `obj` must be a scenario, stream, or experience.
    """
    # TODO: doc, test
    # TODO: if data doesn't have task labels, add them (possibly under flag)
    def _add_task_labels(exp):
        tls = exp.dataset.targets_task_labels.uniques
        if len(tls) == 1:
            # tls is a set. we need to convert to list to call __getitem__
            exp.task_label = list(tls)[0]
        exp.task_labels = tls
        return exp

    return _decorate_generic(obj, _add_task_labels)
