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
"""Task-aware scenario. Everything that provides task labels at each experience."""
import warnings
from copy import copy
from typing import Protocol, List, Sequence

from .generic_scenario import CLScenario, CLStream, EagerCLStream, CLExperience
from .dataset_scenario import DatasetExperience
from ..utils import AvalancheDataset, TaskLabels, ConstantSequence
from ..utils.data_attribute import has_task_labels


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


def _decorate_generic(obj, exp_decorator):
    """Call `exp_decorator` on each experience in `obj`.

    `obj` can be a scenario, stream, or a single experience.

    `exp_decorator` is a decorator method that adds the desired attributes.

    streams must be eager! internal use only.
    `exp_decorator` will receive a copy of the experience.
    """

    # TODO: respect stream generators. Should return a new generators which applies
    #  foo_decorate_exp every time a new experience is generated.
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
        raise ValueError(
            "Unsupported object type: must be one of {CLScenario, CLStream, CLExperience}"
        )


def with_task_labels(obj):
    """Add `TaskAware` attributes.

    The dataset must already have task labels.

    `obj` must be a scenario, stream, or experience.
    """

    def _add_task_labels(exp):
        tls = exp.dataset.targets_task_labels.uniques
        if len(tls) == 1:
            # tls is a set. we need to convert to list to call __getitem__
            exp.task_label = list(tls)[0]
        exp.task_labels = tls
        return exp

    return _decorate_generic(obj, _add_task_labels)


def task_incremental_benchmark(bm: CLScenario, reset_task_labels=False) -> CLScenario:
    """Creates a task-incremental benchmark from a dataset scenario.

    Adds progressive task labels to each stream (experience $i$ has task label $i$).
    Task labels are also added to each `AvalancheDataset` and will be returned by the `__getitem__`.
    For example, if your datasets have `<x, y>` samples (input, class),
    the new datasets will return `<x, y, t>` triplets, where `t` is the task label.

    Example of usage - SplitMNIST with task labels::

        bm = SplitMNIST(2)  # create class-incremental splits
        bm = task_incremental_benchmark(bm)  # adds task labels to the benchmark

    If `reset_task_labels is False` (default) the datasets *must not have task labels
    already set*. If the dataset have task labels, use::

        with_task_labels(benchmark_from_datasets(**dataset_streams)

    :param **dataset_streams: keys are stream names, values are list of datasets.
    :param reset_task_labels: whether existing task labels should be ignored.
        If False (default) if any dataset has task labels the function will raise
        a ValueError. If `True`, it will reset task labels.

    :return: a CLScenario in the task-incremental setting.
    """
    # TODO: when/how to do label_remapping?

    streams = []
    for name, stream in bm.streams.items():
        new_stream = []
        for eid, exp in enumerate(stream):
            if has_task_labels(exp.dataset) and (not reset_task_labels):
                raise ValueError(
                    "AvalancheDataset already has task labels. Use `benchmark_from_datasets` "
                    "instead or set `reset_task_labels=True`."
                )
            tls = TaskLabels(ConstantSequence(eid, len(exp.dataset)))
            new_dd = exp.dataset.update_data_attribute(
                name="targets_task_labels", new_value=tls
            )
            new_exp = DatasetExperience(dataset=new_dd, current_experience=eid)
            new_stream.append(new_exp)
        s = EagerCLStream(name, new_stream)
        streams.append(s)
    return with_task_labels(CLScenario(streams))
