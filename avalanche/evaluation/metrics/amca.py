################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 26-05-2022                                                             #
# Author(s): Eli Verwimp, Lorenzo Pellegrini                                   #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################


from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Union,
    TYPE_CHECKING,
    Optional,
    Sequence,
    Set,
)

fmean: Callable[[Iterable[float]], float]
try:
    from statistics import fmean
except ImportError:
    from statistics import mean as fmean

from collections import defaultdict, OrderedDict

import torch
from torch import Tensor
from avalanche.evaluation import (
    Metric,
    PluginMetric,
    _ExtendedGenericPluginMetric,
    _ExtendedPluginMetricValue,
)
from avalanche.evaluation.metric_utils import generic_get_metric_name
from avalanche.evaluation.metrics.class_accuracy import (
    ClassAccuracy,
    TrackedClassesType,
)

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class AverageMeanClassAccuracy(Metric[Dict[int, float]]):
    """
    The Average Mean Class Accuracy (AMCA) metric. This is a standalone metric
    used to compute more specific ones.

    Instances of this metric keeps the running average accuracy
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.

    Beware that this class does not provide mechanisms to separate scores based
    on the originating data stream. For this, please refer to
    :class:`MultiStreamAMCA`.

    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average mean accuracy
    as the average accuracy of all previous experiences (also considering the
    accuracy in the current experience).
    The metric expects that the :meth:`next_experience` method will be called
    after each experience. This is needed to consolidate the current mean
    accuracy. After calling :meth:`next_experience`, a new experience with
    accuracy 0.0 is immediately started. If you need to obtain the AMCA up to
    experience `t-1`, obtain the :meth:`result` before calling
    :meth:`next_experience`.


    The set of classes to be tracked can be reduced (please refer to the
    constructor parameters).

    The reset method will bring the metric to its initial state
    (tracked classes will be kept). By default, this metric in its initial state
    will return a `{task_id -> amca}` dictionary in which all AMCAs are set to 0
    (that is, the `reset` method will hardly be useful when using this metric).
    """

    def __init__(self, classes: Optional[TrackedClassesType] = None):
        """
        Creates an instance of the standalone AMCA metric.

        By default, this metric in its initial state will return an empty
        dictionary. The metric can be updated by using the `update` method
        while the running AMCA can be retrieved using the `result` method.

        By using the `classes` parameter, one can restrict the list of classes
        to be tracked and in addition will initialize the accuracy for that
        class to 0.0.

        Setting the `classes` parameter is very important, as the mean class
        accuracy may vary based on this! If the test set is fixed and contains
        at least a sample for each class, then it is safe to leave `classes`
        to None.

        :param classes: The classes to keep track of. If None (default), all
            classes seen are tracked. Otherwise, it can be a dict of classes
            to be tracked (as "task-id" -> "list of class ids") or, if running
            a task-free benchmark (with only task 0), a simple list of class
            ids. By passing this parameter, the list of classes to be considered
            is created immediately. This will ensure that the mean class
            accuracy is correctly computed. In addition, this can be used to
            restrict the classes that should be considered when computing the
            mean class accuracy.
        """
        self._class_accuracies = ClassAccuracy(classes=classes)
        """
        A dictionary "task_id -> {class_id -> Mean}".
        """

        # Here a Mean metric could be used as well. However, that could make it
        # difficult to compute the running AMCA...
        self._prev_exps_accuracies: Dict[int, List[float]] = defaultdict(list)
        """
        The mean class accuracy of previous experiences as a dictionary
        `{task_id -> [accuracies]}`.
        """

        self._updated_once = False

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        task_labels: Union[int, Tensor],
    ) -> None:
        """
        Update the running accuracy given the true and predicted labels for each
        class.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.
        :return: None.
        """
        self._updated_once = True
        self._class_accuracies.update(predicted_y, true_y, task_labels)

    def result(self) -> Dict[int, float]:
        """
        Retrieves the running AMCA for each task.

        Calling this method will not change the internal state of the metric.

        :return: A dictionary `{task_id -> amca}`. The
            running AMCA of each task is a float value between 0 and 1.
        """
        curr_task_acc = self._get_curr_task_acc()

        all_task_ids = set(self._prev_exps_accuracies.keys())
        all_task_ids = all_task_ids.union(curr_task_acc.keys())

        mean_accs = OrderedDict()
        for task_id in sorted(all_task_ids):
            prev_accs = self._prev_exps_accuracies.get(task_id, list())
            curr_acc = curr_task_acc.get(task_id, 0)
            mean_accs[task_id] = fmean(prev_accs + [curr_acc])

        return mean_accs

    def next_experience(self):
        """
        Moves to the next experience.

        This will consolidate the class accuracies for the current experience.

        This method can also be safely called before even calling the `update`
        method for the first time. In that case, this call will be ignored.
        """
        if not self._updated_once:
            return

        for task_id, mean_class_acc in self._get_curr_task_acc().items():
            self._prev_exps_accuracies[task_id].append(mean_class_acc)
        self._class_accuracies.reset()

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._updated_once = False
        self._class_accuracies.reset()
        self._prev_exps_accuracies.clear()

    def _get_curr_task_acc(self):
        task_acc = dict()
        class_acc = self._class_accuracies.result()
        for task_id, task_classes in class_acc.items():
            class_accuracies = list(task_classes.values())
            mean_class_acc = fmean(class_accuracies)

            task_acc[task_id] = mean_class_acc
        return task_acc


class MultiStreamAMCA(Metric[Dict[str, Dict[int, float]]]):
    """
    An extension of the Average Mean Class Accuracy (AMCA) metric
    (class:`AverageMeanClassAccuracy`) able to separate the computation of the
    AMCA based on the current stream.
    """

    def __init__(self, classes=None, streams=None):
        """
        Creates an instance of a MultiStream AMCA.

        :param classes: The list of classes to track. This has the same semantic
            of the `classes` parameter of class
            :class:`AverageMeanClassAccuracy`.
        :param streams: The list of streams to track. Defaults to None, which
            means that all stream will be tracked. This is not recommended, as
            you usually will want to track the "test" stream only.
        """

        self._limit_streams = streams
        if self._limit_streams is not None:
            self._limit_streams = set(self._limit_streams)

        self._limit_classes = classes
        self._amcas: Dict[str, AverageMeanClassAccuracy] = dict()

        self._current_stream: Optional[str] = None
        self._streams_in_this_phase: Set[str] = set()

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        task_labels: Union[int, Tensor],
    ) -> None:
        """
        Update the running accuracy given the true and predicted labels for each
        class.

        This will update the accuracies for the "current stream" (the one set
        through `next_experience`). If `next_experience` has not been called,
        then an error will be raised.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.
        :return: None.
        """
        if self._current_stream is None:
            raise RuntimeError(
                "No current stream set. " 'Call "set_stream" to set the current stream.'
            )

        if self._is_stream_tracked(self._current_stream):
            self._amcas[self._current_stream].update(predicted_y, true_y, task_labels)

    def result(self) -> Dict[str, Dict[int, float]]:
        """
        Retrieves the running AMCA for each stream.

        Calling this method will not change the internal state of the metric.

        :return: A dictionary `{stream_name -> {task_id -> amca}}`. The
            running AMCA of each task is a float value between 0 and 1.
        """
        all_streams_dict = OrderedDict()
        for stream_name in sorted(self._amcas.keys()):
            stream_metric = self._amcas[stream_name]
            stream_result = stream_metric.result()
            all_streams_dict[stream_name] = stream_result
        return all_streams_dict

    def set_stream(self, stream_name: str):
        """
        Switches to a specific stream.

        :param stream_name: The name of the stream.
        """
        self._current_stream = stream_name
        if not self._is_stream_tracked(stream_name):
            return

        if self._current_stream not in self._amcas:
            self._amcas[stream_name] = AverageMeanClassAccuracy(
                classes=self._limit_classes
            )
        self._streams_in_this_phase.add(stream_name)

    def finish_phase(self):
        """
        Moves to the next phase.

        This will consolidate the class accuracies recorded so far.
        """
        for stream_name in self._streams_in_this_phase:
            self._amcas[stream_name].next_experience()

        self._streams_in_this_phase.clear()

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        for metric in self._amcas.values():
            metric.reset()
        self._current_stream = None
        self._streams_in_this_phase.clear()

    def _is_stream_tracked(self, stream_name):
        return self._limit_streams is None or stream_name in self._limit_streams


class AMCAPluginMetric(_ExtendedGenericPluginMetric):
    """
    Plugin metric for the Average Mean Class Accuracy (AMCA).

    The AMCA is tracked for the classes and streams defined in the constructor.

    In addition, by default, the results obtained through the periodic
    evaluation (mid-training validation) mechanism are ignored.
    """

    VALUE_NAME = "{metric_name}/{stream_name}_stream/Task{task_label:03}"

    def __init__(self, classes=None, streams=None, ignore_validation=True):
        """
        Instantiates the AMCA plugin metric.

        :param classes: The classes to track. Refer to :class:`MultiStreamAMCA`
            for more details.
        :param streams: The streams to track. Defaults to None, which means that
            all streams will be considered. Beware that, when creating instances
            of this class using the :func:`amca_metrics` helper, the resulting
            metric will only track the "test" stream by default.
        :param ignore_validation: Defaults to True, which means that periodic
            evaluations will be ignored (recommended).
        """
        self._ms_amca = MultiStreamAMCA(classes=classes, streams=streams)
        self._ignore_validation = ignore_validation

        self._is_training = False
        super().__init__(self._ms_amca, reset_at="never", emit_at="stream", mode="eval")

    def update(self, strategy: "SupervisedTemplate"):
        if self._is_training and self._ignore_validation:
            # Running a validation (eval phase inside a train phase), ignore it
            return

        self._ms_amca.update(strategy.mb_output, strategy.mb_y, strategy.mb_task_id)

    def before_training(self, strategy: "SupervisedTemplate"):
        self._is_training = True
        return super().before_training(strategy)

    def after_training(self, strategy: "SupervisedTemplate"):
        self._is_training = False
        return super().after_training(strategy)

    def before_eval(self, strategy: "SupervisedTemplate"):
        # In the first eval phase, calling finish_phase will do nothing
        # (as expected)
        if not (self._is_training and self._ignore_validation):
            # If not running a validation
            self._ms_amca.finish_phase()
        return super().before_eval(strategy)

    def before_eval_exp(self, strategy: "SupervisedTemplate"):
        assert strategy.experience is not None
        if not (self._is_training and self._ignore_validation):
            # If not running a validation
            self._ms_amca.set_stream(strategy.experience.origin_stream.name)
        return super().before_eval_exp(strategy)

    def result(self) -> List[_ExtendedPluginMetricValue]:
        if self._is_training and self._ignore_validation:
            # Running a validation, ignore it
            return []

        metric_values = []
        stream_amca = self._ms_amca.result()

        for stream_name, stream_accs in stream_amca.items():
            for task_id, task_amca in stream_accs.items():
                metric_values.append(
                    _ExtendedPluginMetricValue(
                        metric_name=str(self),
                        metric_value=task_amca,
                        phase_name="eval",
                        stream_name=stream_name,
                        task_label=task_id,
                        experience_id=None,
                    )
                )

        return metric_values

    def metric_value_name(self, m_value: _ExtendedPluginMetricValue) -> str:
        return generic_get_metric_name(AMCAPluginMetric.VALUE_NAME, vars(m_value))

    def __str__(self):
        return "Top1_AMCA_Stream"


def amca_metrics(streams: Sequence[str] = ("test",)) -> PluginMetric:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    The returned metric will not compute the AMCA when the
    :class:`PeriodicEval` plugin is used. To change this behavior,
    you can instantiate a :class:`AMCAPluginMetric` by setting
    `ignore_validation` to False.

    :param streams: The list of streams to track. Defaults to "test" only.

    :return: The AMCA plugin metric.
    """
    return AMCAPluginMetric(streams=streams, ignore_validation=True)


__all__ = [
    "AverageMeanClassAccuracy",
    "MultiStreamAMCA",
    "AMCAPluginMetric",
    "amca_metrics",
]
