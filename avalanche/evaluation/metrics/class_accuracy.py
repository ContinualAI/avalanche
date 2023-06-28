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

from typing import Dict, List, Set, Union, TYPE_CHECKING, Iterable, Optional
from collections import defaultdict, OrderedDict

import torch
from torch import Tensor
from avalanche.evaluation import (
    Metric,
    PluginMetric,
    _ExtendedGenericPluginMetric,
    _ExtendedPluginMetricValue,
)
from avalanche.evaluation.metric_utils import (
    default_metric_name_template,
    generic_get_metric_name,
)
from avalanche.evaluation.metrics import Mean

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


TrackedClassesType = Union[Dict[int, Iterable[int]], Iterable[int]]


class ClassAccuracy(Metric[Dict[int, Dict[int, float]]]):
    """
    The Class Accuracy metric. This is a standalone metric
    used to compute more specific ones.

    Instances of this metric keeps the running average accuracy
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average accuracy
    for all classes seen and across all predictions made since the last `reset`.
    The set of classes to be tracked can be reduced (please refer to the
    constructor parameters).

    The reset method will bring the metric to its initial state. By default,
    this metric in its initial state will return a
    `{task_id -> {class_id -> accuracy}}` dictionary in which all accuracies are
    set to 0.
    """

    def __init__(self, classes: Optional[TrackedClassesType] = None):
        """
        Creates an instance of the standalone Accuracy metric.

        By default, this metric in its initial state will return an empty
        dictionary. The metric can be updated by using the `update` method
        while the running accuracies can be retrieved using the `result` method.

        By using the `classes` parameter, one can restrict the list of classes
        to be tracked and in addition can immediately create plots for
        yet-to-be-seen classes.

        :param classes: The classes to keep track of. If None (default), all
            classes seen are tracked. Otherwise, it can be a dict of classes
            to be tracked (as "task-id" -> "list of class ids") or, if running
            a task-free benchmark (with only task 0), a simple list of class
            ids. By passing this parameter, the plot of each class is
            created immediately (with a default value of 0.0) and plots
            will be aligned across all classes. In addition, this can be used to
            restrict the classes for which the accuracy should be logged.
        """

        self.classes: Dict[int, Set[int]] = defaultdict(set)
        """
        The list of tracked classes.
        """

        self.dynamic_classes = False
        """
        If True, newly encountered classes will be tracked.
        """

        self._class_accuracies: Dict[int, Dict[int, Mean]] = defaultdict(
            lambda: defaultdict(Mean)
        )
        """
        A dictionary "task_id -> {class_id -> Mean}".
        """

        if classes is not None:
            if isinstance(classes, dict):
                # Task-id -> classes dict
                self.classes = {
                    task_id: self._ensure_int_classes(class_list)
                    for task_id, class_list in classes.items()
                }
            else:
                # Assume is a plain iterable
                self.classes = {0: self._ensure_int_classes(classes)}
        else:
            self.dynamic_classes = True

        self.__init_accs_for_known_classes()

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
        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError("Size mismatch for true_y and task_labels tensors")

        if not isinstance(task_labels, (int, Tensor)):
            raise ValueError(
                f"Task label type: {type(task_labels)}, " f"expected int or Tensor"
            )

        if isinstance(task_labels, int):
            task_labels = [task_labels] * len(true_y)

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        for pred, true, t in zip(predicted_y, true_y, task_labels):
            t = int(t)

            if self.dynamic_classes:
                self.classes[t].add(int(true))
            else:
                if t not in self.classes:
                    continue
                if int(true) not in self.classes[t]:
                    continue

            true_positives = (pred == true).float().item()  # 1.0 or 0.0
            self._class_accuracies[t][int(true)].update(true_positives, 1)

    def result(self) -> Dict[int, Dict[int, float]]:
        """
        Retrieves the running accuracy for each class.

        Calling this method will not change the internal state of the metric.

        :return: A dictionary `{task_id -> {class_id -> running_accuracy}}`. The
            running accuracy of each class is a float value between 0 and 1.
        """
        running_class_accuracies: Dict[int, Dict[int, float]] = OrderedDict()
        for task_label in sorted(self._class_accuracies.keys()):
            task_dict = self._class_accuracies[task_label]
            running_class_accuracies[task_label] = OrderedDict()
            for class_id in sorted(task_dict.keys()):
                running_class_accuracies[task_label][class_id] = task_dict[
                    class_id
                ].result()

        return running_class_accuracies

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._class_accuracies = defaultdict(lambda: defaultdict(Mean))
        self.__init_accs_for_known_classes()

    def __init_accs_for_known_classes(self):
        for task_id, task_classes in self.classes.items():
            for c in task_classes:
                self._class_accuracies[task_id][c].reset()

    @staticmethod
    def _ensure_int_classes(classes_iterable: Iterable[int]):
        return set(int(c) for c in classes_iterable)


class ClassAccuracyPluginMetric(_ExtendedGenericPluginMetric[ClassAccuracy]):
    """
    Base class for all class accuracy plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, classes=None):
        super(ClassAccuracyPluginMetric, self).__init__(
            ClassAccuracy(classes=classes),
            reset_at=reset_at,
            emit_at=emit_at,
            mode=mode,
        )
        self.phase_name = "train"
        self.stream_name = "train"
        self.experience_id = 0

    def update(self, strategy: "SupervisedTemplate"):
        assert strategy.mb_output is not None
        assert strategy.experience is not None
        self._metric.update(strategy.mb_output, strategy.mb_y, strategy.mb_task_id)

        self.phase_name = "train" if strategy.is_training else "eval"
        self.stream_name = strategy.experience.origin_stream.name
        self.experience_id = strategy.experience.current_experience

    def result(self) -> List[_ExtendedPluginMetricValue]:
        metric_values = []
        task_accuracies = self._metric.result()

        for task_id, task_classes in task_accuracies.items():
            for class_id, class_accuracy in task_classes.items():
                metric_values.append(
                    _ExtendedPluginMetricValue(
                        metric_name=str(self),
                        metric_value=class_accuracy,
                        phase_name=self.phase_name,
                        stream_name=self.stream_name,
                        task_label=task_id,
                        experience_id=self.experience_id,
                        class_id=class_id,
                    )
                )

        return metric_values

    def metric_value_name(self, m_value: _ExtendedPluginMetricValue) -> str:
        m_value_values = vars(m_value)
        add_exp = self._emit_at == "experience"
        if not add_exp:
            del m_value_values["experience_id"]
        m_value_values["class_id"] = m_value.other_info["class_id"]

        return generic_get_metric_name(
            default_metric_name_template(m_value_values) + "/{class_id}",
            m_value_values,
        )


class MinibatchClassAccuracy(ClassAccuracyPluginMetric):
    """
    The minibatch plugin class accuracy metric.
    This metric only works at training time.

    This metric computes the average accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochClassAccuracy` instead.
    """

    def __init__(self, classes=None):
        """
        Creates an instance of the MinibatchClassAccuracy metric.
        """
        super().__init__(
            reset_at="iteration",
            emit_at="iteration",
            mode="train",
            classes=classes,
        )

    def __str__(self):
        return "Top1_ClassAcc_MB"


class EpochClassAccuracy(ClassAccuracyPluginMetric):
    """
    The average class accuracy over a single training epoch.
    This plugin metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch (separately
    for each class).
    """

    def __init__(self, classes=None):
        """
        Creates an instance of the EpochClassAccuracy metric.
        """
        super().__init__(
            reset_at="epoch", emit_at="epoch", mode="train", classes=classes
        )

    def __str__(self):
        return "Top1_ClassAcc_Epoch"


class RunningEpochClassAccuracy(ClassAccuracyPluginMetric):
    """
    The average class accuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch (separately for each class).
    The metric resets its state after each training epoch.
    """

    def __init__(self, classes=None):
        """
        Creates an instance of the RunningEpochClassAccuracy metric.
        """

        super().__init__(
            reset_at="epoch", emit_at="iteration", mode="train", classes=classes
        )

    def __str__(self):
        return "Top1_RunningClassAcc_Epoch"


class ExperienceClassAccuracy(ClassAccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience (separately
    for each class).

    This metric only works at eval time.
    """

    def __init__(self, classes=None):
        """
        Creates an instance of ExperienceClassAccuracy metric
        """
        super().__init__(
            reset_at="experience",
            emit_at="experience",
            mode="eval",
            classes=classes,
        )

    def __str__(self):
        return "Top1_ClassAcc_Exp"


class StreamClassAccuracy(ClassAccuracyPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences
    (separately for each class).

    This metric only works at eval time.
    """

    def __init__(self, classes=None):
        """
        Creates an instance of StreamClassAccuracy metric
        """
        super().__init__(
            reset_at="stream", emit_at="stream", mode="eval", classes=classes
        )

    def __str__(self):
        return "Top1_ClassAcc_Stream"


def class_accuracy_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    classes=None,
) -> List[ClassAccuracyPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the per-class minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the per-class epoch accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the per-class  running epoch accuracy at training time.
    :param experience: If True, will return a metric able to log
        the per-class accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the per-class accuracy averaged over the entire evaluation stream of
        experiences.
    :param classes: The list of classes to track. See the corresponding
        parameter of :class:`ClassAccuracy` for a precise explanation.

    :return: A list of plugin metrics.
    """
    metrics: List[ClassAccuracyPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchClassAccuracy(classes=classes))

    if epoch:
        metrics.append(EpochClassAccuracy(classes=classes))

    if epoch_running:
        metrics.append(RunningEpochClassAccuracy(classes=classes))

    if experience:
        metrics.append(ExperienceClassAccuracy(classes=classes))

    if stream:
        metrics.append(StreamClassAccuracy(classes=classes))

    return metrics


__all__ = [
    "TrackedClassesType",
    "ClassAccuracy",
    "MinibatchClassAccuracy",
    "EpochClassAccuracy",
    "RunningEpochClassAccuracy",
    "ExperienceClassAccuracy",
    "StreamClassAccuracy",
    "class_accuracy_metrics",
]
