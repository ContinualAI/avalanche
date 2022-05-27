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
from statistics import fmean
from typing import Dict, List, Union, TYPE_CHECKING
from collections import defaultdict, OrderedDict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, PluginMetric, \
    GenericPluginMetric
from avalanche.evaluation.metrics import ClassAccuracy

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class AverageMeanClassAccuracy(Metric[Dict[int, float]]):
    # TODO: unit tests
    """
    The Average Mean Class Accuracy (AMCA) metric. This is a standalone metric
    used to compute more specific ones.

    Instances of this metric keeps the running average accuracy
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average mean accuracy
    as the average accuracy of all previous experiences (also considering the
    accuracy in the current experience).
    The metric expects that the `next_train_experience()` method will be called
    after each experience. This is needed to consolidate the current mean
    accuracy. After calling `next_train_experience()`, a new experience with
    accuracy 0.0 is immediately started. If you need to obtain the AMCA up to
    experience `t-1`, obtain the `result()` before calling `next_experience()`.

    The set of classes to be tracked can be reduced (please refer to the
    constructor parameters).

    The reset method will bring the metric to its initial state
    (tracked classes will be kept). By default, this metric in its initial state
    will return a `{task_id -> amca}` dictionary in which all AMCAs are set to 0
    (that is, the `reset` method will hardly be useful when using this metric).
    """

    def __init__(self, classes=None):
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
            a task-free benchmark (with only task "0"), a simple list of class
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

    @torch.no_grad()
    def update(self,
               predicted_y: Tensor,
               true_y: Tensor,
               task_labels: Union[int, Tensor]) -> None:
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

    def next_train_experience(self):
        for task_id, mean_class_acc in self._get_curr_task_acc():
            self._prev_exps_accuracies[task_id].append(mean_class_acc)
        self._class_accuracies.reset()

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
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


class AverageClassAccuracyPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all class accuracy plugin metrics
    """

    def __init__(self, emit_at, mode, classes=None):
        self._amca= AverageMeanClassAccuracy(classes=classes)
        super().__init__(
            self._amca, reset_at='never', emit_at=emit_at,
            mode=mode)

    def update(self, strategy):
        self._amca.update(
            strategy.mb_output,
            strategy.mb_y,
            self._get_task_labels(strategy))

    def before_training_exp(self, *args, **kwargs):
        self._amca.next_train_experience()
        return super().before_training_exp(*args, **kwargs)

    @staticmethod
    def _get_task_labels(strategy: "SupervisedTemplate"):
        if hasattr(strategy, 'mb_task_id'):
            # Common situation
            return strategy.mb_task_id

        if hasattr(strategy.experience, "task_labels"):
            task_labels = strategy.experience.task_labels
        else:
            task_labels = [0]  # add fixed task label if not available.

        if len(task_labels) > 1:
            # task labels defined for each pattern
            # fall back to single task case
            task_label = 0
        else:
            task_label = task_labels[0]
        return task_label


class StreamAMCA(AverageClassAccuracyPluginMetric):
    """
    At the end of the entire stream of test experiences, this plugin metric
    reports the AMCA so far.

    This metric only works at eval time.
    """
    def __init__(self):
        """
        Creates an instance of StreamAMCA metric
        """
        super().__init__(
            emit_at='stream',
            mode='eval')

    def __str__(self):
        return "Top1_AMCA_Stream"


def amca_metrics(*, stream=True) \
        -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param stream: If True, will return a metric able to log
        the Average Mean Class Accuracy (AMCA) after testing on the stream of
        experiences of the evaluation stream.

    :return: A list of plugin metrics.
    """
    metrics = []

    if stream:
        metrics.append(StreamAMCA())

    return metrics


__all__ = [
    'AverageMeanClassAccuracy',
    'AverageClassAccuracyPluginMetric',
    'StreamAMCA',
    'amca_metrics'
]
