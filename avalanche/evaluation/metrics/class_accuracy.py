# Date: 22-07-2021                                                             #
# Author(s): Eli Verwimp                                                       #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import Dict, List
from collections import defaultdict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, GenericPluginMetric, PluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy


class ClassAccuracy(Metric[Dict[int, float]]):
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

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an empty dictionary.
    """

    def __init__(self, classes=None):
        """
        Creates an instance of the standalone Accuracy metric.

        By default this metric in its initial state will return an empty
        dictionary. The metric can be updated by using the `update` method
        while the running accuracies can be retrieved using the `result` method.

        :param classes: The classes to keep track of. If None (default), all
        classes seen are tracked.
        :return: None
        """

        if classes is not None:
            self.classes = set(int(c) for c in classes)
        else:
            self.classes = None
        self._class_accuracies: Dict[int, Accuracy] = defaultdict(Accuracy)

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor) -> None:
        """
        Update the running accuracy given the true and predicted labels for each
        class.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError('Size mismatch for true_y and predicted_y tensors')

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        for label in true_y.unique():
            label = int(label)
            if self.classes is not None and label not in self.classes:
                continue

            mask = torch.eq(true_y, label)
            self._class_accuracies[label].update(predicted_y[mask],
                                                 true_y[mask], 0)

    def result(self) -> Dict[int, float]:
        """
        Retrieves the running accuracy for each class.

        Calling this method will not change the internal state of the metric.

        :return: The running accuracy, as a float value between 0 and 1.
        """
        return [(label, acc.result(0)[0]) for label, acc in
                self._class_accuracies.items()
                if label != "default_factory"]

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._class_accuracies = defaultdict(Accuracy)


class ClassAccuracyPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all class accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, classes=None):
        self._class_accuracy = ClassAccuracy(classes)
        super(ClassAccuracyPluginMetric, self).__init__(
            self._class_accuracy, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def update(self, strategy):
        self._class_accuracy.update(strategy.mb_output, strategy.mb_y)


class MinibatchClassAccuracy(ClassAccuracyPluginMetric):
    """
    The minibatch plugin accuracy metric.
    This metric only works at training time.

    This metric computes the average accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracy` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super(MinibatchClassAccuracy, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "Top1_ClassAcc_MB"


class EpochClassAccuracy(ClassAccuracyPluginMetric):
    """
    The average accuracy over a single training epoch.
    This plugin metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super(EpochClassAccuracy, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "Top1_ClassAcc_Epoch"


class RunningEpochClassAccuracy(ClassAccuracyPluginMetric):
    """
    The average accuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochAccuracy metric.
        """

        super(RunningEpochClassAccuracy, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train')

    def __str__(self):
        return "Top1_RunningClassAcc_Epoch"


class ExperienceClassAccuracy(ClassAccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceClassAccuracy, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return "Top1_ClassAcc_Exp"


class StreamClassAccuracy(ClassAccuracyPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamClassAccuracy, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return "Top1_ClassAcc_Stream"


def class_accuracy_metrics(*, minibatch=False, epoch=False, epoch_running=False,
                           experience=False, stream=False) \
        -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch accuracy at training time.
    :param experience: If True, will return a metric able to log
        the accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the accuracy averaged over the entire evaluation stream of experiences.

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchClassAccuracy())

    if epoch:
        metrics.append(EpochClassAccuracy())

    if epoch_running:
        metrics.append(RunningEpochClassAccuracy())

    if experience:
        metrics.append(ExperienceClassAccuracy())

    if stream:
        metrics.append(StreamClassAccuracy())

    return metrics


__all__ = [
    'ClassAccuracy',
    'MinibatchClassAccuracy',
    'EpochClassAccuracy',
    'RunningEpochClassAccuracy',
    'ExperienceClassAccuracy',
    'StreamClassAccuracy',
    'class_accuracy_metrics'
]
