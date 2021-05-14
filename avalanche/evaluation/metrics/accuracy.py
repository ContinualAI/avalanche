################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import TYPE_CHECKING, List

import torch
from torch import Tensor
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean


class Accuracy(Metric[float]):
    """
    The Accuracy metric. This is a standalone metric
    used to compute more specific ones.

    Instances of this metric keeps the running average accuracy
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average accuracy
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an accuracy value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the standalone Accuracy metric.

        By default this metric in its initial state will return an accuracy
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """

        self._mean_accuracy = Mean()
        """
        The mean utility that will be used to store the running accuracy.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor) -> None:
        """
        Update the running accuracy given the true and predicted labels.

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

        true_positives = float(torch.sum(torch.eq(predicted_y, true_y)))
        total_patterns = len(true_y)

        self._mean_accuracy.update(true_positives / total_patterns,
                                   total_patterns)

    def result(self) -> float:
        """
        Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        :return: The running accuracy, as a float value between 0 and 1.
        """
        return self._mean_accuracy.result()

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._mean_accuracy.reset()


class AccuracyPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all accuracies plugin metrics
    """
    def __init__(self, reset_at, emit_at, mode):
        self._accuracy = Accuracy()
        super(AccuracyPluginMetric, self).__init__(
            self._accuracy, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def update(self, strategy):
        self._accuracy.update(strategy.mb_output, strategy.mb_y)


class MinibatchAccuracy(AccuracyPluginMetric):
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
        super(MinibatchAccuracy, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "Top1_Acc_MB"


class EpochAccuracy(AccuracyPluginMetric):
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

        super(EpochAccuracy, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "Top1_Acc_Epoch"


class RunningEpochAccuracy(AccuracyPluginMetric):
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

        super(RunningEpochAccuracy, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train')

    def __str__(self):
        return "Top1_RunningAcc_Epoch"


class ExperienceAccuracy(AccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceAccuracy, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return "Top1_Acc_Exp"


class StreamAccuracy(AccuracyPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamAccuracy, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return "Top1_Acc_Stream"


def accuracy_metrics(*, minibatch=False, epoch=False, epoch_running=False,
                     experience=False, stream=False) -> List[PluginMetric]:
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
        metrics.append(MinibatchAccuracy())

    if epoch:
        metrics.append(EpochAccuracy())

    if epoch_running:
        metrics.append(RunningEpochAccuracy())

    if experience:
        metrics.append(ExperienceAccuracy())

    if stream:
        metrics.append(StreamAccuracy())

    return metrics


__all__ = [
    'Accuracy',
    'MinibatchAccuracy',
    'EpochAccuracy',
    'RunningEpochAccuracy',
    'ExperienceAccuracy',
    'StreamAccuracy',
    'accuracy_metrics'
]
