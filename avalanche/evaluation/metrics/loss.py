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

from typing import List, Dict

import torch
from torch import Tensor

from avalanche.evaluation import PluginMetric, Metric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task
from collections import defaultdict


class LossMetric(Metric[float]):
    """Loss Metric.

    Instances of this metric keep the running average loss
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.

    Each time `result` is called, this metric emits the average loss
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return a loss value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the loss metric.

        By default this metric in its initial state will return a loss
        value of 0. The metric can be updated by using the `update` method
        while the running loss can be retrieved using the `result` method.
        """
        self._mean_loss = Mean()
        """
        The mean utility that will be used to store the running accuracy
        for each task label.
        """

    @torch.no_grad()
    def update(self, loss: Tensor, patterns: int) -> None:
        """Update the running loss.

        :param loss: The loss Tensor. Different reduction types don't affect
            the result.
        :param patterns: The number of patterns in the minibatch.
        :return: None.
        """
        self._mean_loss.update(torch.mean(loss), weight=patterns)

    def result(self) -> float:
        """Retuns the running average loss per pattern.

        Calling this method will not change the internal state of the metric.

        :return: The running loss, as a float.
        """
        return self._mean_loss.result()

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        self._mean_loss.reset()


class TaskAwareLoss(Metric[Dict[int, float]]):
    """The standalone Loss metric. This is a general metric
    used to compute more specific ones.

    Instances of this metric keeps the running average loss
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average loss
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return a loss value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the loss metric.

        By default this metric in its initial state will return a loss
        value of 0. The metric can be updated by using the `update` method
        while the running loss can be retrieved using the `result` method.
        """
        self._mean_loss = defaultdict(Mean)
        """
        The mean utility that will be used to store the running accuracy
        for each task label.
        """

    @torch.no_grad()
    def update(self, loss: Tensor, patterns: int, task_label: int) -> None:
        """
        Update the running loss given the loss Tensor and the minibatch size.

        :param loss: The loss Tensor. Different reduction types don't affect
            the result.
        :param patterns: The number of patterns in the minibatch.
        :param task_label: the task label associated to the current experience
        :return: None.
        """
        self._mean_loss[task_label].update(torch.mean(loss), weight=patterns)

    def result(self) -> Dict[int, float]:
        """
        Retrieves the running average loss per pattern for all tasks.

        Calling this method will not change the internal state of the metric.
        :return: A dictionary `{task_label: mean_loss}`.
        """
        return {k: v.result() for k, v in self._mean_loss.items()}

    def result_task_label(self, task_label: int):
        """
        Retrieves the running average loss per pattern for a specific task.

        Calling this method will not change the internal state of the metric.
        :param task_label: The task label
        :return: A dictionary `{task_label: mean_loss}`.
        """
        assert task_label is not None
        return {task_label: self._mean_loss[task_label].result()}

    def reset(self) -> None:
        """
        Resets the metric.

        :param task_label: None to reset all metric values. If an int,
            reset metric value corresponding to that task label.
        :return: None.
        """
        self._mean_loss = defaultdict(Mean)

    def reset_task_label(self, task_label: int):
        assert task_label is not None
        self._mean_loss[task_label].reset()


class LossPluginMetric(GenericPluginMetric[float, LossMetric]):
    def __init__(self, reset_at, emit_at, mode):
        self._loss = LossMetric()
        super(LossPluginMetric, self).__init__(self._loss, reset_at, emit_at, mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._loss.update(strategy.loss, patterns=len(strategy.mb_y))


class LossPerTaskPluginMetric(GenericPluginMetric[Dict[int, float], TaskAwareLoss]):
    def __init__(self, reset_at, emit_at, mode):
        self._loss = TaskAwareLoss()
        super().__init__(self._loss, reset_at, emit_at, mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> Dict[int, float]:
        return self._metric.result()

    def update(self, strategy):
        self._loss.update(
            loss=strategy.loss,
            patterns=len(strategy.mb_y),
            task_label=strategy.mb_task_id,
        )


class MinibatchLoss(LossPluginMetric):
    """
    The minibatch loss metric.
    This plugin metric only works at training time.

    This metric computes the average loss over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochLoss` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchLoss metric.
        """
        super(MinibatchLoss, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Loss_MB"


class EpochLoss(LossPluginMetric):
    """
    The average loss over a single training epoch.
    This plugin metric only works at training time.

    The loss will be logged after each training epoch by computing
    the loss on the predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochLoss metric.
        """

        super(EpochLoss, self).__init__(reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "Loss_Epoch"


class RunningEpochLoss(LossPluginMetric):
    """
    The average loss across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the loss averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochLoss metric.
        """

        super(RunningEpochLoss, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "RunningLoss_Epoch"


class ExperienceLoss(LossPluginMetric):
    """
    At the end of each experience, this metric reports
    the average loss over all patterns seen in that experience.
    This plugin metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceLoss metric
        """
        super(ExperienceLoss, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "Loss_Exp"


class StreamLoss(LossPluginMetric):
    """
    At the end of the entire stream of experiences, this metric reports the
    average loss over all patterns seen in all experiences.
    This plugin metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamLoss metric
        """
        super(StreamLoss, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "Loss_Stream"


def loss_metrics(
    *, minibatch=False, epoch=False, epoch_running=False, experience=False, stream=False
) -> List[LossPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch loss at training time.
    :param epoch: If True, will return a metric able to log
        the epoch loss at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch loss at training time.
    :param experience: If True, will return a metric able to log
        the loss on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the loss averaged over the entire evaluation stream of experiences.

    :return: A list of plugin metrics.
    """

    metrics: List[LossPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchLoss())

    if epoch:
        metrics.append(EpochLoss())

    if epoch_running:
        metrics.append(RunningEpochLoss())

    if experience:
        metrics.append(ExperienceLoss())

    if stream:
        metrics.append(StreamLoss())

    return metrics


__all__ = [
    "LossMetric",
    "TaskAwareLoss",
    "MinibatchLoss",
    "EpochLoss",
    "RunningEpochLoss",
    "ExperienceLoss",
    "StreamLoss",
    "loss_metrics",
]
