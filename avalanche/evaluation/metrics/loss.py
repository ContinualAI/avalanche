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

from collections import defaultdict
from typing import TYPE_CHECKING, List

import torch
from torch import Tensor

from avalanche.evaluation import PluginMetric, Metric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, \
    phase_and_task, stream_type
from avalanche.evaluation.metrics.mean import Mean
if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class Loss(Metric[float]):
    """
    The Loss metric. This is a general metric
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
        self._mean_loss = Mean()

    @torch.no_grad()
    def update(self, loss: Tensor, patterns: int) -> None:
        """
        Update the running loss given the loss Tensor and the minibatch size.

        :param loss: The loss Tensor. Different reduction types don't affect
            the result.
        :param patterns: The number of patterns in the minibatch.
        :return: None.
        """
        self._mean_loss.update(torch.mean(loss), weight=patterns)

    def result(self) -> float:
        """
        Retrieves the running average loss per pattern.

        Calling this method will not change the internal state of the metric.

        :return: The running loss, as a float.
        """
        return self._mean_loss.result()

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._mean_loss.reset()


class MinibatchLoss(PluginMetric[float]):
    """
    The minibatch loss metric.
    This metric only works at training time.

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
        super().__init__()

        self._loss_metric = Loss()

    def result(self) -> float:
        return self._loss_metric.result()

    def reset(self) -> None:
        self._loss_metric.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        self.reset()  # Because this metric computes the loss of a single mb
        self._loss_metric.update(strategy.loss,
                                 patterns=len(strategy.mb_y))
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        metric_value = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "Loss_MB"


class EpochLoss(PluginMetric[float]):
    """
    The average loss over a single training epoch.
    This metric only works at training time.

    The loss will be logged after each training epoch by computing
    the loss on the predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochLoss metric.
        """

        super().__init__()

        self._loss_metric = Loss()

    def before_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        self._loss_metric.update(strategy.loss, len(strategy.mb_y))

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        return self._package_result(strategy)

    def reset(self) -> None:
        self._loss_metric.reset()

    def result(self) -> float:
        return self._loss_metric.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        metric_value = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "Loss_Epoch"


class RunningEpochLoss(EpochLoss):
    """
    The average loss across all minibatches up to the current
    epoch iteration.
    This metric only works at training time.

    At each iteration, this metric logs the loss averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochLoss metric.
        """

        super().__init__()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        super().after_training_iteration(strategy)
        return self._package_result(strategy)

    def after_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        # Overrides the method from EpochLoss so that it doesn't
        # emit a metric value on epoch end!
        return None

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        metric_value = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "RunningLoss_Epoch"


class ExperienceLoss(PluginMetric[float]):
    """
    At the end of each experience, this metric reports
    the average loss over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceLoss metric
        """
        super().__init__()

        self._loss_metric = Loss()

    def reset(self) -> None:
        self._loss_metric.reset()

    def result(self) -> float:
        return self._loss_metric.result()

    def before_eval_exp(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        self._loss_metric.update(strategy.loss, len(strategy.mb_y))

    def after_eval_exp(self, strategy: 'PluggableStrategy') -> \
            'MetricResult':
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> \
            MetricResult:
        metric_value = self.result()

        metric_name = get_metric_name(self, strategy, add_experience=True)

        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "Loss_Exp"


class StreamLoss(PluginMetric[float]):
    """
    At the end of the entire stream of experiences, this metric reports the
    average loss over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamLoss metric
        """
        super().__init__()

        self._loss_metric = Loss()

    def reset(self) -> None:
        self._loss_metric.reset()

    def result(self) -> float:
        return self._loss_metric.result()

    def before_eval(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        self._loss_metric.update(strategy.loss, len(strategy.mb_y))

    def after_eval(self, strategy: 'PluggableStrategy') -> \
            'MetricResult':
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> \
            MetricResult:
        metric_value = self.result()

        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = '{}/{}_phase/{}_stream' \
            .format(str(self),
                    phase_name,
                    stream)

        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "Loss_Stream"


def loss_metrics(*, minibatch=False, epoch=False, epoch_running=False,
                 experience=False, stream=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of metric.

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

    metrics = []
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
    'Loss',
    'MinibatchLoss',
    'EpochLoss',
    'RunningEpochLoss',
    'ExperienceLoss',
    'StreamLoss',
    'loss_metrics'
]
