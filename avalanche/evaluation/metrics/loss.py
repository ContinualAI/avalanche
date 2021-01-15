#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################
from collections import defaultdict
from typing import Dict, TYPE_CHECKING

import torch
from torch import Tensor

from avalanche.evaluation import PluginMetric, Metric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics.mean import Mean
if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class Loss(Metric[float]):
    """
    The average loss metric.

    Instances of this metric compute the running average loss by receiving a
    Tensor describing the loss of a minibatch. This metric then uses that tensor
    to computes the average loss per pattern.

    The Tensor passed to the `update` method are averaged to obtain a
    minibatch average loss. In order to compute the per-pattern running loss,
    the users should must pass the number of patterns in that minibatch as the
    second parameter of the `update` method. The number of patterns can't be
    usually obtained by analyzing the shape of the loss Tensor, which usually
    consists of a single float value.

    The result is the running loss computed as the accumulated average loss.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an accuracy value of 0.
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

    The logged loss value is the per-pattern loss obtained by averaging the loss
    of patterns contained in the minibatch.

    This metric "logs" the loss value after each iteration. Beware that this
    metric will not average the loss across minibatches!

    If a more coarse-grained logging is needed, consider using
    :class:`EpochLoss` and/or :class:`TaskLoss` instead.
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the MinibatchLoss metric.

        The train and test parameters are used to control if this metric should
        compute and log values referred to the train phase, test phase or both.
        At least one of them must be True!

        Beware that the test parameter defaults to False because logging
        the test minibatch loss it's and uncommon practice.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             ' time.')
        self._minibatch_loss = Loss()
        self._compute_train_loss = train
        self._compute_test_loss = test

    def result(self) -> float:
        return self._minibatch_loss.result()

    def reset(self) -> None:
        self._minibatch_loss.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._compute_train_loss:
            return self._on_iteration(strategy)

    def after_test_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._compute_test_loss:
            return self._on_iteration(strategy)

    def _on_iteration(self, strategy: 'PluggableStrategy'):
        self.reset()  # Because this metric computes the loss of a single mb
        self._minibatch_loss.update(strategy.loss,
                                    patterns=len(strategy.mb_y))
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'Loss_MB/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class EpochLoss(PluginMetric[float]):
    """
    The average epoch loss metric.

    The logged loss value is the per-pattern loss obtained by averaging the loss
    of all patterns encountered in that epoch, which means that having
    unbalanced minibatch sizes will not affect the metric.
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the EpochLoss metric.

        The train and test parameters are used to control if this metric should
        compute and log values referred to the train phase, test phase or both.
        At least one of them must be True!

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to True.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             ' time.')

        self._mean_loss = Loss()
        self._compute_train_accuracy = train
        self._compute_test_accuracy = test

    def before_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        if self._compute_train_accuracy:
            self.reset()

    def before_test_step(self, strategy: 'PluggableStrategy') -> None:
        if self._compute_test_accuracy:
            self.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        if self._compute_train_accuracy:
            self._mean_loss.update(strategy.loss, len(strategy.mb_y))

    def after_test_iteration(self, strategy: 'PluggableStrategy') -> None:
        if self._compute_test_accuracy:
            self._mean_loss.update(strategy.loss, len(strategy.mb_y))

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._compute_train_accuracy:
            return self._package_result(strategy)

    def after_test_step(self, strategy: 'PluggableStrategy') -> MetricResult:
        if self._compute_test_accuracy:
            return self._package_result(strategy)

    def reset(self) -> None:
        self._mean_loss.reset()

    def result(self) -> float:
        return self._mean_loss.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'Loss/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class RunningEpochLoss(EpochLoss):
    """
    The running average loss metric.

    This metric behaves like :class:`EpochLoss` but, differently from it,
    this metric will log the running loss value after each iteration.
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the RunningEpochLoss metric.

        The train and test parameters are used to control if this metric should
        compute and log values referred to the train phase, test phase or both.
        At least one of them must be True!

        Beware that the test parameter defaults to False because logging
        the running test accuracy it's and uncommon practice.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             ' time.')

        self._compute_train_accuracy = train
        self._compute_test_accuracy = test

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        super().after_training_iteration(strategy)
        if self._compute_train_accuracy:
            return self._package_result(strategy)

    def after_test_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        super().after_test_iteration(strategy)
        if self._compute_test_accuracy:
            return self._package_result(strategy)

    def after_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        # Overrides the method from EpochLoss so that it doesn't
        # emit a metric value on epoch end!
        return None

    def after_test_step(self, strategy: 'PluggableStrategy') -> None:
        # Overrides the method from EpochLoss so that it doesn't
        # emit a metric value on epoch end!
        return None

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'Loss_Running/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class TaskLoss(PluginMetric[Dict[int, float]]):
    """
    The task loss metric.

    The logged loss value is the per-pattern loss obtained by averaging the loss
    of all test patterns of a task. This is a common metric used in the
    evaluation of a Continual Learning algorithm.

    Can be safely used when evaluation task-free scenarios, in which case the
    default task label "0" will be used.

    The task losses will be logged at the end of the test phase. This metric
    doesn't apply to the training phase.
    """

    def __init__(self):
        """
        Creates an instance of the TaskLoss metric.
        """
        super().__init__()

        self._task_loss: Dict[int, Loss] = defaultdict(Loss)
        """
        A dictionary used to store the loss for each task.
        """

    def reset(self) -> None:
        self._task_loss = defaultdict(Loss)

    def result(self) -> Dict[int, float]:
        result_dict = dict()
        for task_id in self._task_loss:
            result_dict[task_id] = self._task_loss[task_id].result()
        return result_dict

    def update(self, loss: Tensor, patterns: int, task_label: int) -> None:
        self._task_loss[task_label].update(loss, patterns)

    def before_test(self, strategy) -> None:
        self.reset()

    def after_test_iteration(self, strategy: 'PluggableStrategy') -> None:
        self.update(strategy.loss, len(strategy.mb_y), strategy.test_task_label)

    def after_test(self, strategy: 'PluggableStrategy') -> MetricResult:
        return self._package_result()

    def _package_result(self) -> MetricResult:
        metric_values = []
        for task_label, task_loss in self.result().items():
            metric_name = 'Task_Loss/Task{:03}'.format(task_label)
            plot_x_position = self._next_x_position(metric_name)

            metric_values.append(MetricValue(
                self, metric_name, task_loss, plot_x_position))
        return metric_values


__all__ = [
    'Loss',
    'MinibatchLoss',
    'EpochLoss',
    'RunningEpochLoss',
    'TaskLoss'
]
