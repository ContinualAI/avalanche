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

from typing import Union

import torch
from torch import Tensor

from avalanche.evaluation import OnTrainEpochEnd, OnTestStepEnd, \
    OnTrainIterationEnd, OnTestIterationEnd, PluginMetric, EvalData,\
    AggregatedMetric, OnTrainEpochStart, OnTestStepStart, Metric
from avalanche.evaluation.metric_results import MetricTypes, MetricValue, \
    MetricResult
from avalanche.evaluation.metric_utils import get_task_label
from avalanche.evaluation.metrics.mean import Mean


class Loss(Metric[float]):

    def __init__(self):
        self._mean_loss = Mean()

    @torch.no_grad()
    def update(self, loss: Tensor, weight: float = 1.0) -> None:
        self._mean_loss.update(torch.mean(loss), weight=weight)

    def result(self) -> float:
        return self._mean_loss.result()

    def reset(self) -> None:
        self._mean_loss.reset()


class MinibatchLoss(PluginMetric[float]):
    """
    The average accuracy metric.

    This metric is computed separately for each task.

    The accuracy will be emitted after each epoch by aggregating minibatch
    values. Beware that the training accuracy is the "running" one.
    TODO: doc
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the EpochAccuracy metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to True.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')
        self._minibatch_loss = Loss()
        self._compute_train_loss = train
        self._compute_test_loss = test

    def result(self) -> float:
        return self._minibatch_loss.result()

    def reset(self) -> None:
        self._minibatch_loss.reset()

    def after_training_iteration(self, eval_data: OnTrainIterationEnd) \
            -> MetricResult:
        if self._compute_train_loss:
            return self._on_iteration(eval_data)

    def after_test_iteration(self, eval_data: OnTestIterationEnd) -> MetricResult:
        if self._compute_test_loss:
            return self._on_iteration(eval_data)

    def _on_iteration(self, eval_data: Union[OnTrainIterationEnd,
                                             OnTestIterationEnd]):
        self.reset()  # Because this metric computes the loss of a single mb
        self._minibatch_loss.update(eval_data.loss,
                                    weight=len(eval_data.ground_truth))
        return self._package_result(eval_data)

    def _package_result(self, eval_data: EvalData) -> MetricResult:
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        metric_value = self.result()

        metric_name = 'Loss_MB/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, MetricTypes.LOSS,
                            metric_value, plot_x_position)]


class EpochLoss(AggregatedMetric[float, MinibatchLoss]):
    """
    The average accuracy metric.

    This metric is computed separately for each task.

    The accuracy will be emitted after each epoch by aggregating minibatch
    values. Beware that the training accuracy is the "running" one.
    TODO: doc
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the EpochAccuracy metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to True.
        """
        super().__init__(MinibatchLoss(train=train, test=test))

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')

        self._weighted_mean = Mean()
        self._compute_train_accuracy = train
        self._compute_test_accuracy = test

    def before_training_epoch(self,
                              eval_data: OnTrainEpochStart) -> MetricResult:
        super().before_training_epoch(eval_data)
        if not self._compute_train_accuracy:
            return
        self.reset()

    def before_test_step(self, eval_data: OnTestStepStart) -> MetricResult:
        super().before_test_step(eval_data)
        if not self._compute_test_accuracy:
            return
        self.reset()

    def after_training_iteration(self, eval_data: OnTrainIterationEnd) \
            -> MetricResult:
        super().after_training_iteration(eval_data)
        if not self._compute_train_accuracy:
            return

        self._weighted_mean.update(self.base_metric.result(),
                                   weight=len(eval_data.ground_truth))

    def after_test_iteration(self, eval_data: OnTestIterationEnd) \
            -> MetricResult:
        super().after_test_iteration(eval_data)
        if not self._compute_test_accuracy:
            return

        self.base_metric.after_test_iteration(eval_data)
        self._weighted_mean.update(self.base_metric.result(),
                                   weight=len(eval_data.ground_truth))

    def after_training_epoch(self, eval_data: OnTrainEpochEnd) -> MetricResult:
        super().after_training_epoch(eval_data)
        if self._compute_train_accuracy:
            return self._package_result(eval_data)

    def after_test_step(self, eval_data: OnTestStepEnd) -> MetricResult:
        super().after_test_step(eval_data)
        if self._compute_test_accuracy:
            return self._package_result(eval_data)

    def reset(self) -> None:
        super().reset()
        self._weighted_mean.reset()

    def result(self) -> float:
        return self._weighted_mean.result()

    def _package_result(self, eval_data: EvalData) -> MetricResult:
        eval_data: Union[OnTrainEpochEnd, OnTestStepEnd]
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        metric_value = self.result()

        metric_name = 'Loss/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, MetricTypes.LOSS,
                            metric_value, plot_x_position)]


class RunningEpochLoss(AggregatedMetric[float, EpochLoss]):
    """
    The running average accuracy metric.

    Differently from :class:`EpochAccuracy`, this metric will emit a value
    after each iteration, too. The metric value will be also emitted on
    "train epoch end" and "test step end" events, exactly as
    :class:`EpochAccuracy`.
    TODO: doc
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the RunningEpochAccuracy metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to True.
        """
        super().__init__(EpochLoss(train=train, test=test))

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')

        self._compute_train_accuracy = train
        self._compute_test_accuracy = test

    def result(self) -> float:
        return self.base_metric.result()

    def after_training_iteration(self, eval_data: OnTrainIterationEnd) \
            -> MetricResult:
        super().after_training_iteration(eval_data)
        if not self._compute_train_accuracy:
            return

        return self._package_result(eval_data)

    def after_test_iteration(self, eval_data: OnTestIterationEnd) \
            -> MetricResult:
        super().after_test_iteration(eval_data)
        if not self._compute_test_accuracy:
            return

        return self._package_result(eval_data)

    def _package_result(self, eval_data: EvalData) -> MetricResult:
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        metric_value = self.result()

        metric_name = 'Loss_Running/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, MetricTypes.LOSS,
                            metric_value, plot_x_position)]


__all__ = [
    'Loss',
    'MinibatchLoss',
    'EpochLoss',
    'RunningEpochLoss'
]
