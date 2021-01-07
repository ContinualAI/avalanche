#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

import time
from typing import Union, SupportsFloat

from avalanche.evaluation import OnTrainEpochStart, OnTestStepStart, \
    OnTrainEpochEnd, OnTestStepEnd, OnTrainStepStart, OnTrainStepEnd, Metric, \
    PluginMetric, OnTrainIterationStart, OnTestIterationStart, \
    OnTrainIterationEnd, OnTestIterationEnd, AggregatedMetric, EvalData
from avalanche.evaluation.metric_results import MetricTypes, MetricValue, \
    MetricResult
from avalanche.evaluation.abstract_metric import AbstractMetric
from avalanche.evaluation.metric_utils import filter_accepted_events, \
    get_task_label
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metrics.sum import Sum


class ElapsedTime(Metric[float]):

    def __init__(self):
        self._sum_time = Sum()
        self._prev_time = None

    def update(self) -> None:
        now = time.perf_counter()
        if self._prev_time is not None:
            self._sum_time.update(now - self._prev_time)
        self._prev_time = now

    def result(self) -> float:
        return self._sum_time.result()

    def reset(self) -> None:
        self._prev_time = None
        self._sum_time.reset()


class MinibatchTime(PluginMetric[float]):
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
        self._minibatch_time = ElapsedTime()
        self._compute_train_time = train
        self._compute_test_time = test

    def result(self) -> float:
        return self._minibatch_time.result()

    def reset(self) -> None:
        self._minibatch_time.reset()

    def before_training_iteration(self, eval_data) -> MetricResult:
        if not self._compute_train_time:
            return
        self.reset()
        self._minibatch_time.update()

    def before_test_iteration(self, eval_data) -> MetricResult:
        if not self._compute_test_time:
            return
        self.reset()
        self._minibatch_time.update()

    def after_training_iteration(self, eval_data: OnTrainIterationEnd) \
            -> MetricResult:
        if self._compute_train_time:
            self._minibatch_time.update()
            return self._on_iteration(eval_data)

    def after_test_iteration(self, eval_data: OnTestIterationEnd) \
            -> MetricResult:
        if self._compute_test_time:
            self._minibatch_time.update()
            return self._on_iteration(eval_data)

    def _on_iteration(self, eval_data: Union[OnTrainIterationEnd,
                                             OnTestIterationEnd]):
        self._last_mb_time = self._minibatch_time.result()
        return self._package_result(eval_data)

    def _package_result(self, eval_data: EvalData) -> MetricResult:
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        metric_value = self.result()

        metric_name = 'Time_MB/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, MetricTypes.LOSS,
                            metric_value, plot_x_position)]


class EpochTime(PluginMetric[float]):
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

        self._elapsed_time = ElapsedTime()
        self._take_train_time = train
        self._take_test_time = test

    def before_training_epoch(self, eval_data) -> MetricResult:
        if not self._take_train_time:
            return
        self.reset()

    def before_test_step(self, eval_data) -> MetricResult:
        if not self._take_test_time:
            return
        self.reset()

    def after_training_epoch(self, eval_data: OnTrainEpochEnd) -> MetricResult:
        if self._take_train_time:
            self._elapsed_time.update()
            return self._package_result(eval_data)

    def after_test_step(self, eval_data: OnTestStepEnd) -> MetricResult:
        if self._take_test_time:
            self._elapsed_time.update()
            return self._package_result(eval_data)

    def reset(self) -> None:
        self._elapsed_time.reset()

    def result(self) -> float:
        return self._elapsed_time.result()

    def _package_result(self, eval_data: EvalData) -> MetricResult:
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        elapsed_time = self.result()

        metric_name = 'Epoch_Time/{}/Task{:03}'.format(phase_name,
                                                       task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, MetricTypes.ELAPSED_TIME,
                            elapsed_time, plot_x_position)]


class AverageEpochTime(AggregatedMetric[float, EpochTime]):
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
        super().__init__(EpochTime(train=train, test=test))

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')

        self._time_mean = Mean()
        self._take_train_time = train
        self._take_test_time = test

    def after_training_epoch(self, eval_data: OnTrainEpochEnd) -> MetricResult:
        super().after_training_epoch(eval_data)
        if self._take_train_time:
            return self._package_result(eval_data)

    def after_test_step(self, eval_data: OnTestStepEnd) -> MetricResult:
        super().after_test_step(eval_data)
        if self._take_test_time:
            return self._package_result(eval_data)

    def reset(self) -> None:
        super().reset()
        self._time_mean.reset()

    def result(self) -> float:
        return self._time_mean.result()

    def _package_result(self, eval_data: EvalData) -> MetricResult:
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        average_epoch_time = self.result()

        metric_name = 'Avg_Epoch_Time/{}/Task{:03}'.format(
            phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, MetricTypes.ELAPSED_TIME,
                            average_epoch_time, plot_x_position)]


class StepTime(PluginMetric[float]):
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

        self._elapsed_time = ElapsedTime()
        self._take_train_time = train
        self._take_test_time = test

    def before_training_step(self, eval_data) -> MetricResult:
        if not self._take_train_time:
            return
        self.reset()
        self._elapsed_time.update()

    def before_test_step(self, eval_data) -> MetricResult:
        if not self._take_test_time:
            return
        self.reset()
        self._elapsed_time.update()

    def after_training_step(self, eval_data) -> MetricResult:
        if self._take_train_time:
            self._elapsed_time.update()
            return self._package_result(eval_data)

    def after_test_step(self, eval_data: OnTestStepEnd) -> MetricResult:
        if self._take_test_time:
            self._elapsed_time.update()
            return self._package_result(eval_data)

    def reset(self) -> None:
        self._elapsed_time.reset()

    def result(self) -> float:
        return self._elapsed_time.result()

    def _package_result(self, eval_data: EvalData) -> MetricResult:
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        step_time = self.result()

        metric_name = 'Step_Time/{}/Task{:03}'.format(
            phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, MetricTypes.ELAPSED_TIME,
                            step_time, plot_x_position)]


__all__ = [
    'ElapsedTime',
    'MinibatchTime',
    'EpochTime',
    'AverageEpochTime',
    'StepTime'
]
