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
from typing import TYPE_CHECKING

from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics.mean import Mean
if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class ElapsedTime(Metric[float]):
    """
    The elapsed time metric.

    Instances of this metric keep track of the time elapsed between calls to the
    `update` method. The starting time is set when the `update` method is called
    for the first time. That is, the starting time is *not* taken at the time
    the constructor is invoked.

    Calling the `update` method more than twice will update the metric to the
    elapsed time between the first and the last call to `update`.

    The result, obtained using the `result` method, is the time, in seconds,
    computed as stated above.

    The `reset` method will set the metric to its initial state, thus resetting
    the initial time. This metric in its initial state (or if the `update`
    method was invoked only once) will return an elapsed time of 0.
    """
    def __init__(self):
        """
        Creates an instance of the accuracy metric.

        This metric in its initial state (or if the `update` method was invoked
        only once) will return an elapsed time of 0. The metric can be updated
        by using the `update` method while the running accuracy can be retrieved
        using the `result` method.
        """
        self._init_time = None
        self._prev_time = None

    def update(self) -> None:
        """
        Update the elapsed time.

        For more info on how to set the initial time see the class description.

        :return: None.
        """
        now = time.perf_counter()
        if self._init_time is None:
            self._init_time = now
        self._prev_time = now

    def result(self) -> float:
        """
        Retrieves the elapsed time.

        Calling this method will not change the internal state of the metric.

        :return: The elapsed time, in seconds, as a float value.
        """
        if self._init_time is None:
            return 0.0
        return self._prev_time - self._init_time

    def reset(self) -> None:
        """
        Resets the metric, including the initial time.

        :return: None.
        """
        self._prev_time = None
        self._init_time = None


class MinibatchTime(PluginMetric[float]):
    """
    The minibatch time metric.

    This metric "logs" the elapsed time for each iteration. Beware that this
    metric will not average the time across minibatches!

    If a more coarse-grained logging is needed, consider using
    :class:`EpochTime`, :class:`AverageEpochTime` or
    :class:`StepTime` instead.
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the minibatch time metric.

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
                             ' time.')
        self._minibatch_time = ElapsedTime()
        self._compute_train_time = train
        self._compute_test_time = test

    def result(self) -> float:
        return self._minibatch_time.result()

    def reset(self) -> None:
        self._minibatch_time.reset()

    def before_training_iteration(self, strategy) -> MetricResult:
        if not self._compute_train_time:
            return
        self.reset()
        self._minibatch_time.update()

    def before_test_iteration(self, strategy) -> MetricResult:
        if not self._compute_test_time:
            return
        self.reset()
        self._minibatch_time.update()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._compute_train_time:
            self._minibatch_time.update()
            return self._package_result(strategy)

    def after_test_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._compute_test_time:
            self._minibatch_time.update()
            return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'Time_MB/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class EpochTime(PluginMetric[float]):
    """
    The epoch elapsed time metric.

    The elapsed time will be logged after each epoch. Beware that this
    metric will not average the time across epochs!

    If logging the average time across epochs is needed, consider using
    :class:`AverageEpochTime` instead.
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the epoch time metric.

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
                             ' time.')

        self._elapsed_time = ElapsedTime()
        self._take_train_time = train
        self._take_test_time = test

    def before_training_epoch(self, strategy) -> MetricResult:
        if not self._take_train_time:
            return
        self.reset()
        self._elapsed_time.update()

    def before_test_step(self, strategy) -> MetricResult:
        if not self._take_test_time:
            return
        self.reset()
        self._elapsed_time.update()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._take_train_time:
            self._elapsed_time.update()
            return self._package_result(strategy)

    def after_test_step(self, strategy: 'PluggableStrategy') -> MetricResult:
        if self._take_test_time:
            self._elapsed_time.update()
            return self._package_result(strategy)

    def reset(self) -> None:
        self._elapsed_time.reset()

    def result(self) -> float:
        return self._elapsed_time.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        elapsed_time = self.result()

        metric_name = 'Epoch_Time/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, elapsed_time, plot_x_position)]


class AverageEpochTime(PluginMetric[float]):
    """
    The average epoch time metric.

    The average elapsed time will be logged at the end of the step.

    Beware that this metric will average the time across epochs! If logging the
    epoch-specific time is needed, consider using :class:`EpochTime` instead.
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the average epoch time metric.

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
                             ' time.')

        self._time_mean = Mean()
        self._epoch_time = ElapsedTime()
        self._take_train_time = train
        self._take_test_time = test

    def before_training_epoch(self, strategy) -> MetricResult:
        if not self._take_train_time:
            return
        self._epoch_time.reset()
        self._epoch_time.update()

    def before_test_step(self, strategy) -> MetricResult:
        if not self._take_test_time:
            return
        self.reset()
        self._epoch_time.reset()
        self._epoch_time.update()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if not self._take_train_time:
            return
        self._epoch_time.update()
        self._time_mean.update(self._epoch_time.result())
        return self._package_result(strategy)

    def after_test_step(self, strategy: 'PluggableStrategy') -> MetricResult:
        if not self._take_test_time:
            return
        self._epoch_time.update()
        self._time_mean.update(self._epoch_time.result())
        return self._package_result(strategy)

    def reset(self) -> None:
        self._epoch_time.reset()
        self._time_mean.reset()

    def result(self) -> float:
        return self._time_mean.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        average_epoch_time = self.result()

        metric_name = 'Avg_Epoch_Time/{}/Task{:03}'.format(
            phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(
            self, metric_name, average_epoch_time, plot_x_position)]


class StepTime(PluginMetric[float]):
    """
    The step time metric.

    This metric may seem very similar to :class:`AverageEpochTime`. However,
    differently from that: 1) obviously, the time is not averaged by dividing
    by the number of epochs; 2) most importantly, the time consumed outside the
    epoch loop is accounted too (a thing that :class:`AverageEpochTime` doesn't
    support). For instance, this metric is more suitable when measuring times
    of algorithms involving after-training consolidation, replay pattern
    selection and other time consuming mechanisms.
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the step time metric.

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
                             ' time.')

        self._elapsed_time = ElapsedTime()
        self._take_train_time = train
        self._take_test_time = test

    def before_training_step(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if not self._take_train_time:
            return
        self.reset()
        self._elapsed_time.update()

    def before_test_step(self, strategy: 'PluggableStrategy') -> MetricResult:
        if not self._take_test_time:
            return
        self.reset()
        self._elapsed_time.update()

    def after_training_step(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._take_train_time:
            self._elapsed_time.update()
            return self._package_result(strategy)

    def after_test_step(self, strategy: 'PluggableStrategy') -> MetricResult:
        if self._take_test_time:
            self._elapsed_time.update()
            return self._package_result(strategy)

    def reset(self) -> None:
        self._elapsed_time.reset()

    def result(self) -> float:
        return self._elapsed_time.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        step_time = self.result()

        metric_name = 'Step_Time/{}/Task{:03}'.format(
            phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, step_time, plot_x_position)]


__all__ = [
    'ElapsedTime',
    'MinibatchTime',
    'EpochTime',
    'AverageEpochTime',
    'StepTime'
]
