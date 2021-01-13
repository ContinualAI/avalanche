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
from abc import ABC, abstractmethod
from typing_extensions import Protocol
from typing import TypeVar, Optional, Dict, TYPE_CHECKING

from .evaluation_data import OnTrainPhaseStart, OnTrainPhaseEnd, \
    OnTrainStepStart, OnTrainStepEnd, OnTrainEpochStart, OnTrainEpochEnd, \
    OnTrainIterationEnd, OnTestStepStart, OnTestStepEnd, OnTestIterationEnd, \
    OnTestPhaseStart, OnTestPhaseEnd, OnTrainIterationStart, \
    OnTestIterationStart

if TYPE_CHECKING:
    from .metric_results import MetricResult

TResult = TypeVar('TResult')
TAggregated = TypeVar('TAggregated', bound='PluginMetric')


class Metric(Protocol[TResult]):
    """
    Definition of a metric.

    A metric exposes methods to reset the internal counters as well as
    a method used to retrieve the result.

    The specific metric implementation exposes ways to update the metric
    state. Usually, standalone metrics like :class:`Sum`, :class:`Mean`,
    :class:`Accuracy`, ... expose an `update` method.

    On the other hand, metrics that are to be used by using the
    :class:`EvaluationPlugin` usually update their value by receiving events
    from the plugin (see the :class:`PluginMetric` class for more details).
    """

    def result(self) -> Optional[TResult]:
        """
        Obtains the value of the metric.

        :return: The value of the metric.
        """
        pass

    def reset(self) -> None:
        """
        Resets the metric internal state.

        :return: None.
        """
        pass


class PluginMetric(Metric[TResult], ABC):
    """
    A kind of metric that can be used by the :class:`EvaluationPlugin`.

    This class leaves the implementation of the `result` and `reset` methods
    to child classes while providing an empty implementation of the callbacks
    invoked by the :class:`EvaluationPlugin`. Subclasses should implement
    the `result`, `reset` and the desired callbacks to compute the specific
    metric.

    This class also provides a utility method, `_next_x_position`, which can
    be used to label each metric value with its appropriate "x" position in the
    plot.
    """
    def __init__(self):
        """
        Creates an instance of a plugin metric.

        Child classes can safely invoke this (super) constructor as the first
        step.
        """
        self._metric_x_counters: Dict[str, int] = dict()

    @abstractmethod
    def result(self) -> Optional[TResult]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def before_training(self, eval_data: OnTrainPhaseStart) -> 'MetricResult':
        return None

    def after_training(self, eval_data: OnTrainPhaseEnd) -> 'MetricResult':
        return None

    def before_training_step(self, eval_data: OnTrainStepStart) \
            -> 'MetricResult':
        return None

    def after_training_step(self, eval_data: OnTrainStepEnd) -> 'MetricResult':
        return None

    def before_training_epoch(self, eval_data: OnTrainEpochStart) \
            -> 'MetricResult':
        return None

    def after_training_epoch(self, eval_data: OnTrainEpochEnd) \
            -> 'MetricResult':
        return None

    def before_training_iteration(self, eval_data: OnTrainIterationStart) \
            -> 'MetricResult':
        return None

    def after_training_iteration(self, eval_data: OnTrainIterationEnd) \
            -> 'MetricResult':
        return None

    def before_test_step(self, eval_data: OnTestStepStart) -> 'MetricResult':
        return None

    def after_test_step(self, eval_data: OnTestStepEnd) -> 'MetricResult':
        return None

    def before_test_iteration(self, eval_data: OnTestIterationStart) \
            -> 'MetricResult':
        return None

    def after_test_iteration(self, eval_data: OnTestIterationEnd) \
            -> 'MetricResult':
        return None

    def before_test(self, eval_data: OnTestPhaseStart) -> 'MetricResult':
        return None

    def after_test(self, eval_data: OnTestPhaseEnd) -> 'MetricResult':
        return None

    def _next_x_position(self, metric_name: str, initial_x: int = 0) -> int:
        """
        Utility method that can be used to get the next "x" position of a
        metric value (given its name).

        :param metric_name: The metric value name.
        :param initial_x: The initial "x" value. Defaults to 0.
        :return: The next "x" value to use.
        """
        if metric_name not in self._metric_x_counters:
            self._metric_x_counters[metric_name] = initial_x
        x_result = self._metric_x_counters[metric_name]
        self._metric_x_counters[metric_name] += 1
        return x_result


__all__ = ['Metric', 'PluginMetric']
