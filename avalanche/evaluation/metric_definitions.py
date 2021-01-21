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
from typing import TypeVar, Optional, Dict, TYPE_CHECKING

from typing_extensions import Protocol

if TYPE_CHECKING:
    from .metric_results import MetricResult
    from avalanche.training.plugins import PluggableStrategy

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
    def __init__(self, log_to_board: bool, log_to_text: bool):
        """
        Creates an instance of a plugin metric.

        Child classes can safely invoke this (super) constructor as the first
        step.

        :param log_to_board: If True, the logger will log the metric value
            on the dashboard (for instance, :class:`TensorboardLogger`).
        :param log_to_text: If True, the values emitted by this metric will be
            sent the strategy trace instance(s) to be printed in a text form
            (for an example see :class:`DotTrace`).
        """
        self._metric_x_counters: Dict[str, int] = dict()

        self.log_to_board: bool = log_to_board
        """
        If True, the logger will log the metric value on the dashboard 
        (for instance, :class:`TensorboardLogger`).
        """

        self.log_to_text: bool = log_to_text
        """
        If True, the values emitted by this metric will be sent the strategy 
        trace instance(s) to be printed in a text form (for an example see 
        :class:`DotTrace`).
        """

    @abstractmethod
    def result(self) -> Optional[TResult]:
        return None

    @abstractmethod
    def reset(self) -> None:
        return None

    def before_training(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def before_training_step(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def adapt_train_dataset(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def before_training_epoch(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def before_training_iteration(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def before_forward(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def after_forward(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def before_backward(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def after_backward(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def before_update(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def after_update(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def after_training_step(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def after_training(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def before_test(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def adapt_test_dataset(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def before_test_step(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def after_test_step(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def after_test(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def before_test_iteration(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def before_test_forward(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def after_test_forward(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return None

    def after_test_iteration(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
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
