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

from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Dict, TYPE_CHECKING

from typing_extensions import Protocol

from ..core import StrategyCallbacks

if TYPE_CHECKING:
    from .metric_results import MetricResult
    from ..training import BaseStrategy

TResult = TypeVar('TResult')
TAggregated = TypeVar('TAggregated', bound='PluginMetric')


class Metric(Protocol[TResult]):
    """
    Definition of a standalone metric.

    A standalone metric exposes methods to reset its internal state and
    to emit a result. Emitting a result does not automatically cause
    a reset in the internal state.

    The specific metric implementation exposes ways to update the internal
    state. Usually, standalone metrics like :class:`Sum`, :class:`Mean`,
    :class:`Accuracy`, ... expose an `update` method.

    The `Metric` class can be used as a standalone metric by directly calling
    its methods.
    In order to automatically integrate the metric with the training and
    evaluation flows, you can use :class:`PluginMetric` class. The class
    receives events directly from the :class:`EvaluationPlugin` and can
    emits values on each callback. Usually, an instance of `Metric` is
    created within `PluginMetric`, which is then responsible for its
    update and results. See :class:`PluginMetric` for more details.
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


class PluginMetric(Metric[TResult], StrategyCallbacks['MetricResult'], ABC):
    """
    A metric that can be used together with :class:`EvaluationPlugin`.

    This class leaves the implementation of the `result` and `reset` methods
    to child classes while providing an empty implementation of the callbacks
    invoked by the :class:`EvaluationPlugin`. Subclasses should implement
    the `result`, `reset` and the desired callbacks to compute the specific
    metric.
    An instance of this class usually leverages a `Metric` instance to update,
    reset and emit metric results at appropriate times
    (during specific callbacks).

    This class also provides a utility method, `_next_x_position`, which can
    be used to progressively label each metric value with its appropriate "x"
    position in the plot.
    """
    def __init__(self):
        """
        Creates an instance of a plugin metric.

        Child classes can safely invoke this (super) constructor as the first
        experience.
        """
        super().__init__()
        self._metric_x_counters: Dict[str, int] = dict()

    @abstractmethod
    def result(self) -> Optional[TResult]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def before_training(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def before_training_exp(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def after_train_dataset_adaptation(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def before_training_epoch(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def before_training_iteration(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def before_forward(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def after_forward(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def before_backward(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def after_backward(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def after_training_iteration(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def before_update(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def after_update(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def after_training_epoch(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def after_training_exp(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def after_training(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def before_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def before_eval_dataset_adaptation(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def after_eval_dataset_adaptation(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def before_eval_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def after_eval_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        pass

    def before_eval_iteration(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def before_eval_forward(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def after_eval_forward(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

    def after_eval_iteration(self, strategy: 'BaseStrategy') \
            -> 'MetricResult':
        pass

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
