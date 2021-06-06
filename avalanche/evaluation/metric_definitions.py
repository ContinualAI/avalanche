################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta, Andrea Cossu                   #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from abc import ABC, abstractmethod
from typing import TypeVar, Optional, TYPE_CHECKING
from typing_extensions import Protocol
from .metric_results import MetricValue
from .metric_utils import get_metric_name
from ..core import StrategyCallbacks

if TYPE_CHECKING:
    from .metric_results import MetricResult
    from ..training.strategies import BaseStrategy

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

    Remember to call the `super()` method when overriding
    `after_train_iteration` or `after_eval_iteration`.

    An instance of this class usually leverages a `Metric` instance to update,
    reset and emit metric results at appropriate times
    (during specific callbacks).
    """
    def __init__(self):
        """
        Creates an instance of a plugin metric.

        Child classes can safely invoke this (super) constructor as the first
        experience.
        """
        super().__init__()

        self.global_it_counter = 0
        """
        Counter that can be used by each metric to get increasing x values.
        """

    @abstractmethod
    def result(self) -> Optional[TResult]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def get_global_counter(self):
        """
        :return: the global counter incremented after each minibatch.
        """
        return self.global_it_counter

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
        self.global_it_counter += 1

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
        self.global_it_counter += 1


class GenericPluginMetric(PluginMetric[TResult]):
    """
    This class provides a generic implementation of a Plugin Metric.
    The user can subclass this class to easily implement custom plugin
    metrics.
    """
    def __init__(self, metric, reset_at='experience', emit_at='experience',
                 mode='eval'):
        super(GenericPluginMetric, self).__init__()
        assert mode in {'train', 'eval'}
        if mode == 'train':
            assert reset_at in {'iteration', 'epoch'}
            assert reset_at in {'iteration', 'epoch'}
        else:
            assert reset_at in {'experience', 'stream'}
            assert reset_at in {'experience', 'stream'}
        self._metric = metric
        self._reset_at = reset_at
        self._emit_at = emit_at
        self._mode = mode

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        assert NotImplementedError()

    def _package_result(self, strategy: 'BaseStrategy') -> 'MetricResult':
        metric_value = self.result()

        add_exp = self._emit_at == 'experience'
        add_task = self._emit_at != 'stream'
        metric_name = get_metric_name(self, strategy, add_experience=add_exp,
                                      add_task=add_task)
        plot_x_position = self.get_global_counter()

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def before_training(self, strategy: 'BaseStrategy'):
        super().before_training(strategy)
        if self._reset_at == 'stream' and self._mode == 'train':
            self.reset()

    def before_training_exp(self, strategy: 'BaseStrategy'):
        super().before_training_exp(strategy)
        if self._reset_at == 'experience' and self._mode == 'train':
            self.reset()

    def before_training_epoch(self, strategy: 'BaseStrategy'):
        super().before_training_epoch(strategy)
        if self._reset_at == 'epoch' and self._mode == 'train':
            self.reset()

    def before_training_iteration(self, strategy: 'BaseStrategy'):
        super().before_training_iteration(strategy)
        if self._reset_at == 'iteration' and self._mode == 'train':
            self.reset()

    def after_training_iteration(self, strategy: 'BaseStrategy') -> None:
        super().after_training_iteration(strategy)
        self.update(strategy)
        if self._emit_at == 'iteration' and self._mode == 'train':
            return self._package_result(strategy)

    def after_training_epoch(self, strategy: 'BaseStrategy'):
        super().after_training_epoch(strategy)
        if self._emit_at == 'epoch' and self._mode == 'train':
            return self._package_result(strategy)

    def after_training_exp(self, strategy: 'BaseStrategy'):
        super().after_training_exp(strategy)
        if self._emit_at == 'experience' and self._mode == 'train':
            return self._package_result(strategy)

    def after_training(self, strategy: 'BaseStrategy'):
        super().after_training(strategy)
        if self._emit_at == 'stream' and self._mode == 'train':
            return self._package_result(strategy)

    def before_eval(self, strategy: 'BaseStrategy'):
        super().before_eval(strategy)
        if self._reset_at == 'stream' and self._mode == 'eval':
            self.reset()

    def before_eval_exp(self, strategy: 'BaseStrategy'):
        super().before_eval_exp(strategy)
        if self._reset_at == 'experience' and self._mode == 'eval':
            self.reset()

    def after_eval_exp(self, strategy: 'BaseStrategy'):
        super().after_eval_exp(strategy)
        if self._emit_at == 'experience' and self._mode == 'eval':
            return self._package_result(strategy)

    def after_eval(self, strategy: 'BaseStrategy'):
        super().after_eval(strategy)
        if self._emit_at == 'stream' and self._mode == 'eval':
            return self._package_result(strategy)

    def before_eval_iteration(self, strategy: 'BaseStrategy'):
        super().before_eval_iteration(strategy)
        if self._emit_at == 'iteration' and self._mode == 'eval':
            return self._package_result(strategy)

    def after_eval_iteration(self, strategy: 'BaseStrategy'):
        super().after_eval_iteration(strategy)
        self.update(strategy)
        if self._reset_at == 'iteration' and self._mode == 'eval':
            self.reset()


__all__ = ['Metric', 'PluginMetric', 'GenericPluginMetric']
