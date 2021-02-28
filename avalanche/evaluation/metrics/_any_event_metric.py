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
from typing import TYPE_CHECKING, TypeVar

from avalanche.evaluation import PluginMetric

if TYPE_CHECKING:
    from avalanche.evaluation.metric_results import MetricResult
    from avalanche.training.plugins import PluggableStrategy

TResult = TypeVar('TResult')


class AnyEventMetric(PluginMetric[TResult], ABC):
    """
    A metric that calls a given abstract method each time a new event is
    encountered. For internal use only.
    """
    def __init__(self):
        """
        Initializes the "AnyEventMetric" instance.

        Can be safely called by subclasses at the beginning of the
        initialization procedure.
        """
        super().__init__()

    @abstractmethod
    def on_event(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return None

    def before_training(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def before_training_exp(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def adapt_train_dataset(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def before_training_epoch(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def before_training_iteration(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def before_forward(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def after_forward(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def before_backward(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def after_backward(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def before_update(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def after_update(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def after_training_exp(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def after_training(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def before_eval(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def adapt_eval_dataset(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def before_eval_exp(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def after_eval_exp(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def after_eval(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        return self.on_event(strategy)

    def before_eval_iteration(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def before_eval_forward(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def after_eval_forward(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)

    def after_eval_iteration(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        return self.on_event(strategy)


__all__ = [
    'AnyEventMetric'
]
