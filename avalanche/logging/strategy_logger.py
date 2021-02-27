from abc import ABC

from typing import List

from avalanche.evaluation.metric_results import MetricValue
from avalanche.training.plugins import PluggableStrategy
from avalanche.training.strategy_callbacks import StrategyCallbacks


class StrategyLogger(StrategyCallbacks[None], ABC):
    """
    The base class for the strategy loggers.

    Strategy loggers will receive events, under the form of callback calls,
    from the :class:`EvaluationPlugin` carrying a reference to the strategy
    as well as the values emitted by the metrics.

    Child classes can implement the desired callbacks. An alternative, simpler,
    mechanism exists: child classes may instead implement the `log_metric`
    method which will be invoked with each received metric value.

    Implementing `log_metric` is not mutually exclusive with the callback
    implementation. Make sure, when implementing the callbacks, to call
    the proper super method.
    """

    def __init__(self):
        super().__init__()

    def log_metric(self, metric_value: 'MetricValue', callback: str) -> None:
        """
        Helper method that will be invoked each time a metric value will become
        available. To know from which callback the value originated, the
        callback parameter can be used.

        Implementing this method is a practical, non-exclusive, alternative the
        implementation of the single callbacks. See the class description for
        details and hints.

        :param metric_value: The value to be logged.
        :param callback: The callback (event) from which the metric value was
            obtained.
        :return: None
        """
        pass

    def before_training(self, strategy: PluggableStrategy,
                        metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_training')

    def before_training_exp(self, strategy: PluggableStrategy,
                            metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_training_exp')

    def adapt_train_dataset(self, strategy: PluggableStrategy,
                            metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'adapt_train_dataset')

    def before_training_epoch(self, strategy: PluggableStrategy,
                              metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_training_epoch')

    def before_training_iteration(self, strategy: PluggableStrategy,
                                  metric_values: List['MetricValue'],
                                  **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_training_iteration')

    def before_forward(self, strategy: PluggableStrategy,
                       metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_forward')

    def after_forward(self, strategy: PluggableStrategy,
                      metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_forward')

    def before_backward(self, strategy: PluggableStrategy,
                        metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_backward')

    def after_backward(self, strategy: PluggableStrategy,
                       metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_backward')

    def after_training_iteration(self, strategy: PluggableStrategy,
                                 metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_training_iteration')

    def before_update(self, strategy: PluggableStrategy,
                      metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_update')

    def after_update(self, strategy: PluggableStrategy,
                     metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_update')

    def after_training_epoch(self, strategy: PluggableStrategy,
                             metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_training_epoch')

    def after_training_exp(self, strategy: PluggableStrategy,
                           metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_training_exp')

    def after_training(self, strategy: PluggableStrategy,
                       metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_training')

    def before_eval(self, strategy: PluggableStrategy,
                    metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_eval')

    def adapt_eval_dataset(self, strategy: PluggableStrategy,
                           metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'adapt_eval_dataset')

    def before_eval_exp(self, strategy: PluggableStrategy,
                        metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_eval_exp')

    def after_eval_exp(self, strategy: PluggableStrategy,
                       metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_eval_exp')

    def after_eval(self, strategy: PluggableStrategy,
                   metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_eval')

    def before_eval_iteration(self, strategy: PluggableStrategy,
                              metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_eval_iteration')

    def before_eval_forward(self, strategy: PluggableStrategy,
                            metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'before_eval_forward')

    def after_eval_forward(self, strategy: PluggableStrategy,
                           metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_eval_forward')

    def after_eval_iteration(self, strategy: PluggableStrategy,
                             metric_values: List['MetricValue'], **kwargs):
        for val in metric_values:
            self.log_metric(val, 'after_eval_iteration')


__all__ = [
    'StrategyLogger'
]
