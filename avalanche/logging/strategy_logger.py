from abc import ABC

from typing import List

from avalanche.evaluation.metric_results import MetricValue
from avalanche.training import PluggableStrategy
from avalanche.training.strategy_callbacks import StrategyCallbacks


class StrategyLogger(StrategyCallbacks[None], ABC):
    """
    The base class for the strategy loggers.

    Strategy loggers will receive events, under the form of callback calls,
    from the :class:`EvaluationPlugin` carrying a reference to the strategy
    as well as the values emitted by the metrics.

    Each child class should implement the `log_metric` method, which
    specifies how to report to the user the metrics gathered during
    training and evaluation flows. The `log_metric` method is invoked
    by default on each callback.
    In addition, child classes may override the desired callbacks
    to customize the logger behavior.

    Make sure, when overriding callbacks, to call
    the proper `super` method.
    """

    def __init__(self):
        super().__init__()

    def log_metric(self, metric_value: 'MetricValue', callback: str) -> None:
        """
        This abstract method will has to be implemented by child classes.
        This method will be invoked on each callback.
        The `callback` parameter describes the callback from which the metric
        value is coming from.

        :param metric_value: The value to be logged.
        :param callback: The name of the callback (event) from which the
            metric value was obtained.
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
