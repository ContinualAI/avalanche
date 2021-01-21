from abc import ABC, abstractmethod

from typing import List, Literal, Union

from avalanche.evaluation.metric_results import MetricValue
from avalanche.extras.trace import StrategyTrace, DefaultStrategyTrace
from avalanche.training.plugins import PluggableStrategy
from avalanche.training.strategies.strategy_callbacks import StrategyCallbacks


TraceType = Union[None, StrategyTrace, Literal['default']]
TraceList = Union[TraceType, List[TraceType]]


class StrategyLogger(StrategyCallbacks, ABC):
    # TODO: doc

    def __init__(self, text_trace: TraceList = None):
        super().__init__()
        if text_trace == 'default':
            text_trace = [DefaultStrategyTrace()]

        if text_trace is None:
            text_trace = []

        if not isinstance(text_trace, List):
            text_trace = list(text_trace)

        for i in range(len(text_trace)):
            if text_trace[i] == 'default':
                text_trace[i] = DefaultStrategyTrace()
        self.text_trace: List[StrategyTrace] = text_trace

    @abstractmethod
    def log_metric(self, metric_value: 'MetricValue'):
        for trace in self.text_trace:
            trace.log_metric(metric_value)

    def before_training(self, strategy: PluggableStrategy,
                        metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.before_training(strategy, metric_values, **kwargs)

    def before_training_step(self, strategy: PluggableStrategy,
                             metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.before_training_step(strategy, metric_values, **kwargs)

    def adapt_train_dataset(self, strategy: PluggableStrategy,
                            metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.adapt_train_dataset(strategy, metric_values, **kwargs)

    def before_training_epoch(self, strategy: PluggableStrategy,
                              metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.before_training_epoch(strategy, metric_values, **kwargs)

    def before_training_iteration(self, strategy: PluggableStrategy,
                                  metric_values: List['MetricValue'],
                                  **kwargs):
        for trace in self.text_trace:
            trace.before_training_iteration(strategy, metric_values, **kwargs)

    def before_forward(self, strategy: PluggableStrategy,
                       metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.before_forward(strategy, metric_values, **kwargs)

    def after_forward(self, strategy: PluggableStrategy,
                      metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_forward(strategy, metric_values, **kwargs)

    def before_backward(self, strategy: PluggableStrategy,
                        metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.before_backward(strategy, metric_values, **kwargs)

    def after_backward(self, strategy: PluggableStrategy,
                       metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_backward(strategy, metric_values, **kwargs)

    def after_training_iteration(self, strategy: PluggableStrategy,
                                 metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_training_iteration(strategy, metric_values, **kwargs)

    def before_update(self, strategy: PluggableStrategy,
                      metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.before_update(strategy, metric_values, **kwargs)

    def after_update(self, strategy: PluggableStrategy,
                     metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_update(strategy, metric_values, **kwargs)

    def after_training_epoch(self, strategy: PluggableStrategy,
                             metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_training_epoch(strategy, metric_values, **kwargs)

    def after_training_step(self, strategy: PluggableStrategy,
                            metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_training_step(strategy, metric_values, **kwargs)

    def after_training(self, strategy: PluggableStrategy,
                       metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_training(strategy, metric_values, **kwargs)

    def before_test(self, strategy: PluggableStrategy,
                    metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.before_test(strategy, metric_values, **kwargs)

    def adapt_test_dataset(self, strategy: PluggableStrategy,
                           metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.adapt_test_dataset(strategy, metric_values, **kwargs)

    def before_test_step(self, strategy: PluggableStrategy,
                         metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.before_test_step(strategy, metric_values, **kwargs)

    def after_test_step(self, strategy: PluggableStrategy,
                        metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_test_step(strategy, metric_values, **kwargs)

    def after_test(self, strategy: PluggableStrategy,
                   metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_test(strategy, metric_values, **kwargs)

    def before_test_iteration(self, strategy: PluggableStrategy,
                              metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.before_test_iteration(strategy, metric_values, **kwargs)

    def before_test_forward(self, strategy: PluggableStrategy,
                            metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.before_test_forward(strategy, metric_values, **kwargs)

    def after_test_forward(self, strategy: PluggableStrategy,
                           metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_test_forward(strategy, metric_values, **kwargs)

    def after_test_iteration(self, strategy: PluggableStrategy,
                             metric_values: List['MetricValue'], **kwargs):
        for trace in self.text_trace:
            trace.after_test_iteration(strategy, metric_values, **kwargs)


__all__ = [
    'StrategyLogger',
    'TraceList'
]
