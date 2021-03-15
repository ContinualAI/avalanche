from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Union, Sequence, TYPE_CHECKING

from avalanche.training.plugins.strategy_plugin import StrategyPlugin

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy


class EvaluationPlugin(StrategyPlugin):
    """
    An evaluation plugin that obtains relevant data from the
    training and eval loops of the strategy through callbacks.

    This plugin updates the given metrics and logs them using the provided
    loggers.
    """

    def __init__(self,
                 *metrics: Union['PluginMetric', Sequence['PluginMetric']],
                 loggers: Union['StrategyLogger',
                                Sequence['StrategyLogger']] = None,
                 collect_all=True):
        """
        Creates an instance of the evaluation plugin.

        :param metrics: The metrics to compute.
        :param loggers: The loggers to be used to log the metric values.
        :param collect_curves (bool): enables the collection of the metric
            curves. If True `self.metric_curves` stores all the values of
            each curve in a dictionary. Please disable this if you log large
            values (embeddings, parameters) and you want to reduce memory usage.
        """
        super().__init__()
        self.collect_all = collect_all
        flat_metrics_list = []
        for metric in metrics:
            if isinstance(metric, Sequence):
                flat_metrics_list += list(metric)
            else:
                flat_metrics_list.append(metric)
        self.metrics = flat_metrics_list

        if loggers is None:
            loggers = []
        elif not isinstance(loggers, Sequence):
            loggers = [loggers]
        self.loggers: Sequence['StrategyLogger'] = loggers

        if len(self.loggers) == 0:
            warnings.warn('No loggers specified, metrics will not be logged')

        # for each curve  store last emitted value (train/eval separated).
        self.current_metrics = {}
        if self.collect_all:
            # for each curve collect all emitted values.
            self.all_metrics = defaultdict(lambda: ([], []))
        else:
            self.all_metrics = None

    def _update_metrics(self, strategy: 'BaseStrategy', callback: str):
        metric_values = []
        for metric in self.metrics:
            metric_result = getattr(metric, callback)(strategy)
            if isinstance(metric_result, Sequence):
                metric_values += list(metric_result)
            elif metric_result is not None:
                metric_values.append(metric_result)

        for metric_value in metric_values:
            name = metric_value.name
            x = metric_value.x_plot
            val = metric_value.value
            self.current_metrics[name] = val
            if self.collect_all:
                self.all_metrics[name][0].append(x)
                self.all_metrics[name][1].append(val)

        for logger in self.loggers:
            getattr(logger, callback)(strategy, metric_values)
        return metric_values

    def before_training(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_training')

    def before_training_exp(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_training_exp')

    def adapt_train_dataset(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'adapt_train_dataset')

    def before_training_epoch(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_training_epoch')

    def before_training_iteration(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_training_iteration')

    def before_forward(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_forward')

    def after_forward(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_forward')

    def before_backward(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_backward')

    def after_backward(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_backward')

    def after_training_iteration(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_training_iteration')

    def before_update(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_update')

    def after_update(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_update')

    def after_training_epoch(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_training_epoch')

    def after_training_exp(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_training_exp')

    def after_training(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_training')
        self.current_metrics = {}  # reset current metrics

    def before_eval(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_eval')

    def adapt_eval_dataset(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'adapt_eval_dataset')

    def before_eval_exp(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_eval_exp')

    def after_eval_exp(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_eval_exp')

    def after_eval(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_eval')
        self.current_metrics = {}  # reset current metrics

    def before_eval_iteration(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_eval_iteration')

    def before_eval_forward(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'before_eval_forward')

    def after_eval_forward(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_eval_forward')

    def after_eval_iteration(self, strategy: BaseStrategy, **kwargs):
        self._update_metrics(strategy, 'after_eval_iteration')
