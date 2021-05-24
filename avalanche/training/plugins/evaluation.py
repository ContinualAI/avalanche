import warnings
from copy import copy
from collections import defaultdict
from typing import Union, Sequence, TYPE_CHECKING

from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.logging import StrategyLogger, InteractiveLogger

if TYPE_CHECKING:
    from avalanche.evaluation import PluginMetric
    from avalanche.training import BaseStrategy


class EvaluationPlugin(StrategyPlugin):
    """
    An evaluation plugin that obtains relevant data from the
    training and eval loops of the strategy through callbacks.
    The plugin keeps a dictionary with the last recorded value for each metric.
    The dictionary will be returned by the `train` and `eval` methods of the
    strategies.
    It is also possible to keep a dictionary with all recorded metrics by
    specifying `collect_all=True`. The dictionary can be retrieved via
    the `get_all_metrics` method.

    This plugin also logs metrics using the provided loggers.
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
        :param collect_all: if True, collect in a separate dictionary all
            metric curves values. This dictionary is accessible with
            `get_all_metrics` method.
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

        if self.collect_all:
            # for each curve collect all emitted values.
            # dictionary key is full metric name.
            # Dictionary value is a tuple of two lists.
            # first list gathers x values (indices representing
            # time steps at which the corresponding metric value
            # has been emitted)
            # second list gathers metric values
            self.all_metric_results = defaultdict(lambda: ([], []))
        # Dictionary of last values emitted. Dictionary key
        # is the full metric name, while dictionary value is
        # metric value.
        self.last_metric_results = {}

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
            if self.collect_all:
                self.all_metric_results[name][0].append(x)
                self.all_metric_results[name][1].append(val)

            self.last_metric_results[name] = val

        for logger in self.loggers:
            getattr(logger, callback)(strategy, metric_values)
        return metric_values

    def get_last_metrics(self):
        """
        Return a shallow copy of dictionary with metric names
        as keys and last metrics value as values.

        :return: a dictionary with full metric
            names as keys and last metric value as value.
        """
        return copy(self.last_metric_results)

    def get_all_metrics(self):
        """
        Return the dictionary of all collected metrics.
        This method should be called only when `collect_all` is set to True.

        :return: if `collect_all` is True, returns a dictionary
            with full metric names as keys and a tuple of two lists
            as value. The first list gathers x values (indices
            representing time steps at which the corresponding
            metric value has been emitted). The second list
            gathers metric values. a dictionary. If `collect_all`
            is False return an empty dictionary
        """
        if self.collect_all:
            return self.all_metric_results
        else:
            return {}

    def reset_last_metrics(self):
        """
        Set the dictionary storing last value for each metric to be
        empty dict.
        """
        self.last_metric_results = {}

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_training')

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_training_exp')

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):
        self._update_metrics(strategy, 'before_train_dataset_adaptation')

    def after_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        self._update_metrics(strategy, 'after_train_dataset_adaptation')

    def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_training_epoch')

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_training_iteration')

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_forward')

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_forward')

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_backward')

    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_backward')

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_training_iteration')

    def before_update(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_update')

    def after_update(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_update')

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_training_epoch')

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_training_exp')

    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_training')

    def before_eval(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_eval')

    def before_eval_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        self._update_metrics(strategy, 'before_eval_dataset_adaptation')

    def after_eval_dataset_adaptation(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_eval_dataset_adaptation')

    def before_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_eval_exp')

    def after_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_eval_exp')

    def after_eval(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_eval')

    def before_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_eval_iteration')

    def before_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_eval_forward')

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_eval_forward')

    def after_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_eval_iteration')


default_logger = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loggers=[InteractiveLogger()])


__all__ = [
    'EvaluationPlugin',
    'default_logger'
]