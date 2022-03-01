import warnings
from copy import copy
from collections import defaultdict
from typing import Union, Sequence, TYPE_CHECKING

from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger

if TYPE_CHECKING:
    from avalanche.evaluation import PluginMetric
    from avalanche.logging import BaseLogger
    from avalanche.training.templates.supervised import SupervisedTemplate


class EvaluationPlugin:
    """Manager for logging and metrics.

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

    def __init__(
        self,
        *metrics: Union["PluginMetric", Sequence["PluginMetric"]],
        loggers: Union["BaseLogger", Sequence["BaseLogger"]] = None,
        collect_all=True,
        benchmark=None,
        strict_checks=False,
        suppress_warnings=False
    ):
        """Creates an instance of the evaluation plugin.

        :param metrics: The metrics to compute.
        :param loggers: The loggers to be used to log the metric values.
        :param collect_all: if True, collect in a separate dictionary all
            metric curves values. This dictionary is accessible with
            `get_all_metrics` method.
        :param benchmark: continual learning benchmark needed to check stream
            completeness during evaluation or other kind of properties. If
            None, no check will be conducted and the plugin will emit a
            warning to signal this fact.
        :param strict_checks: if True, `benchmark` has to be provided.
            In this case, only full evaluation streams are admitted when
            calling `eval`. An error will be raised otherwise. When False,
            `benchmark` can be `None` and only warnings will be raised.
        :param suppress_warnings: if True, warnings and errors will never be
            raised from the plugin.
            If False, warnings and errors will be raised following
            `benchmark` and `strict_checks` behavior.
        """
        super().__init__()
        self.collect_all = collect_all
        self.benchmark = benchmark
        self.strict_checks = strict_checks
        self.suppress_warnings = suppress_warnings
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

        if benchmark is None:
            if not suppress_warnings:
                if strict_checks:
                    raise ValueError(
                        "Benchmark cannot be None " "in strict mode."
                    )
                else:
                    warnings.warn(
                        "No benchmark provided to the evaluation plugin. "
                        "Metrics may be computed on inconsistent portion "
                        "of streams, use at your own risk."
                    )
        else:
            self.complete_test_stream = benchmark.test_stream

        self.loggers: Sequence["BaseLogger"] = loggers

        if len(self.loggers) == 0:
            warnings.warn("No loggers specified, metrics will not be logged")

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

        self._active = True
        """If False, no metrics will be collected."""

        self._metric_values = []
        """List of metrics that have yet to be processed by loggers."""

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        assert (
            value is True or value is False
        ), "Active must be set as either True or False"
        self._active = value

    def publish_metric_value(self, mval: MetricValue):
        """Publish a MetricValue to be processed by the loggers."""
        self._metric_values.append(mval)

        name = mval.name
        x = mval.x_plot
        val = mval.value
        if self.collect_all:
            self.all_metric_results[name][0].append(x)
            self.all_metric_results[name][1].append(val)
        self.last_metric_results[name] = val

    def _update_metrics_and_loggers(
        self, strategy: "SupervisedTemplate", callback: str
    ):
        """Call the metric plugins with the correct callback `callback` and
        update the loggers with the new metric values."""
        if not self._active:
            return []

        for metric in self.metrics:
            if hasattr(metric, callback):
                metric_result = getattr(metric, callback)(strategy)
                if isinstance(metric_result, Sequence):
                    for mval in metric_result:
                        self.publish_metric_value(mval)
                elif metric_result is not None:
                    self.publish_metric_value(metric_result)

        for logger in self.loggers:
            logger.log_metrics(self._metric_values)
            if hasattr(logger, callback):
                getattr(logger, callback)(strategy, self._metric_values)
        self._metric_values = []

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

    def __getattribute__(self, item):
        # We don't want to reimplement all the callbacks just to call the
        # metrics. What we don't instead is to assume that any method that
        # starts with `before` or `after` is a callback of the plugin system,
        # and we forward that call to the metrics.
        try:
            return super().__getattribute__(item)
        except AttributeError as e:
            if item.startswith("before_") or item.startswith("after_"):
                # method is a callback. Forward to metrics.
                return lambda strat, **kwargs: self._update_metrics_and_loggers(
                    strat, item
                )
            raise

    def before_eval(self, strategy: "SupervisedTemplate", **kwargs):
        self._update_metrics_and_loggers(strategy, "before_eval")
        msgw = (
            "Evaluation stream is not equal to the complete test stream. "
            "This may result in inconsistent metrics. Use at your own risk."
        )
        msge = (
            "Stream provided to `eval` must be the same of the entire "
            "evaluation stream."
        )
        if self.benchmark is not None:
            for i, exp in enumerate(self.complete_test_stream):
                try:
                    current_exp = strategy.current_eval_stream[i]
                    if exp.current_experience != current_exp.current_experience:
                        if not self.suppress_warnings:
                            if self.strict_checks:
                                raise ValueError(msge)
                            else:
                                warnings.warn(msgw)
                except IndexError:
                    if self.strict_checks:
                        raise ValueError(msge)
                    else:
                        warnings.warn(msgw)


default_evaluator = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loggers=[InteractiveLogger()],
    suppress_warnings=True,
)


__all__ = ["EvaluationPlugin", "default_evaluator"]
