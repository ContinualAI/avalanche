from abc import ABC

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from avalanche.evaluation.metric_results import MetricValue
    from avalanche.training.templates.supervised import SupervisedTemplate


class BaseLogger(ABC):
    """Base class for loggers.

    Strategy loggers receive MetricValues from the Evaluation plugin and
    decide when and how to log them. MetricValues are processed
    by default using `log_metric` and `log_single_metric`.

    Additionally, loggers may implement any callback's handlers supported by
    the plugin's system of the template in use, which will be called
    automatically during the template's execution.
    This allows to control when the logging happen and how. For example,
    interactive loggers typically prints at the end of an
    epoch/experience/stream.

    Each child class should implement the `log_single_metric` method, which
    logs a single MetricValue.
    """

    def __init__(self):
        super().__init__()

    def log_single_metric(self, name, value, x_plot):
        """Log a metric value.

        This method is called whenever new metrics are available.
        By default, all the values are ignored.

        :param name: str, metric name
        :param value: the metric value, will be ignored if
            not supported by the logger
        :param x_plot: an integer representing the x value
            associated to the metric value
        """
        pass

    def log_metrics(self, metric_values: List["MetricValue"]) -> None:
        """Receive a list of MetricValues to log.

        This method is called whenever new metrics are available.

        :param metric_values: list of MetricValues to log.
        :param callback: The name of the callback (event) during which the
            metric value was collected.
        :return: None
        """
        for mval in metric_values:
            name = mval.name
            value = mval.value
            x_plot = mval.x_plot

            if isinstance(value, dict):
                for k, v in value.items():
                    n = f"{name}/{k}"
                    self.log_single_metric(n, v, x_plot)
            else:
                self.log_single_metric(name, value, x_plot)


__all__ = ["BaseLogger"]
