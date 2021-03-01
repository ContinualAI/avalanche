################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-01-2021                                                             #
# Author(s): Vincenzo Lomonaco, Lorenzo Pellegrini                             #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

import os
import time
from typing import Optional, List, TYPE_CHECKING
from threading import Thread
from psutil import Process

from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import get_metric_name, \
    phase_and_task, stream_type

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class MaxRAM(Metric[float]):
    """
    The RAM usage metric.
    Important: this metric approximates the real maximum RAM usage since
    it sample at discrete amount of time the RAM values.

    Instances of this metric keeps the maximum RAM usage detected.
    The update method starts the usage tracking. The reset method stops
    the tracking.


    The result, obtained using the `result` method, is the usage in mega-bytes.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an usage value of 0.
    """

    def __init__(self, delay=1):
        """
        Creates an instance of the RAM usage metric.
        :param delay: seconds after which update the maximum RAM
            usage
        """

        self._process_handle: Optional[Process] = None
        """
        The process handle, lazily initialized.
        """

        self.delay = delay

        self.thread = None
        """
        Thread executing RAM monitoring code
        """

        self.stop_f = False
        """
        Flag to stop the thread
        """

        self.max_usage = 0
        """
        Main metric result. Max RAM usage.
        """

    def _f(self):
        """
        Until a stop signal is encountered,
        this function monitors each `delay` seconds
        the maximum amount of RAM used by the process
        """
        start_time = time.monotonic()
        while not self.stop_f:
            # ram usage in MB
            ram_usage = self._process_handle.memory_info().rss / 1024 / 1024
            if ram_usage > self.max_usage:
                self.max_usage = ram_usage
            time.sleep(self.delay - ((time.monotonic() - start_time)
                                     % self.delay))

    def result(self) -> Optional[float]:
        """
        Retrieves the RAM usage.

        Calling this method will not change the internal state of the metric.

        :return: The average RAM usage in bytes, as a float value.
        """
        return self.max_usage

    def update(self):
        self._process_handle = Process(os.getpid())
        self.thread = Thread(target=self._f, daemon=True)
        self.thread.start()

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        if self.thread:
            self.stop_f = True
            self.thread.join()
            self.thread = None
        self.stop_f = False
        self.max_usage = 0
        self._process_handle = None


class MinibatchMaxRAM(PluginMetric[float]):
    """
    The Minibatch Max RAM metric.
    This metric only works at training time.
    """

    def __init__(self, delay=1):
        """
        Creates an instance of the Minibatch Max RAM metric
        :param delay: seconds after which update the maximum RAM
            usage
        """
        super().__init__()

        self._ram = MaxRAM(delay)

    def before_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        self.reset()
        self._ram.update()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        return self._package_result(strategy)

    def after_training(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def reset(self) -> None:
        self._ram.reset()

    def result(self) -> float:
        return self._ram.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        ram_usage = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, ram_usage, plot_x_position)]

    def __str__(self):
        return "MaxRAMUsage_MB"


class EpochMaxRAM(PluginMetric[float]):
    """
    The Epoch Max RAM metric.
    This metric only works at training time.
    """

    def __init__(self, delay=1):
        """
        Creates an instance of the epoch Max RAM metric.
        :param delay: seconds after which update the maximum RAM
            usage
        """
        super().__init__()

        self._ram = MaxRAM(delay)

    def before_training_epoch(self, strategy) -> MetricResult:
        self.reset()
        self._ram.update()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        return self._package_result(strategy)

    def after_training(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def reset(self) -> None:
        self._ram.reset()

    def result(self) -> float:
        return self._ram.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        ram_usage = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, ram_usage, plot_x_position)]

    def __str__(self):
        return "MaxRAMUsage_Epoch"


class ExperienceMaxRAM(PluginMetric[float]):
    """
    The Experience Max RAM metric.
    This metric only works at eval time.
    """

    def __init__(self, delay=1):
        """
        Creates an instance of the Experience CPU usage metric.
        :param delay: seconds after which update the maximum RAM
            usage
        """
        super().__init__()

        self._ram = MaxRAM(delay)

    def before_eval_exp(self, strategy) -> MetricResult:
        self.reset()
        self._ram.update()

    def after_eval_exp(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        return self._package_result(strategy)

    def after_eval(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def reset(self) -> None:
        self._ram.reset()

    def result(self) -> float:
        return self._ram.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        ram_usage = self.result()

        metric_name = get_metric_name(self, strategy, add_experience=True)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, ram_usage, plot_x_position)]

    def __str__(self):
        return "MaxRAMUsage_Experience"


class StreamMaxRAM(PluginMetric[float]):
    """
    The Stream Max RAM metric.
    This metric only works at eval time.
    """

    def __init__(self, delay=1):
        """
        Creates an instance of the Experience CPU usage metric.
        :param delay: seconds after which update the maximum RAM
            usage
        """
        super().__init__()

        self._ram = MaxRAM(delay)

    def before_eval(self, strategy) -> MetricResult:
        self.reset()
        self._ram.update()

    def after_eval(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        packed = self._package_result(strategy)
        self.reset()
        return packed

    def reset(self) -> None:
        self._ram.reset()

    def result(self) -> float:
        return self._ram.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        ram_usage = self.result()

        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = '{}/{}_phase/{}_stream' \
            .format(str(self),
                    phase_name,
                    stream)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, ram_usage, plot_x_position)]

    def __str__(self):
        return "MaxRAMUsage_Stream"


def max_ram_metrics(*, minibatch=False, epoch=False,
                    experience=False, stream=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of metric.

    :param minibatch: If True, will return a metric able to log the minibatch
        max RAM usage.
    :param epoch: If True, will return a metric able to log the epoch
        max RAM usage.
    :param experience: If True, will return a metric able to log the experience
        max RAM usage.
    :param stream: If True, will return a metric able to log the evaluation
        max stream RAM usage.

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchMaxRAM())

    if epoch:
        metrics.append(EpochMaxRAM())

    if experience:
        metrics.append(ExperienceMaxRAM())

    if stream:
        metrics.append(StreamMaxRAM())

    return metrics


__all__ = [
    'MaxRAM',
    'MinibatchMaxRAM',
    'EpochMaxRAM',
    'ExperienceMaxRAM',
    'StreamMaxRAM',
    'max_ram_metrics'
]
