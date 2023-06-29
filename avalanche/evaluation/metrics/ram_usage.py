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

from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metric_results import MetricResult

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class MaxRAM(Metric[float]):
    """The standalone RAM usage metric.

    Important: this metric approximates the real maximum RAM usage since
    it sample at discrete amount of time the RAM values.

    Instances of this metric keeps the maximum RAM usage detected.
    The `start_thread` method starts the usage tracking.
    The `stop_thread` method stops the tracking.

    The result, obtained using the `result` method, is the usage in mega-bytes.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an usage value of 0.
    """

    def __init__(self, every=1):
        """Creates an instance of the RAM usage metric.

        :param every: seconds after which update the maximum RAM usage.
        """

        self._process_handle: Process = Process(os.getpid())
        """The process handle."""

        self.every = every

        self.stop_f = False
        """Flag to stop the thread."""

        self.max_usage = 0
        """
        Main metric result. Max RAM usage.
        """

        self.thread = None
        """
        Thread executing RAM monitoring code
        """

    def _f(self):
        """
        Until a stop signal is encountered,
        this function monitors each `every` seconds
        the maximum amount of RAM used by the process
        """
        start_time = time.monotonic()
        while not self.stop_f:
            # ram usage in MB
            ram_usage = self._process_handle.memory_info().rss / 1024 / 1024
            if ram_usage > self.max_usage:
                self.max_usage = ram_usage
            time.sleep(self.every - ((time.monotonic() - start_time) % self.every))

    def result(self) -> Optional[float]:
        """
        Retrieves the RAM usage.

        Calling this method will not change the internal state of the metric.

        :return: The average RAM usage in bytes, as a float value.
        """
        return self.max_usage

    def start_thread(self):
        assert not self.thread, (
            "Trying to start thread " "without joining the previous."
        )
        self.thread = Thread(target=self._f, daemon=True)
        self.thread.start()

    def stop_thread(self):
        if self.thread:
            self.stop_f = True
            self.thread.join()
            self.stop_f = False
            self.thread = None

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self.max_usage = 0

    def update(self):
        pass


class RAMPluginMetric(GenericPluginMetric[float, MaxRAM]):
    def __init__(self, every, reset_at, emit_at, mode):
        super(RAMPluginMetric, self).__init__(MaxRAM(every), reset_at, emit_at, mode)

    def update(self, strategy):
        self._metric.update()


class MinibatchMaxRAM(RAMPluginMetric):
    """The Minibatch Max RAM metric.

    This plugin metric only works at training time.
    """

    def __init__(self, every=1):
        """Creates an instance of the Minibatch Max RAM metric.

        :param every: seconds after which update the maximum RAM
            usage
        """
        super(MinibatchMaxRAM, self).__init__(
            every, reset_at="iteration", emit_at="iteration", mode="train"
        )

    def before_training(self, strategy: "SupervisedTemplate") -> None:
        super().before_training(strategy)
        self._metric.start_thread()

    def after_training(self, strategy: "SupervisedTemplate") -> None:
        super().after_training(strategy)
        self._metric.stop_thread()

    def __str__(self):
        return "MaxRAMUsage_MB"


class EpochMaxRAM(RAMPluginMetric):
    """The Epoch Max RAM metric.

    This plugin metric only works at training time.
    """

    def __init__(self, every=1):
        """Creates an instance of the epoch Max RAM metric.

        :param every: seconds after which update the maximum RAM usage.
        """
        super(EpochMaxRAM, self).__init__(
            every, reset_at="epoch", emit_at="epoch", mode="train"
        )

    def before_training(self, strategy: "SupervisedTemplate") -> None:
        super().before_training(strategy)
        self._metric.start_thread()

    def after_training(self, strategy: "SupervisedTemplate") -> None:
        super().before_training(strategy)
        self._metric.stop_thread()

    def __str__(self):
        return "MaxRAMUsage_Epoch"


class ExperienceMaxRAM(RAMPluginMetric):
    """The Experience Max RAM metric.

    This plugin metric only works at eval time.
    """

    def __init__(self, every=1):
        """Creates an instance of the Experience CPU usage metric.

        :param every: seconds after which update the maximum RAM usage.
        """
        super(ExperienceMaxRAM, self).__init__(
            every, reset_at="experience", emit_at="experience", mode="eval"
        )

    def before_eval(self, strategy: "SupervisedTemplate") -> None:
        super().before_eval(strategy)
        self._metric.start_thread()

    def after_eval(self, strategy: "SupervisedTemplate") -> None:
        super().after_eval(strategy)
        self._metric.stop_thread()

    def __str__(self):
        return "MaxRAMUsage_Experience"


class StreamMaxRAM(RAMPluginMetric):
    """The Stream Max RAM metric.

    This plugin metric only works at eval time.
    """

    def __init__(self, every=1):
        """Creates an instance of the Experience CPU usage metric.

        :param every: seconds after which update the maximum RAM usage.
        """
        super(StreamMaxRAM, self).__init__(
            every, reset_at="stream", emit_at="stream", mode="eval"
        )

    def before_eval(self, strategy):
        super().before_eval(strategy)
        self._metric.start_thread()

    def after_eval(self, strategy: "SupervisedTemplate") -> MetricResult:
        packed = super().after_eval(strategy)
        self._metric.stop_thread()
        return packed

    def __str__(self):
        return "MaxRAMUsage_Stream"


def ram_usage_metrics(
    *, every=1, minibatch=False, epoch=False, experience=False, stream=False
) -> List[RAMPluginMetric]:
    """Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param every: seconds after which update the maximum RAM
        usage
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

    metrics: List[RAMPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchMaxRAM(every=every))

    if epoch:
        metrics.append(EpochMaxRAM(every=every))

    if experience:
        metrics.append(ExperienceMaxRAM(every=every))

    if stream:
        metrics.append(StreamMaxRAM(every=every))

    return metrics


__all__ = [
    "MaxRAM",
    "MinibatchMaxRAM",
    "EpochMaxRAM",
    "ExperienceMaxRAM",
    "StreamMaxRAM",
    "ram_usage_metrics",
]
