#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
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
from typing import Optional, TYPE_CHECKING

from psutil import Process

from avalanche.evaluation import Metric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics._any_event_metric import AnyEventMetric

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class RamUsage(Metric[float]):
    """
    The RAM usage metric.

    Instances of this metric compute the punctual RAM usage as a float value.
    The metric updates the value each time the `update` method is called.

    The result, obtained using the `result` method, is the usage in bytes.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an usage value of `None`.
    """

    def __init__(self, two_read_average=False):
        """
        Creates an instance of the RAM usage metric.

        By default this metric in its initial state will return a RAM usage
        value of `None`. The metric can be updated by using the `update` method
        while the average usage value can be retrieved using the `result`
        method.

        :param two_read_average: If True, the value resulting from calling
            `update` more than once will set the result to the average between
            the last read and the current RAM usage value.
        """

        self._process_handle: Optional[Process] = None
        """
        The process handle, lazily initialized.
        """

        self._last_values = None
        """
        The last detected RAM usage.
        """

        self._first_update = True
        """
        An internal flag to keep track of the first call to the `update` method.
        """

        self._two_read_average = two_read_average
        """
        If True, the value resulting from calling `update` more than once will 
        set the result to the average between the last read and the current RAM
        usage value.
        """

    def update(self) -> None:
        """
        Update the RAM usage.

        :return: None.
        """
        if self._first_update:
            self._process_handle = Process(os.getpid())
        memory_usage = self._process_handle.memory_info().rss

        if self._first_update:
            self._last_values = [memory_usage]
            self._first_update = False
        else:
            if self._two_read_average:
                if len(self._last_values) > 1:
                    self._last_values.pop(0)
                self._last_values.append(memory_usage)
            else:
                self._last_values = [memory_usage]

    def result(self) -> Optional[float]:
        """
        Retrieves the RAM usage.

        Calling this method will not change the internal state of the metric.

        :return: The average RAM usage in bytes, as a float value.
        """
        if self._first_update:
            return None
        return sum(self._last_values) / len(self._last_values)

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._process_handle = None
        self._first_update = True
        self._last_values = None


class RamUsageMonitor(AnyEventMetric[float]):
    """
    The RAM usage metric.

    This metric logs the RAM usage.

    The logged value is in MiB.

    The metric can be either configured to log after a certain timeout or
    at each event.

    RAM usage is logged separately for the train and test phases.
    """

    def __init__(self, *, timeout: float = 5.0, train=True, test=False):
        """
        Creates an instance of the RAM usage metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param timeout: The timeout between each RAM usage check, in seconds.
            If None, the RAM usage is checked at every possible event (not
            recommended). Defaults to 5 seconds.
        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             ' time.')

        self._ram_sensor = RamUsage()
        self._timeout = timeout
        self._last_time = None
        self._track_train_usage = train
        self._track_test_usage = test

    def on_event(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        if (strategy.is_training and not self._track_train_usage) or \
                (strategy.is_eval and not self._track_test_usage):
            return None

        is_elapsed = False
        if self._timeout is not None:
            current_time = time.time()
            is_elapsed = self._last_time is None or (
                    (current_time - self._last_time) >= self._timeout)
            if is_elapsed:
                self._last_time = current_time

        if self._timeout is None or is_elapsed:
            self._ram_sensor.update()
            return self._package_result(strategy)

    def result(self) -> Optional[float]:
        byte_value = self._ram_sensor.result()
        if byte_value is None:
            return None
        return byte_value / (1024 * 1204)  # MiB

    def reset(self) -> None:
        self._ram_sensor.reset()
        self._last_time = None

    def _package_result(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        phase_name, _ = phase_and_task(strategy)
        step_ram = self.result()

        metric_name = 'RAM_usage/{}'.format(phase_name)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, step_ram, plot_x_position)]


__all__ = [
    'RamUsage',
    'RamUsageMonitor'
]
