################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-01-2021                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

import os
import warnings
from typing import Optional, TYPE_CHECKING, List

from psutil import Process

from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import get_metric_name, \
    phase_and_task, stream_type
from avalanche.evaluation.metrics import Mean

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class CPUUsage(Metric[float]):
    """
    The CPU usage metric.

    Instances of this metric compute the average CPU usage as a float value.
    The metric starts tracking the CPU usage when the `update` method is called
    for the first time. That is, the tracking does not start at the time the
    constructor is invoked.

    Calling the `update` method more than twice will update the metric to the
    average usage between the first and the last call to `update`.

    The result, obtained using the `result` method, is the usage computed
    as stated above.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an usage value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the CPU usage metric.

        By default this metric in its initial state will return a CPU usage
        value of 0. The metric can be updated by using the `update` method
        while the average CPU usage can be retrieved using the `result` method.
        """

        self._mean_usage = Mean()
        """
        The mean utility that will be used to store the average usage.
        """

        self._process_handle: Optional[Process] = None
        """
        The process handle, lazily initialized.
        """

        self._first_update = True
        """
        An internal flag to keep track of the first call to the `update` method.
        """

    def update(self) -> None:
        """
        Update the running CPU usage.

        For more info on how to set the starting moment see the class
        description.

        :return: None.
        """
        if self._first_update:
            self._process_handle = Process(os.getpid())

        last_time = getattr(
            self._process_handle, '_last_sys_cpu_times', None)
        utilization = self._process_handle.cpu_percent()
        current_time = getattr(
            self._process_handle, '_last_sys_cpu_times', None)

        if self._first_update:
            self._first_update = False
        else:
            if current_time is None or last_time is None:
                warnings.warn('CPUUsage can\'t detect the elapsed time. It is '
                              'recommended to update avalanche to the latest '
                              'version.')
                # Fallback, shouldn't happen
                current_time = 1.0
                last_time = 0.0
            self._mean_usage.update(utilization, current_time - last_time)

    def result(self) -> float:
        """
        Retrieves the average CPU usage.

        Calling this method will not change the internal state of the metric.

        :return: The average CPU usage, as a float value.
        """
        return self._mean_usage.result()

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._mean_usage.reset()
        self._process_handle = None
        self._first_update = True


class MinibatchCPUUsage(PluginMetric[float]):
    """
    The minibatch CPU usage metric.
    This metric only works at training time.

    This metric "logs" the CPU usage for each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochCPUUsage`.
    """

    def __init__(self):
        """
        Creates an instance of the minibatch CPU usage metric.
        """
        super().__init__()

        self._minibatch_cpu = CPUUsage()

    def result(self) -> float:
        return self._minibatch_cpu.result()

    def reset(self) -> None:
        self._minibatch_cpu.reset()

    def before_training_iteration(self, strategy) -> MetricResult:
        self.reset()
        self._minibatch_cpu.update()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        self._minibatch_cpu.update()
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        metric_value = self.result()

        metric_name = get_metric_name(self, strategy)

        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "CPUUsage_MB"


class EpochCPUUsage(PluginMetric[float]):
    """
    The Epoch CPU usage metric.
    This metric only works at training time.

    The average usage will be logged after each epoch.
    """

    def __init__(self):
        """
        Creates an instance of the epoch CPU usage metric.
        """
        super().__init__()

        self._epoch_cpu = CPUUsage()

    def before_training_epoch(self, strategy) -> MetricResult:
        self.reset()
        self._epoch_cpu.update()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        self._epoch_cpu.update()
        return self._package_result(strategy)

    def reset(self) -> None:
        self._epoch_cpu.reset()

    def result(self) -> float:
        return self._epoch_cpu.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        cpu_usage = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, cpu_usage, plot_x_position)]

    def __str__(self):
        return "CPUUsage_Epoch"


class RunningEpochCPUUsage(PluginMetric[float]):
    """
    The running epoch CPU usage metric.
    This metric only works at training time

    After each iteration, the metric logs the average CPU usage up
    to the current epoch iteration.
    """

    def __init__(self):
        """
        Creates an instance of the average epoch cpu usage metric.
        """
        super().__init__()

        self._cpu_mean = Mean()
        self._epoch_cpu = CPUUsage()

    def before_training_epoch(self, strategy) -> MetricResult:
        self.reset()

    def before_training_iteration(self, strategy: 'PluggableStrategy') \
            -> 'MetricResult':
        self._epoch_cpu.update()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> None:
        self._epoch_cpu.update()
        self._cpu_mean.update(self._epoch_cpu.result())
        self._epoch_cpu.reset()
        return self._package_result(strategy)

    def reset(self) -> None:
        self._epoch_cpu.reset()
        self._cpu_mean.reset()

    def result(self) -> float:
        return self._cpu_mean.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        cpu_usage = self.result()

        metric_name = get_metric_name(self, strategy)

        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(
            self, metric_name, cpu_usage, plot_x_position)]

    def __str__(self):
        return "RunningCPUUsage_Epoch"


class ExperienceCPUUsage(PluginMetric[float]):
    """
    The average experience CPU usage metric.
    This metric works only at eval time.

    After each experience, this metric emits the average CPU usage on that
    experienc.
    """

    def __init__(self):
        """
        Creates an instance of the experience CPU usage metric.
        """
        super().__init__()

        self._exp_cpu = CPUUsage()

    def before_eval_exp(self, strategy: 'PluggableStrategy') -> None:
        self.reset()
        self._exp_cpu.update()

    def after_eval_exp(self, strategy: 'PluggableStrategy') -> MetricResult:
        self._exp_cpu.update()
        return self._package_result(strategy)

    def reset(self) -> None:
        self._exp_cpu.reset()

    def result(self) -> float:
        return self._exp_cpu.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        exp_cpu = self.result()

        metric_name = get_metric_name(self, strategy, add_experience=True)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, exp_cpu, plot_x_position)]

    def __str__(self):
        return "CPUUsage_Exp"


class StreamCPUUsage(PluginMetric[float]):
    """
    The average stream CPU usage metric.
    This metric works only at eval time.

    After the entire evaluation stream, this metric emits
    the average CPU usage on all experiences.
    """

    def __init__(self):
        """
        Creates an instance of the stream CPU usage metric.
        """
        super().__init__()

        self._exp_cpu = CPUUsage()

    def before_eval(self, strategy: 'PluggableStrategy') -> None:
        self.reset()
        self._exp_cpu.update()

    def after_eval(self, strategy: 'PluggableStrategy') -> MetricResult:
        self._exp_cpu.update()
        return self._package_result(strategy)

    def reset(self) -> None:
        self._exp_cpu.reset()

    def result(self) -> float:
        return self._exp_cpu.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        exp_cpu = self.result()

        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = '{}/{}_phase/{}_stream' \
            .format(str(self),
                    phase_name,
                    stream)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, exp_cpu, plot_x_position)]

    def __str__(self):
        return "CPUUsage_Stream"


def cpu_usage_metrics(*, minibatch=False, epoch=False, epoch_running=False,
                      experience=False, stream=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of metric.

    :param minibatch: If True, will return a metric able to log the minibatch
        CPU usage
    :param epoch: If True, will return a metric able to log the epoch
        CPU usage
    :param epoch_running: If True, will return a metric able to log the running
        epoch CPU usage.
    :param experience: If True, will return a metric able to log the experience
        CPU usage.
    :param stream: If True, will return a metric able to log the evaluation
        stream CPU usage.

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchCPUUsage())

    if epoch:
        metrics.append(EpochCPUUsage())

    if epoch_running:
        metrics.append(RunningEpochCPUUsage())

    if experience:
        metrics.append(ExperienceCPUUsage())

    if stream:
        metrics.append(StreamCPUUsage())

    return metrics


__all__ = [
    'CPUUsage',
    'MinibatchCPUUsage',
    'EpochCPUUsage',
    'RunningEpochCPUUsage',
    'ExperienceCPUUsage',
    'StreamCPUUsage',
    'cpu_usage_metrics'
]
