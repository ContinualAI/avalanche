################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-01-2021                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

import os
import time
import warnings
from typing import Optional, Callable, TYPE_CHECKING, List

from psutil import Process

from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics import Mean
if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class CpuUsage(Metric[float]):
    """
    The CPU usage metric.

    Instances of this metric compute the average CPU usage as a float value.
    The metric starts tracking the CPU usage when the `update` method is called
    for the first time. That is, the tracking doesn't start at the time the
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

        self._timer: Callable[[], float] = getattr(time, 'monotonic', time.time)
        """
        The timer implementation (aligned with the one used by psutil).
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
                warnings.warn('CpuUsage can\'t detect the elapsed time. It is '
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


class MinibatchCpuUsage(PluginMetric[float]):
    """
    The minibatch CPU usage metric.

    This metric "logs" the CPU usage for each iteration. Beware that this
    metric will not average the usage across minibatches!

    If a more coarse-grained logging is needed, consider using
    :class:`EpochCpuUsage`, :class:`AverageEpochCpuUsage` or
    :class:`StepCpuUsage` instead.
    """

    def __init__(self, *, train=True, eval=False):
        """
        Creates an instance of the minibatch CPU usage metric.

        The train and eval parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param eval: When True, the metric will be computed on the eval
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not eval:
            raise ValueError('train and eval can\'t be both False at the same'
                             ' time.')
        self._minibatch_cpu = CpuUsage()
        self._track_train_cpu = train
        self._track_eval_cpu = eval

    def result(self) -> float:
        return self._minibatch_cpu.result()

    def reset(self) -> None:
        self._minibatch_cpu.reset()

    def before_training_iteration(self, strategy) -> MetricResult:
        if not self._track_train_cpu:
            return
        self.reset()
        self._minibatch_cpu.update()

    def before_eval_iteration(self, strategy) -> MetricResult:
        if not self._track_eval_cpu:
            return
        self.reset()
        self._minibatch_cpu.update()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._track_train_cpu:
            self._minibatch_cpu.update()
            return self._package_result(strategy)

    def after_eval_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._track_eval_cpu:
            self._minibatch_cpu.update()
            return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'CPU_MB/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class EpochCpuUsage(PluginMetric[float]):
    """
    The epoch average CPU usage metric.

    The average usage will be logged after each epoch. Beware that this
    metric will not average the CPU usage across epochs!

    If logging the average usage across epochs is needed, consider using
    :class:`AverageEpochCpuUsage` instead.
    """

    def __init__(self, *, train=True, eval=False):
        """
        Creates an instance of the epoch CPU usage metric.

        The train and eval parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param eval: When True, the metric will be computed on the eval
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not eval:
            raise ValueError('train and eval can\'t be both False at the same'
                             ' time.')

        self._epoch_cpu = CpuUsage()
        self._track_train_cpu = train
        self._track_eval_cpu = eval

    def before_training_epoch(self, strategy) -> MetricResult:
        if not self._track_train_cpu:
            return
        self.reset()
        self._epoch_cpu.update()

    def before_eval_step(self, strategy) -> MetricResult:
        if not self._track_eval_cpu:
            return
        self.reset()
        self._epoch_cpu.update()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._track_train_cpu:
            self._epoch_cpu.update()
            return self._package_result(strategy)

    def after_eval_step(self, strategy: 'PluggableStrategy') -> MetricResult:
        if self._track_eval_cpu:
            self._epoch_cpu.update()
            return self._package_result(strategy)

    def reset(self) -> None:
        self._epoch_cpu.reset()

    def result(self) -> float:
        return self._epoch_cpu.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        cpu_usage = self.result()

        metric_name = 'Epoch_CPU/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, cpu_usage, plot_x_position)]


class AverageEpochCpuUsage(PluginMetric[float]):
    """
    The average epoch CPU usage metric.

    The average usage will be logged at the end of the step.

    Beware that this metric will average the usage across epochs! If logging the
    epoch-specific usage is needed, consider using :class:`EpochCpuUsage`
    instead.
    """

    def __init__(self, *, train=True, eval=False):
        """
        Creates an instance of the average epoch cpu usage metric.

        The train and eval parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param eval: When True, the metric will be computed on the eval
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not eval:
            raise ValueError('train and eval can\'t be both False at the same'
                             ' time.')

        self._cpu_mean = Mean()
        self._epoch_cpu = CpuUsage()
        self._track_train_cpu = train
        self._track_eval_cpu = eval

    def before_training_epoch(self, strategy) -> MetricResult:
        if not self._track_train_cpu:
            return
        self._epoch_cpu.reset()
        self._epoch_cpu.update()

    def before_eval_step(self, strategy) -> MetricResult:
        if not self._track_eval_cpu:
            return
        self.reset()
        self._epoch_cpu.reset()
        self._epoch_cpu.update()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if not self._track_train_cpu:
            return
        self._epoch_cpu.update()
        self._cpu_mean.update(self._epoch_cpu.result())
        return self._package_result(strategy)

    def after_eval_step(self, strategy: 'PluggableStrategy') -> MetricResult:
        if not self._track_eval_cpu:
            return
        self._epoch_cpu.update()
        self._cpu_mean.update(self._epoch_cpu.result())
        return self._package_result(strategy)

    def reset(self) -> None:
        self._epoch_cpu.reset()
        self._cpu_mean.reset()

    def result(self) -> float:
        return self._cpu_mean.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        cpu_usage = self.result()

        metric_name = 'Avg_Epoch_CPU/{}/Task{:03}'.format(
            phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(
            self, metric_name, cpu_usage, plot_x_position)]


class StepCpuUsage(PluginMetric[float]):
    """
    The average step CPU usage metric.

    This metric may seem very similar to :class:`AverageEpochCpuUsage`. However,
    differently from that: 1) obviously, the usage is not averaged by dividing
    by the number of epochs; 2) most importantly, the usage of code running
    outside the epoch loop is accounted too (a thing that
    :class:`AverageEpochCpuUsage` doesn't support). For instance, this metric is
    more suitable when measuring the CPU usage of algorithms involving
    after-training consolidation, replay pattern selection and other CPU bound
    mechanisms.
    """

    def __init__(self, *, train=True, eval=False):
        """
        Creates an instance of the step CPU usage metric.

        The train and eval parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param eval: When True, the metric will be computed on the eval
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not eval:
            raise ValueError('train and eval can\'t be both False at the same'
                             ' time.')

        self._step_cpu = CpuUsage()
        self._track_train_cpu = train
        self._track_eval_cpu = eval

    def before_training_step(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if not self._track_train_cpu:
            return
        self.reset()
        self._step_cpu.update()

    def before_eval_step(self, strategy: 'PluggableStrategy') -> MetricResult:
        if not self._track_eval_cpu:
            return
        self.reset()
        self._step_cpu.update()

    def after_training_step(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._track_train_cpu:
            self._step_cpu.update()
            return self._package_result(strategy)

    def after_eval_step(self, strategy: 'PluggableStrategy') -> MetricResult:
        if self._track_eval_cpu:
            self._step_cpu.update()
            return self._package_result(strategy)

    def reset(self) -> None:
        self._step_cpu.reset()

    def result(self) -> float:
        return self._step_cpu.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        step_cpu = self.result()

        metric_name = 'Step_CPU/{}/Task{:03}'.format(
            phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, step_cpu, plot_x_position)]


def cpu_usage_metrics(*, minibatch=False, epoch=False, epoch_average=False,
                      step=False, train=None, eval=None) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of metric.

    :param minibatch: If True, will return a metric able to log the minibatch
        elapsed time.
    :param epoch: If True, will return a metric able to log the epoch elapsed
        time.
    :param epoch_average: If True, will return a metric able to log the average
        epoch elapsed time.
    :param step: If True, will return a metric able to log the step elapsed
        time.
    :param train: If True, metrics will log values for the train flow. Defaults
        to None, which means that the per-metric default value will be used.
    :param eval: If True, metrics will log values for the eval flow. Defaults
        to None, which means that the per-metric default value will be used.

    :return: A list of plugin metrics.
    """

    if (train is not None and not train) and (eval is not None and not eval):
        raise ValueError('train and eval can\'t be both False at the same'
                         ' time.')

    train_eval_flags = dict()
    if train is not None:
        train_eval_flags['train'] = train

    if eval is not None:
        train_eval_flags['eval'] = eval

    metrics = []
    if minibatch:
        metrics.append(MinibatchCpuUsage(**train_eval_flags))

    if epoch:
        metrics.append(EpochCpuUsage(**train_eval_flags))

    if epoch_average:
        metrics.append(AverageEpochCpuUsage(**train_eval_flags))

    if step:
        metrics.append(StepCpuUsage(**train_eval_flags))

    return metrics


__all__ = [
    'CpuUsage',
    'MinibatchCpuUsage',
    'EpochCpuUsage',
    'AverageEpochCpuUsage',
    'StepCpuUsage',
    'cpu_usage_metrics'
]
