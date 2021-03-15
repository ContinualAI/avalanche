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
from pathlib import Path
from typing import Union, Sequence, List, Optional, TYPE_CHECKING

from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_utils import get_metric_name, \
    phase_and_task, stream_type
from avalanche.evaluation.metric_results import MetricResult, MetricValue

if TYPE_CHECKING:
    from avalanche.training import PluggableStrategy

PathAlike = Union[Union[str, Path]]


class DiskUsage(Metric[float]):
    """
    The disk usage metric.

    This metric can be used to monitor the size of a set of directories.
    e.g. This can be useful to monitor the size of a replay buffer,
    """
    def __init__(self,
                 paths_to_monitor: Union[PathAlike, Sequence[PathAlike]] = None
                 ):
        """
        Creates an instance of the disk usage metric.

        The `result` method will return the sum of the size
        of the directories specified as the first parameter in KiloBytes.

        :param paths_to_monitor: a path or a list of paths to monitor. If None,
            the current working directory is used. Defaults to None.
        """

        if paths_to_monitor is None:
            paths_to_monitor = [os.getcwd()]
        if isinstance(paths_to_monitor, (str, Path)):
            paths_to_monitor = [paths_to_monitor]

        self._paths_to_monitor: List[str] = [str(p) for p in paths_to_monitor]

        self.total_usage = 0

    def update(self):
        """
        Updates the disk usage statistics.

        :return None.
        """

        dirs_size = 0
        for directory in self._paths_to_monitor:
            dirs_size += DiskUsage.get_dir_size(directory)

        self.total_usage = dirs_size

    def result(self) -> Optional[float]:
        """
        Retrieves the disk usage as computed during the last call to the
        `update` method.

        Calling this method will not change the internal state of the metric.

        :return: The disk usage or None if `update` was not invoked yet.
        """

        return self.total_usage

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self.total_usage = 0

    @staticmethod
    def get_dir_size(path: str):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    # in KB
                    s = os.path.getsize(fp) / 1024
                    total_size += s

        return total_size


class MinibatchDiskUsage(PluginMetric[float]):
    """
    The minibatch Disk usage metric.
    This metric only works at training time.

    At the end of each iteration, this metric logs the total
    size (in KB) of all the monitored paths.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochDiskUsage`.
    """

    def __init__(self, paths_to_monitor):
        """
        Creates an instance of the minibatch Disk usage metric.
        """
        super().__init__()

        self._minibatch_disk = DiskUsage(paths_to_monitor)

    def result(self) -> float:
        return self._minibatch_disk.result()

    def reset(self) -> None:
        self._minibatch_disk.reset()

    def before_training_iteration(self, strategy) -> MetricResult:
        self.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        self._minibatch_disk.update()
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        metric_value = self.result()

        metric_name = get_metric_name(self, strategy)

        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "DiskUsage_MB"


class EpochDiskUsage(PluginMetric[float]):
    """
    The Epoch Disk usage metric.
    This metric only works at training time.

    At the end of each epoch, this metric logs the total
    size (in KB) of all the monitored paths.
    """

    def __init__(self, paths_to_monitor):
        """
        Creates an instance of the epoch Disk usage metric.
        """
        super().__init__()

        self._epoch_disk = DiskUsage(paths_to_monitor)

    def before_training_epoch(self, strategy) -> MetricResult:
        self.reset()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        self._epoch_disk.update()
        return self._package_result(strategy)

    def reset(self) -> None:
        self._epoch_disk.reset()

    def result(self) -> float:
        return self._epoch_disk.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        disk_usage = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, disk_usage, plot_x_position)]

    def __str__(self):
        return "DiskUsage_Epoch"


class ExperienceDiskUsage(PluginMetric[float]):
    """
    The average experience Disk usage metric.
    This metric works only at eval time.

    At the end of each experience, this metric logs the total
    size (in KB) of all the monitored paths.
    """

    def __init__(self, paths_to_monitor):
        """
        Creates an instance of the experience Disk usage metric.
        """
        super().__init__()

        self._exp_disk = DiskUsage(paths_to_monitor)

    def before_eval_exp(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def after_eval_exp(self, strategy: 'PluggableStrategy') -> MetricResult:
        self._exp_disk.update()
        return self._package_result(strategy)

    def reset(self) -> None:
        self._exp_disk.reset()

    def result(self) -> float:
        return self._exp_disk.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        exp_disk = self.result()

        metric_name = get_metric_name(self, strategy, add_experience=True)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, exp_disk, plot_x_position)]

    def __str__(self):
        return "DiskUsage_Exp"


class StreamDiskUsage(PluginMetric[float]):
    """
    The average stream Disk usage metric.
    This metric works only at eval time.

    At the end of the eval stream, this metric logs the total
    size (in KB) of all the monitored paths.
    """

    def __init__(self, paths_to_monitor):
        """
        Creates an instance of the stream Disk usage metric.
        """
        super().__init__()

        self._exp_disk = DiskUsage(paths_to_monitor)

    def before_eval(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def after_eval(self, strategy: 'PluggableStrategy') -> MetricResult:
        self._exp_disk.update()
        return self._package_result(strategy)

    def reset(self) -> None:
        self._exp_disk.reset()

    def result(self) -> float:
        return self._exp_disk.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        exp_disk = self.result()

        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = '{}/{}_phase/{}_stream' \
            .format(str(self),
                    phase_name,
                    stream)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, exp_disk, plot_x_position)]

    def __str__(self):
        return "DiskUsage_Stream"


def disk_usage_metrics(*, paths_to_monitor=None, minibatch=False, epoch=False,
                       experience=False, stream=False) \
        -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of metric.

    :param minibatch: If True, will return a metric able to log the minibatch
        Disk usage
    :param epoch: If True, will return a metric able to log the epoch
        Disk usage
    :param experience: If True, will return a metric able to log the experience
        Disk usage.
    :param stream: If True, will return a metric able to log the evaluation
        stream Disk usage.

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchDiskUsage(paths_to_monitor=paths_to_monitor))

    if epoch:
        metrics.append(EpochDiskUsage(paths_to_monitor=paths_to_monitor))

    if experience:
        metrics.append(ExperienceDiskUsage(paths_to_monitor=paths_to_monitor))

    if stream:
        metrics.append(StreamDiskUsage(paths_to_monitor=paths_to_monitor))

    return metrics


__all__ = [
    'DiskUsage',
    'MinibatchDiskUsage',
    'EpochDiskUsage',
    'ExperienceDiskUsage',
    'StreamDiskUsage',
    'disk_usage_metrics'
]
