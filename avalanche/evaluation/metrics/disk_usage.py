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
import time
from pathlib import Path
from typing import Union, Sequence, List, Optional, Tuple, TYPE_CHECKING

import psutil

from avalanche.evaluation import Metric
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics._any_event_metric import AnyEventMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy

PathAlike = Union[Union[str, Path]]

DiskUsageResult = Union[int, Tuple[int, int, int, int, int]]


class DiskUsage(Metric[DiskUsageResult]):
    """
    The disk usage metric.

    This metric can be used to monitor the size of a set of directories. This
    can be useful to monitor the size of a replay buffer,

    This metric can also be used to get info regarding the overall amount of
    other system-wide disk stats (see the constructor for more details).
    """
    def __init__(self,
                 paths_to_monitor: Union[PathAlike, Sequence[PathAlike]] = None,
                 monitor_disk_io: bool = False):
        """
        Creates an instance of the disk usage metric.

        By default invoking the `result` method will return the sum of the size
        of the directories specified as the first parameter. By passing
        `monitor_disk_io` as true the `result` method will return a 5 elements
        tuple containing 1) the sum of the size of the directories,
        the system-wide 2) read count, 3) write count, 4) read bytes and
        5) written bytes.

        :param paths_to_monitor: a path or a list of paths to monitor. If None,
            the current working directory is used. Defaults to None.
        :param monitor_disk_io: If True enables monitoring of I/O operations on
            disk. WARNING: Reports are system-wide, grouping all disks. Defaults
            to False.
        """

        if paths_to_monitor is None:
            paths_to_monitor = [os.getcwd()]
        if isinstance(paths_to_monitor, (str, Path)):
            paths_to_monitor = [paths_to_monitor]

        self._paths_to_monitor: List[str] = [str(p) for p in paths_to_monitor]
        self._track_disk_io: bool = monitor_disk_io
        self._last_result: Optional[DiskUsageResult] = None

    def update(self):
        """
        Updates the disk usage statistics.

        :return None.
        """

        dirs_size = 0
        for directory in self._paths_to_monitor:
            dirs_size += DiskUsage.get_dir_size(directory)

        if self._track_disk_io:
            counters = psutil.disk_io_counters()
            read_count, write_count = counters.read_count, counters.write_count
            read_bytes, write_bytes = counters.read_bytes, counters.write_bytes
            self._last_result = (dirs_size, read_count, write_count, read_bytes,
                                 write_bytes)
        else:
            self._last_result = dirs_size

    def result(self) -> Optional[DiskUsageResult]:
        """
        Retrieves the disk usage as computed during the last call to the
        `update` method.

        Calling this method will not change the internal state of the metric.

        The info returned may vary depending on whether the constructor was
        invoked with `monitor_disk_io` to True. See the constructor for more
        details.

        :return: The disk usage or None if `update` was not invoked yet.
        """

        return self._last_result

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._last_result = None

    @staticmethod
    def get_dir_size(path: str):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

        return total_size


class DiskUsageMonitor(AnyEventMetric[float]):
    """
    The disk usage metric.

    This metric logs the disk usage (directory size) of the given list of paths.

    The logged value is in MiB.

    The metric can be either configured to log after a certain timeout or
    at each event.

    Disk usage is logged separately for the train and eval phases.
    """

    def __init__(self,
                 *paths: PathAlike,
                 timeout: float = 5.0,
                 train=True, eval=False):
        """
        Creates an instance of the disk usage metric.

        The train and eval parameters can be True at the same time. However,
        at least one of them must be True.

        :param paths: A list of paths to monitor. If no paths are defined,
            the current working directory is used.
        :param timeout: The timeout between each disk usage check, in seconds.
            If None, the disk usage is checked at every possible event (not
            recommended). Defaults to 5 seconds.
        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param eval: When True, the metric will be computed on the eval
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not eval:
            raise ValueError('train and eval can\'t be both False at the same'
                             ' time.')

        if len(paths) == 0:
            paths = None  # Use current working directory

        self._disk_sensor = DiskUsage(paths)
        self._timeout = timeout
        self._last_time = None
        self._track_train_usage = train
        self._track_eval_usage = eval

    def on_event(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        if (strategy.is_training and not self._track_train_usage) or \
                (strategy.is_eval and not self._track_eval_usage):
            return None

        is_elapsed = False
        if self._timeout is not None:
            current_time = time.time()
            is_elapsed = self._last_time is None or (
                    (current_time - self._last_time) >= self._timeout)
            if is_elapsed:
                self._last_time = current_time

        if self._timeout is None or is_elapsed:
            self._disk_sensor.update()
            return self._package_result(strategy)

    def result(self) -> Optional[float]:
        byte_value = self._disk_sensor.result()
        if byte_value is None:
            return None
        return byte_value / (1024 * 1204)  # MiB

    def reset(self) -> None:
        self._disk_sensor.reset()
        self._last_time = None

    def _package_result(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        phase_name, _ = phase_and_task(strategy)
        experience_cpu = self.result()

        metric_name = 'Disk_usage/{}'.format(phase_name)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, experience_cpu, plot_x_position)]


__all__ = [
    'DiskUsage',
    'DiskUsageMonitor'
]
