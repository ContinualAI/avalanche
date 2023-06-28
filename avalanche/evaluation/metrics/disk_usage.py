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
from typing import Union, Sequence, List, Optional

from avalanche.evaluation import Metric, GenericPluginMetric

PathAlike = Union[str, Path]


class DiskUsage(Metric[float]):
    """
    The standalone disk usage metric.

    This metric can be used to monitor the size of a set of directories.
    e.g. This can be useful to monitor the size of a replay buffer,
    """

    def __init__(
        self, paths_to_monitor: Optional[Union[PathAlike, Sequence[PathAlike]]] = None
    ):
        """
        Creates an instance of the standalone disk usage metric.

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

        self.total_usage: float = 0.0

    def update(self):
        """
        Updates the disk usage statistics.

        :return None.
        """

        dirs_size = 0.0
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
    def get_dir_size(path: str) -> float:
        """
        Obtains the size of the given directory, in KiB.

        :param path: The path of an existing directory.
        :return: A float value describing the size (in KiB)
            of the directory as the sum of all its elements.
        """
        total_size = 0.0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    # in KB
                    s = os.path.getsize(fp) / 1024
                    total_size += s

        return total_size


class DiskPluginMetric(GenericPluginMetric[float, DiskUsage]):
    def __init__(self, paths, reset_at, emit_at, mode):
        disk = DiskUsage(paths_to_monitor=paths)

        super(DiskPluginMetric, self).__init__(
            disk, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def update(self, strategy):
        self._metric.update()


class MinibatchDiskUsage(DiskPluginMetric):
    """
    The minibatch Disk usage metric.
    This plugin metric only works at training time.

    At the end of each iteration, this metric logs the total
    size (in KB) of all the monitored paths.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochDiskUsage`.
    """

    def __init__(self, paths_to_monitor):
        """
        Creates an instance of the minibatch Disk usage metric.
        """
        super(MinibatchDiskUsage, self).__init__(
            paths_to_monitor,
            reset_at="iteration",
            emit_at="iteration",
            mode="train",
        )

    def __str__(self):
        return "DiskUsage_MB"


class EpochDiskUsage(DiskPluginMetric):
    """
    The Epoch Disk usage metric.
    This plugin metric only works at training time.

    At the end of each epoch, this metric logs the total
    size (in KB) of all the monitored paths.
    """

    def __init__(self, paths_to_monitor):
        """
        Creates an instance of the epoch Disk usage metric.
        """
        super(EpochDiskUsage, self).__init__(
            paths_to_monitor, reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "DiskUsage_Epoch"


class ExperienceDiskUsage(DiskPluginMetric):
    """
    The average experience Disk usage metric.
    This plugin metric works only at eval time.

    At the end of each experience, this metric logs the total
    size (in KB) of all the monitored paths.
    """

    def __init__(self, paths_to_monitor):
        """
        Creates an instance of the experience Disk usage metric.
        """
        super(ExperienceDiskUsage, self).__init__(
            paths_to_monitor,
            reset_at="experience",
            emit_at="experience",
            mode="eval",
        )

    def __str__(self):
        return "DiskUsage_Exp"


class StreamDiskUsage(DiskPluginMetric):
    """
    The average stream Disk usage metric.
    This plugin metric works only at eval time.

    At the end of the eval stream, this metric logs the total
    size (in KB) of all the monitored paths.
    """

    def __init__(self, paths_to_monitor):
        """
        Creates an instance of the stream Disk usage metric.
        """
        super(StreamDiskUsage, self).__init__(
            paths_to_monitor, reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "DiskUsage_Stream"


def disk_usage_metrics(
    *,
    paths_to_monitor=None,
    minibatch=False,
    epoch=False,
    experience=False,
    stream=False
) -> List[DiskPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    standalone metrics.

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

    metrics: List[DiskPluginMetric] = []
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
    "DiskUsage",
    "MinibatchDiskUsage",
    "EpochDiskUsage",
    "ExperienceDiskUsage",
    "StreamDiskUsage",
    "disk_usage_metrics",
]
