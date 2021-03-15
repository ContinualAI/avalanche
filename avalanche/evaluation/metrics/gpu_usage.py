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


import GPUtil
from threading import Thread
import time
import warnings
from typing import Optional, TYPE_CHECKING, List

from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, \
    phase_and_task, stream_type
if TYPE_CHECKING:
    from avalanche.training import PluggableStrategy


class MaxGPU(Metric[float]):
    """
    The GPU usage metric.
    Important: this metric approximates the real maximum GPU percentage
     usage since it sample at discrete amount of time the GPU values.

    Instances of this metric keeps the maximum GPU usage percentage detected.
    The update method starts the usage tracking. The reset method stops
    the tracking.

    The result, obtained using the `result` method, is the usage in mega-bytes.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an usage value of 0.
    """

    def __init__(self, gpu_id, every=0.5):
        """
        Creates an instance of the GPU usage metric.

        :param gpu_id: GPU device ID.
        :param every: seconds after which update the maximum GPU
            usage
        """

        self.every = every
        self.gpu_id = gpu_id

        n_gpus = len(GPUtil.getGPUs())
        if n_gpus == 0:
            warnings.warn("Your system has no GPU!")
            self.gpu_id = None
        elif gpu_id < 0:
            warnings.warn("GPU metric called with negative GPU id."
                          "GPU logging disabled")
            self.gpu_id = None
        else:
            if gpu_id >= n_gpus:
                warnings.warn(f"GPU {gpu_id} not found. Using GPU 0.")
                self.gpu_id = 0

        self.thread = None
        """
        Thread executing GPU monitoring code
        """

        self.stop_f = False
        """
        Flag to stop the thread
        """

        self.max_usage = 0
        """
        Main metric result. Max GPU usage.
        """

    def _f(self):
        """
        Until a stop signal is encountered,
        this function monitors each `every` seconds
        the maximum amount of GPU used by the process
        """
        start_time = time.monotonic()
        while not self.stop_f:
            # GPU percentage
            gpu_perc = GPUtil.getGPUs()[self.gpu_id].load * 100
            if gpu_perc > self.max_usage:
                self.max_usage = gpu_perc
            time.sleep(self.every - ((time.monotonic() - start_time)
                                     % self.every))

    def start_thread(self):
        if self.gpu_id:
            assert not self.thread, "Trying to start thread " \
                                    "without joining the previous."
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

    def result(self) -> Optional[float]:
        """
        Returns the max GPU percentage value.

        :return: The percentage GPU usage as a float value in range [0, 1].
        """
        return self.max_usage


class MinibatchMaxGPU(PluginMetric[float]):
    """
    The Minibatch Max GPU metric.
    This metric only works at training time.
    """

    def __init__(self, gpu_id, every=0.5):
        """
        Creates an instance of the Minibatch Max GPU metric

        :param gpu_id: GPU device ID.
        :param every: seconds after which update the maximum GPU
            usage
        """
        super().__init__()

        self.gpu_id = gpu_id
        self._gpu = MaxGPU(gpu_id, every)

    def before_training(self, strategy: 'PluggableStrategy') \
            -> None:
        self._gpu.start_thread()

    def before_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        return self._package_result(strategy)

    def after_training(self, strategy: 'PluggableStrategy') -> None:
        self._gpu.stop_thread()

    def reset(self) -> None:
        self._gpu.reset()

    def result(self) -> float:
        return self._gpu.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        gpu_usage = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, gpu_usage, plot_x_position)]

    def __str__(self):
        return f"MaxGPU{self.gpu_id}Usage_MB"


class EpochMaxGPU(PluginMetric[float]):
    """
    The Epoch Max GPU metric.
    This metric only works at training time.
    """

    def __init__(self, gpu_id, every=0.5):
        """
        Creates an instance of the epoch Max GPU metric.

        :param gpu_id: GPU device ID.
        :param every: seconds after which update the maximum GPU
            usage
        """
        super().__init__()

        self.gpu_id = gpu_id
        self._gpu = MaxGPU(gpu_id, every)

    def before_training(self, strategy: 'PluggableStrategy') \
            -> None:
        self._gpu.start_thread()

    def before_training_epoch(self, strategy) -> MetricResult:
        self.reset()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        return self._package_result(strategy)

    def after_training(self, strategy: 'PluggableStrategy') -> None:
        self._gpu.stop_thread()

    def reset(self) -> None:
        self._gpu.reset()

    def result(self) -> float:
        return self._gpu.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        gpu_usage = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, gpu_usage, plot_x_position)]

    def __str__(self):
        return f"MaxGPU{self.gpu_id}Usage_Epoch"


class ExperienceMaxGPU(PluginMetric[float]):
    """
    The Experience Max GPU metric.
    This metric only works at eval time.
    """

    def __init__(self, gpu_id, every=0.5):
        """
        Creates an instance of the Experience CPU usage metric.

        :param gpu_id: GPU device ID.
        :param every: seconds after which update the maximum GPU
            usage
        """
        super().__init__()

        self.gpu_id = gpu_id
        self._gpu = MaxGPU(gpu_id, every)

    def before_eval(self, strategy: 'PluggableStrategy') \
            -> None:
        self._gpu.start_thread()

    def before_eval_exp(self, strategy) -> MetricResult:
        self.reset()

    def after_eval_exp(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        return self._package_result(strategy)

    def after_eval(self, strategy: 'PluggableStrategy') -> None:
        self._gpu.stop_thread()

    def reset(self) -> None:
        self._gpu.reset()

    def result(self) -> float:
        return self._gpu.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        gpu_usage = self.result()

        metric_name = get_metric_name(self, strategy, add_experience=True)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, gpu_usage, plot_x_position)]

    def __str__(self):
        return f"MaxGPU{self.gpu_id}Usage_Experience"


class StreamMaxGPU(PluginMetric[float]):
    """
    The Stream Max GPU metric.
    This metric only works at eval time.
    """

    def __init__(self, gpu_id, every=0.5):
        """
        Creates an instance of the Experience CPU usage metric.

        :param gpu_id: GPU device ID.
        :param every: seconds after which update the maximum GPU
            usage
        """
        super().__init__()

        self.gpu_id = gpu_id
        self._gpu = MaxGPU(gpu_id, every)

    def before_eval(self, strategy) -> MetricResult:
        self.reset()
        self._gpu.start_thread()

    def after_eval(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        packed = self._package_result(strategy)
        self._gpu.stop_thread()
        return packed

    def reset(self) -> None:
        self._gpu.reset()

    def result(self) -> float:
        return self._gpu.result()

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        gpu_usage = self.result()

        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = '{}/{}_phase/{}_stream' \
            .format(str(self),
                    phase_name,
                    stream)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, gpu_usage, plot_x_position)]

    def __str__(self):
        return f"MaxGPU{self.gpu_id}Usage_Stream"


def gpu_usage_metrics(gpu_id, every=0.5, minibatch=False, epoch=False,
                      experience=False, stream=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of metric.

    :param gpu_id: GPU device ID.
    :param every: seconds after which update the maximum GPU
        usage
    :param minibatch: If True, will return a metric able to log the minibatch
        max GPU usage.
    :param epoch: If True, will return a metric able to log the epoch
        max GPU usage.
    :param experience: If True, will return a metric able to log the experience
        max GPU usage.
    :param stream: If True, will return a metric able to log the evaluation
        max stream GPU usage.

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchMaxGPU(gpu_id, every))

    if epoch:
        metrics.append(EpochMaxGPU(gpu_id, every))

    if experience:
        metrics.append(ExperienceMaxGPU(gpu_id, every))

    if stream:
        metrics.append(StreamMaxGPU(gpu_id, every))

    return metrics


__all__ = [
    'MaxGPU',
    'MinibatchMaxGPU',
    'EpochMaxGPU',
    'ExperienceMaxGPU',
    'StreamMaxGPU',
    'gpu_usage_metrics'
]
