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

import atexit
import collections
import subprocess
import threading
import time
import warnings
from typing import Optional, TYPE_CHECKING

from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics import Mean
from avalanche.evaluation.metrics._any_event_metric import AnyEventMetric
if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class GpuUsage:
    """
    GPU usage metric measured as average usage percentage over time.

    This metric will actively poll the system to get the GPU usage over time
    starting from the first call to `update`. Subsequent calls to the `update`
    method will consolidate the values gathered since the last call.

    The `result` method will return `None` until the `update` method is invoked
    at least two times.

    Invoking the `reset` method will stop the measurement and reset the metric
    to its initial state.
    """

    MAX_BUFFER = 10000
    SMI_NOT_FOUND_MSG = 'No GPU available: nvidia-smi command not ' \
                        'found. Gpu Usage logging will be disabled.'

    def __init__(self, gpu_id, every=2.0):
        """
        Creates an instance of the GPU usage metric.

        For more info about the usage see the class description.

        :param gpu_id: GPU device ID.
        :param every: time delay (in seconds) between measurements.
        """
        # 'nvidia-smi --loop=1 --query-gpu=utilization.gpu --format=csv'
        self._cmd = ['nvidia-smi', f'--loop={every}',
                     '--query-gpu=utilization.gpu', '--format=csv',
                     f'--id={gpu_id}']

        self._values_queue = collections.deque(maxlen=GpuUsage.MAX_BUFFER)
        self._last_result: Optional[float] = None

        # Long running process
        self._p = None
        self._read_thread = None
        self._nvidia_smi_error: bool = False
        self._nvidia_smi_found: bool = False

    def update(self) -> None:
        """
        Consolidates the values got from the GPU sensor.

        This will store the average for retrieval through the `update` method.

        The previously consolidated value will be discarded.

        :return: None
        """
        if self._p is None:
            self._start_watch()
            return None

        mean_usage = Mean()
        for _ in range(GpuUsage.MAX_BUFFER):
            try:
                queue_element = self._values_queue.popleft()
                mean_usage.update(queue_element[0], queue_element[1])
            except IndexError:
                break
        self._last_result = mean_usage.result()

    def _start_watch(self):
        if self._nvidia_smi_error:
            return
        try:
            self._p = subprocess.Popen(self._cmd, bufsize=1,
                                       stdout=subprocess.PIPE)
            self._read_thread = threading.Thread(target=self._push_lines,
                                                 daemon=True)
            self._read_thread.start()
            atexit.register(self.reset)
            self._nvidia_smi_error = False
            self._nvidia_smi_found = True
        except (subprocess.SubprocessError, OSError):
            self._nvidia_smi_error = True
            self._nvidia_smi_found = False
            warnings.warn(GpuUsage.SMI_NOT_FOUND_MSG)

    def _push_lines(self) -> None:
        last_time = None
        last_usage = None
        values_queue = self._values_queue
        for line in iter(self._p.stdout.readline, b''):
            decoded = line.decode('ascii')

            if decoded[0] == 'u':  # skip first line 'utilization.gpu [%]'
                continue
            current_time = time.time()
            # [:-1] removes the trailing "%"
            current_usage = float(decoded.strip()[:-1].strip()) / 100

            if last_usage is None:
                last_time = current_time
                last_usage = current_usage
                continue

            # last_usage not None -> also last_time not None
            queue_element = ((last_usage + current_usage) / 2,
                             current_time - last_time)
            values_queue.append(queue_element)
            last_time = current_time
            last_usage = current_usage

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        if self._p is None:
            return None

        self._p.terminate()
        try:
            self._p.wait(0.5)
        except subprocess.TimeoutExpired:
            self._p.kill()
        self._p = None

        self._read_thread.join()
        self._last_result = None
        self._values_queue.clear()
        self._values_queue = collections.deque(maxlen=GpuUsage.MAX_BUFFER)

    def result(self) -> Optional[float]:
        """
        Returns the last consolidated GPU usage value.

        For more info about the returned value see the class description.

        :return: The percentage GPU usage as a float value in range [0, 1].
            Returns None if the `update` method was invoked less than twice.
        """
        return self._last_result

    def gpu_found(self) -> bool:
        """
        Checks if nvidia-smi could me executed.

        This method is experimental. Please use at you own risk.

        :return: True if nvidia-smi could be launched, False otherwise.
        """
        return self._nvidia_smi_found


class GpuUsageMonitor(AnyEventMetric[float]):
    """
    The GPU usage metric.

    This metric logs the percentage GPU usage.

    The metric can be either configured to log after a certain timeout or
    at each event.

    GPU usage is logged separately for the train and eval phases.
    """

    def __init__(self, gpu_id: int, *, timeout: int = 2,
                 train=True, eval=False):
        """
        Creates an instance of the GPU usage metric.

        The train and eval parameters can be True at the same time. However,
        at least one of them must be True.

        :param gpu_id: The GPU to monitor.
        :param timeout: The timeout between each GPU usage log, in seconds.
             Defaults to 2 seconds. Must be an int.
        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param eval: When True, the metric will be computed on the eval
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not eval:
            raise ValueError('train and eval can\'t be both False at the same'
                             ' time.')

        self._gpu_sensor = GpuUsage(gpu_id, every=1)
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
            self._gpu_sensor.update()
            return self._package_result(strategy)

    def before_training(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        if not self._track_train_usage:
            self._gpu_sensor.reset()
        else:
            self._gpu_sensor.update()

        return super().before_training(strategy)

    def before_eval(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        if not self._track_train_usage:
            self._gpu_sensor.reset()
        else:
            self._gpu_sensor.update()

        return super().before_eval(strategy)

    def result(self) -> Optional[float]:
        gpu_result = self._gpu_sensor.result()
        if gpu_result is None:
            return None
        return gpu_result * 100.0

    def reset(self) -> None:
        self._gpu_sensor.reset()
        self._last_time = None

    def _package_result(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        phase_name, _ = phase_and_task(strategy)
        experience_gpu = self.result()

        metric_name = 'GPU_usage/{}'.format(phase_name)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, experience_gpu, plot_x_position)]


__all__ = [
    'GpuUsage',
    'GpuUsageMonitor'
]
