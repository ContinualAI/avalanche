################################################################################
# Copyright (c) 2025 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-01-2025                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

from typing import List, Optional, Union, Dict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from collections import defaultdict


class RMSE(Metric[float]):
    """RMSE metric. This is a standalone metric.

    The update method computes the RMSE incrementally
    by keeping a running average of the <prediction, target> pairs
    of Tensors provided over time.

    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average RMSE
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an RMSE value of 0.
    """

    def __init__(self):
        """Creates an instance of the standalone RMSE metric.

        By default this metric in its initial state will return an RMSE
        value of 0. The metric can be updated by using the `update` method
        while the running RMSE can be retrieved using the `result` method.
        """
        self._mean_RMSE = Mean()
        """The mean utility that will be used to store the running RMSE."""

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
    ) -> None:
        """Update the running RMSE given the true and predicted labels.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.

        :return: None.
        """

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        rmse = float(torch.sqrt(torch.mean((true_y - predicted_y) ** 2)))
        total_patterns = len(true_y)
        self._mean_RMSE.update(rmse, total_patterns)

    def result(self) -> float:
        """Retrieves the running RMSE.

        Calling this method will not change the internal state of the metric.

        :return: The current running RMSE, which is a float value
            between 0 and 1.
        """
        return self._mean_RMSE.result()

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        self._mean_RMSE.reset()


class TaskAwareRMSE(Metric[Dict[int, float]]):
    """The task-aware RMSE metric.

    The metric computes a dictionary of <task_label, RMSE value> pairs.
    update/result/reset methods are all task-aware.
    """

    def __init__(self):
        """Creates an instance of the task-aware RMSE metric."""
        self._mean_RMSE = defaultdict(RMSE)
        """
        The mean utility that will be used to store the running RMSE
        for each task label.
        """

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        task_labels: Union[float, Tensor],
    ) -> None:
        """Update the running RMSE given the true and predicted labels.

        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if Float, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.

        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError("Size mismatch for true_y and task_labels tensors")

        if isinstance(task_labels, int):
            self._mean_RMSE[task_labels].update(predicted_y, true_y)
        elif isinstance(task_labels, Tensor):
            for pred, true, t in zip(predicted_y, true_y, task_labels):
                if isinstance(t, Tensor):
                    t = t.item()
                self._mean_RMSE[t].update(pred.unsqueeze(0), true.unsqueeze(0))
        else:
            raise ValueError(
                f"Task label type: {type(task_labels)}, "
                f"expected int/float or Tensor"
            )

    def result(self, task_label: Optional[int] = None) -> Dict[int, float]:
        """
        Retrieves the running RMSE.

        Calling this method will not change the internal state of the metric.

        task label is ignored if `self.split_by_task=False`.

        :param task_label: if None, return the entire dictionary of RMSE
            for each task. Otherwise return the dictionary
            `{task_label: RMSE}`.
        :return: A dict of running RMSE for each task label,
            where each value is a float value between 0 and 1.
        """
        assert task_label is None or isinstance(task_label, int)

        if task_label is None:
            return {k: v.result() for k, v in self._mean_RMSE.items()}
        else:
            return {task_label: self._mean_RMSE[task_label].result()}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        task label is ignored if `self.split_by_task=False`.

        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_RMSE = defaultdict(RMSE)
        else:
            self._mean_RMSE[task_label].reset()


class RMSEPluginMetric(GenericPluginMetric[float, RMSE]):
    """
    Base class for all RMSE plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the RMSE plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware RMSE or not.
        """
        super().__init__(RMSE(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)


class RMSEPerTaskPluginMetric(GenericPluginMetric[Dict[int, float], TaskAwareRMSE]):
    """
    Base class for all RMSE plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode):
        """Creates the RMSE plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware RMSE or not.
        """
        super().__init__(TaskAwareRMSE(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> Dict[int, float]:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y, strategy.mb_task_id)


class MinibatchRMSE(RMSEPluginMetric):
    """
    The minibatch plugin RMSE metric.
    This metric only works at training time.

    This metric computes the average RMSE over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochRMSE` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchRMSE metric.
        """
        super(MinibatchRMSE, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Top1_RMSE_MB"


class EpochRMSE(RMSEPluginMetric):
    """
    The average RMSE over a single training epoch.
    This plugin metric only works at training time.

    The RMSE will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochRMSE metric.
        """

        super(EpochRMSE, self).__init__(reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "Top1_RMSE_Epoch"


class RunningEpochRMSE(RMSEPluginMetric):
    """
    The average RMSE across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the RMSE averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochRMSE metric.
        """

        super(RunningEpochRMSE, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Top1_RunningRMSE_Epoch"


class ExperienceRMSE(RMSEPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average RMSE over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceRMSE metric
        """
        super(ExperienceRMSE, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "Top1_RMSE_Exp"


class StreamRMSE(RMSEPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average RMSE over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamRMSE metric
        """
        super(StreamRMSE, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "Top1_RMSE_Stream"


class TrainedExperienceRMSE(RMSEPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    RMSE for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceRMSE metric by first
        constructing RMSEPluginMetric
        """
        super(TrainedExperienceRMSE, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )
        self._current_experience = 0

    def after_training_exp(self, strategy):
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        self.reset()
        return super().after_training_exp(strategy)

    def update(self, strategy):
        """
        Only update the RMSE with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            RMSEPluginMetric.update(self, strategy)

    def __str__(self):
        return "RMSE_On_Trained_Experiences"


def rmse_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
) -> List[RMSEPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch RMSE at training time.
    :param epoch: If True, will return a metric able to log
        the epoch RMSE at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch RMSE at training time.
    :param experience: If True, will return a metric able to log
        the RMSE on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the RMSE averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation RMSE only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics: List[RMSEPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchRMSE())

    if epoch:
        metrics.append(EpochRMSE())

    if epoch_running:
        metrics.append(RunningEpochRMSE())

    if experience:
        metrics.append(ExperienceRMSE())

    if stream:
        metrics.append(StreamRMSE())

    if trained_experience:
        metrics.append(TrainedExperienceRMSE())

    return metrics


__all__ = [
    "RMSE",
    "TaskAwareRMSE",
    "MinibatchRMSE",
    "EpochRMSE",
    "RunningEpochRMSE",
    "ExperienceRMSE",
    "StreamRMSE",
    "TrainedExperienceRMSE",
    "rmse_metrics",
    "RMSEPerTaskPluginMetric",
]
