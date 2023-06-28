################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 29-03-2022                                                             #
# Author(s): Rudy Semola                                                       #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import TYPE_CHECKING, List, Union, Dict

import torch
from torch import Tensor
import torchmetrics
from torchmetrics.functional import accuracy

from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task

from collections import defaultdict
from packaging import version

if TYPE_CHECKING:
    from avalanche.training.templates.common_templates import SupervisedTemplate


class TopkAccuracy(Metric[Dict[int, float]]):
    """
    The Top-k Accuracy metric. This is a standalone metric.
    It is defined using torchmetrics.functional accuracy with top_k
    """

    def __init__(self, top_k: int):
        """
        Creates an instance of the standalone Top-k Accuracy metric.

        By default this metric in its initial state will return a value of 0.
        The metric can be updated by using the `update` method while
        the running top-k accuracy can be retrieved using the `result` method.

        :param top_k: integer number to define the value of k.
        """
        self._topk_acc_dict: Dict[int, Mean] = defaultdict(Mean)
        self.top_k: int = top_k

        self.__torchmetrics_requires_task = version.parse(
            torchmetrics.__version__
        ) >= version.parse("0.11.0")

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        task_labels: Union[float, Tensor],
    ) -> None:
        """
        Update the running top-k accuracy given the true and predicted labels.
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

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if isinstance(task_labels, int):
            total_patterns = len(true_y)
            self._topk_acc_dict[task_labels].update(
                self._compute_topk_acc(predicted_y, true_y, top_k=self.top_k),
                total_patterns,
            )
        elif isinstance(task_labels, Tensor):
            for pred, true, t in zip(predicted_y, true_y, task_labels):
                self._topk_acc_dict[int(t)].update(
                    self._compute_topk_acc(pred, true, top_k=self.top_k), 1
                )
        else:
            raise ValueError(
                f"Task label type: {type(task_labels)}, "
                f"expected int/float or Tensor"
            )

    def _compute_topk_acc(self, pred, gt, top_k):
        if self.__torchmetrics_requires_task:
            num_classes = int(torch.max(torch.as_tensor(gt))) + 1
            pred_t = torch.as_tensor(pred)
            if len(pred_t.shape) > 1:
                num_classes = max(num_classes, pred_t.shape[1])

            return accuracy(
                pred, gt, task="multiclass", num_classes=num_classes, top_k=self.top_k
            )
        else:
            return accuracy(pred, gt, top_k=self.top_k)

    def result_task_label(self, task_label: int) -> Dict[int, float]:
        """
        Retrieves the running top-k accuracy.

        Calling this method will not change the internal state of the metric.

        :param task_label: if None, return the entire dictionary of accuracies
            for each task. Otherwise return the dictionary

        :return: A dictionary `{task_label: topk_accuracy}`, where the accuracy
            is a float value between 0 and 1.
        """
        assert task_label is not None
        return {task_label: self._topk_acc_dict[task_label].result()}

    def result(self) -> Dict[int, float]:
        """
        Retrieves the running top-k accuracy for all tasks.

        Calling this method will not change the internal state of the metric.

        :return: A dict of running top-k accuracies for each task label,
            where each value is a float value between 0 and 1.
        """
        return {k: v.result() for k, v in self._topk_acc_dict.items()}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._topk_acc_dict = defaultdict(Mean)
        else:
            self._topk_acc_dict[task_label].reset()


class TopkAccuracyPluginMetric(GenericPluginMetric[Dict[int, float], TopkAccuracy]):
    """
    Base class for all top-k accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, top_k):
        super(TopkAccuracyPluginMetric, self).__init__(
            TopkAccuracy(top_k=top_k), reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def reset(self, strategy=None) -> None:
        if self._reset_at == "stream" or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> Dict[int, float]:
        if self._emit_at == "stream" or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result_task_label(phase_and_task(strategy)[1])

    def update(self, strategy: "SupervisedTemplate"):
        assert strategy.experience is not None
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._metric.update(strategy.mb_output, strategy.mb_y, task_labels)


class MinibatchTopkAccuracy(TopkAccuracyPluginMetric):
    """
    The minibatch plugin top-k accuracy metric.
    This metric only works at training time.

    This metric computes the average top-k accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.
    """

    def __init__(self, top_k):
        """
        Creates an instance of the MinibatchTopkAccuracy metric.
        """
        super(MinibatchTopkAccuracy, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train", top_k=top_k
        )
        self.top_k = top_k

    def __str__(self):
        return "Topk_" + str(self.top_k) + "_Acc_MB"


class EpochTopkAccuracy(TopkAccuracyPluginMetric):
    """
    The average top-k accuracy over a single training epoch.
    This plugin metric only works at training time.

    The top-k accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self, top_k):
        """
        Creates an instance of the EpochTopkAccuracy metric.
        """

        super(EpochTopkAccuracy, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train", top_k=top_k
        )
        self.top_k = top_k

    def __str__(self):
        return "Topk_" + str(self.top_k) + "_Acc_Epoch"


class RunningEpochTopkAccuracy(TopkAccuracyPluginMetric):
    """
    The average top-k accuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the top-k accuracy averaged over all
    patterns seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self, top_k):
        """
        Creates an instance of the RunningEpochTopkAccuracy metric.
        """

        super(RunningEpochTopkAccuracy, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train", top_k=top_k
        )
        self.top_k = top_k

    def __str__(self):
        return "Topk_" + str(self.top_k) + "_Acc_Epoch"


class ExperienceTopkAccuracy(TopkAccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average top-k accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self, top_k):
        """
        Creates an instance of the ExperienceTopkAccuracy metric.
        """
        super(ExperienceTopkAccuracy, self).__init__(
            reset_at="experience",
            emit_at="experience",
            mode="eval",
            top_k=top_k,
        )
        self.top_k = top_k

    def __str__(self):
        return "Topk_" + str(self.top_k) + "_Acc_Exp"


class TrainedExperienceTopkAccuracy(TopkAccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    top-k accuracy for only the experiences
    that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self, top_k):
        """
        Creates an instance of the TrainedExperienceTopkAccuracy metric.
        """
        super(TrainedExperienceTopkAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval", top_k=top_k
        )
        self._current_experience = 0
        self.top_k = top_k

    def after_training_exp(self, strategy):
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        self.reset(strategy)
        return super().after_training_exp(strategy)

    def update(self, strategy):
        """
        Only update the top-k accuracy with results from experiences
        that have been trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            TopkAccuracyPluginMetric.update(self, strategy)

    def __str__(self):
        return "Topk_" + str(self.top_k) + "_Acc_On_Trained_Experiences"


class StreamTopkAccuracy(TopkAccuracyPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average top-k accuracy over all patterns
    seen in all experiences. This metric only works at eval time.
    """

    def __init__(self, top_k):
        """
        Creates an instance of StreamTopkAccuracy metric
        """
        super(StreamTopkAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval", top_k=top_k
        )
        self.top_k = top_k

    def __str__(self):
        return "Topk_" + str(self.top_k) + "_Acc_Stream"


def topk_acc_metrics(
    *,
    top_k=3,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    trained_experience=False,
    stream=False,
) -> List[TopkAccuracyPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch top-k accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch top-k accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch top-k accuracy at training time.
    :param experience: If True, will return a metric able to log
        the top-k accuracy on each evaluation experience.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation top-k accuracy only for experiences that the
        model has been trained on
    :param stream: If True, will return a metric able to log the top-k accuracy
        averaged over the entire evaluation stream of experiences.

    :return: A list of plugin metrics.
    """

    metrics: List[TopkAccuracyPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchTopkAccuracy(top_k=top_k))
    if epoch:
        metrics.append(EpochTopkAccuracy(top_k=top_k))
    if epoch_running:
        metrics.append(RunningEpochTopkAccuracy(top_k=top_k))
    if experience:
        metrics.append(ExperienceTopkAccuracy(top_k=top_k))
    if trained_experience:
        metrics.append(TrainedExperienceTopkAccuracy(top_k=top_k))
    if stream:
        metrics.append(StreamTopkAccuracy(top_k=top_k))

    return metrics


__all__ = [
    "TopkAccuracy",
    "MinibatchTopkAccuracy",
    "EpochTopkAccuracy",
    "RunningEpochTopkAccuracy",
    "ExperienceTopkAccuracy",
    "StreamTopkAccuracy",
    "TrainedExperienceTopkAccuracy",
    "topk_acc_metrics",
]


"""
UNIT TEST
"""
if __name__ == "__main__":
    metric = topk_acc_metrics(trained_experience=True, top_k=5)
    print(metric)
