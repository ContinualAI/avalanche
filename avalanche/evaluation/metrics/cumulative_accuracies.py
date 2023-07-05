################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from collections import defaultdict
from typing import Dict, List, Union, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from avalanche.benchmarks import OnlineCLExperience
from avalanche.evaluation import GenericPluginMetric, Metric, PluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import (
    phase_and_task,
    stream_type,
    generic_get_metric_name,
    default_metric_name_template,
)
from avalanche.evaluation.metric_results import MetricValue

if TYPE_CHECKING:
    from avalanche.evaluation.metric_results import MetricResult
    from avalanche.training.templates import SupervisedTemplate


class CumulativeAccuracy(Metric[Dict[int, float]]):
    """
    Metric used by the CumulativeAccuracyPluginMetric,
    holds a dictionnary of per-task cumulative accuracies
    and updates the cumulative accuracy based on the classes splits
    provided for the growing incremental task.
    The update is performed as described in the paper
    "On the importance of cross-task
    features for class-incremental learning"
    Soutif et. al, https://arxiv.org/abs/2106.11930
    """

    def __init__(self):
        self._mean_accuracy = defaultdict(lambda: Mean())

    @torch.no_grad()
    def update(
        self,
        classes_splits,
        predicted_y: Tensor,
        true_y: Tensor,
    ) -> None:
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)
        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y " "and predicted_y tensors")
        for t, classes in classes_splits.items():
            # This is to fix a weird bug
            # that was happening in some workflows
            if t not in self._mean_accuracy:
                self._mean_accuracy[t]

            # Only compute Accuracy for classes that are in classes set
            if len(set(true_y.cpu().numpy()).intersection(classes)) == 0:
                # Here this assumes that true_y is only
                # coming from the same classes split,
                # this is a shortcut
                # but sometimes this is not true so we
                # do additional filtering later to make sure
                continue

            idxs = np.where(np.isin(true_y.cpu(), list(classes)))[0]
            y = true_y[idxs]
            logits_exp = predicted_y[idxs, :]

            logits_exp = logits_exp[:, list(classes)]
            prediction = torch.argmax(logits_exp, dim=1)

            # Here remap predictions to true y range
            prediction = torch.tensor(list(classes))[prediction.cpu()]

            true_positives = float(torch.sum(torch.eq(prediction, y.cpu())))
            total_patterns = len(y)
            self._mean_accuracy[t].update(
                true_positives / total_patterns, total_patterns
            )

    def result(self) -> Dict[int, float]:
        """Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        :return: The current running accuracy, which is a float value
            between 0 and 1.
        """
        return {t: self._mean_accuracy[t].result() for t in self._mean_accuracy}

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        for t in self._mean_accuracy:
            self._mean_accuracy[t].reset()


class CumulativeAccuracyPluginMetric(
    GenericPluginMetric[Dict[int, float], CumulativeAccuracy]
):
    def __init__(self, reset_at="stream", emit_at="stream", mode="eval"):
        """
        Creates the CumulativeAccuracy plugin metric,
        this stores and updates the Cumulative Accuracy metric described in
        "On the importance of cross-task
        features for class-incremental learning"
        Soutif et. al, https://arxiv.org/abs/2106.11930
        """

        self.classes_seen_so_far = set()
        self.classes_splits = {}
        super().__init__(
            CumulativeAccuracy(), reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def before_training_exp(self, strategy, **kwargs):
        super().before_training_exp(strategy, **kwargs)
        if isinstance(strategy.experience, OnlineCLExperience):
            new_classes = set(
                strategy.experience.logging().origin_experience.classes_in_this_experience
            )

            task_id = strategy.experience.logging().origin_experience.current_experience
        else:
            new_classes = set(strategy.experience.classes_in_this_experience)
            task_id = strategy.experience.current_experience

        self.classes_seen_so_far = self.classes_seen_so_far.union(new_classes)
        self.classes_splits[task_id] = self.classes_seen_so_far

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> Dict[int, float]:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(self.classes_splits, strategy.mb_output, strategy.mb_y)

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        assert strategy.experience is not None
        metric_value = self.result()
        plot_x_position = strategy.clock.train_iterations

        phase_name, task_label = phase_and_task(strategy)
        stream = stream_type(strategy.experience)

        metrics = []
        for k, v in metric_value.items():
            metric_name = generic_get_metric_name(
                default_metric_name_template,
                {
                    "metric_name": str(self),
                    "task_label": None,
                    "phase_name": phase_name,
                    "experience_id": k,
                    "stream_name": stream,
                },
            )
            metrics.append(MetricValue(self, metric_name, v, plot_x_position))
        return metrics

    def __repr__(self):
        return "CumulativeAccuracy"


class CumulativeForgettingPluginMetric(
    GenericPluginMetric[Dict[int, float], CumulativeAccuracy]
):
    """
    The CumulativeForgetting metric, describing the accuracy loss
    detected for a certain experience.

    This plugin metric, computed separately for each experience,
    is the difference between the cumulative accuracy result obtained after
    first training on a experience and the accuracy result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self, reset_at="stream", emit_at="stream", mode="eval"):
        """
        Creates an instance of the CumulativeForgetting metric.
        """

        self.classes_splits = {}
        self.classes_seen_so_far = set()

        self.initial = {}
        self.last = {}

        self.train_task_id = None

        super().__init__(
            CumulativeAccuracy(), reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def before_training_exp(self, strategy, **kwargs):
        super().before_training_exp(strategy, **kwargs)
        if isinstance(strategy.experience, OnlineCLExperience):
            if strategy.experience.access_task_boundaries:
                new_classes = set(
                    strategy.experience.origin_experience.classes_in_this_experience
                )
                task_id = strategy.experience.origin_experience.current_experience
            else:
                raise AttributeError(
                    "Online Scenario has to allow "
                    "access to task boundaries for"
                    " the Cumulative Accuracy Metric"
                    " to be computed"
                )
        else:
            new_classes = set(strategy.experience.classes_in_this_experience)
            task_id = strategy.experience.current_experience

        self.classes_seen_so_far = self.classes_seen_so_far.union(new_classes)
        self.classes_splits[task_id] = self.classes_seen_so_far

        # Update train task id
        experience = strategy.experience
        if isinstance(experience, OnlineCLExperience):
            self.train_task_id = experience.origin_experience.current_experience
        else:
            self.train_task_id = experience.current_experience

    def reset(self):
        self._metric.reset()

    def result(self) -> Dict[int, float]:
        forgetting = self._compute_forgetting()
        return forgetting

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        assert strategy.experience is not None
        metric_value = self.result()
        plot_x_position = strategy.clock.train_iterations

        phase_name, task_label = phase_and_task(strategy)
        stream = stream_type(strategy.experience)

        metrics = []
        for k, v in metric_value.items():
            metric_name = generic_get_metric_name(
                default_metric_name_template,
                {
                    "metric_name": str(self),
                    "task_label": None,
                    "phase_name": phase_name,
                    "experience_id": k,
                    "stream_name": stream,
                },
            )
            metrics.append(MetricValue(self, metric_name, v, plot_x_position))
        return metrics

    def update(self, strategy):
        self._metric.update(self.classes_splits, strategy.mb_output, strategy.mb_y)

    def _compute_forgetting(self):
        for t, item in self._metric.result().items():
            if t not in self.initial:
                self.initial[t] = item
            else:
                self.last[t] = item

        forgetting = {}
        for k, v in self.last.items():
            forgetting[k] = self.initial[k] - self.last[k]

        return forgetting

    def __str__(self):
        return "CumulativeForgetting"


__all__ = [
    "CumulativeAccuracyPluginMetric",
    "CumulativeForgettingPluginMetric",
    "CumulativeAccuracy",
]
