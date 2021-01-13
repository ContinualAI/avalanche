#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import Dict, Set

from torch import Tensor

from avalanche.evaluation import OnTestPhaseEnd, \
    OnTestIterationEnd
from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.evaluation.metric_results import MetricTypes, MetricValue, \
    MetricResult
from avalanche.evaluation.metrics import Accuracy


class TaskForgetting(PluginMetric[Dict[int, float]]):
    """
    The TaskForgetting metric, describing the accuracy loss detected for a
    certain task.

    This metric, computed separately for each task, is the difference between
    the accuracy result obtained after first training on a task and the accuracy
    result obtained on the same task at the end of successive steps.

    This metric is computed during the test phase only.
    """

    def __init__(self):
        """
        Creates an instance of the Catastrophic TaskForgetting metric.
        """

        super().__init__()

        self._initial_task_accuracy: Dict[int, float] = dict()
        """
        The initial accuracy of each task.
        """

        self._current_task_accuracy: Dict[int, Accuracy] = dict()
        """
        The current accuracy of each task.
        """

    def reset(self) -> None:
        """
        Resets the metric.

        Beware that this will also reset the initial accuracy of each task!

        :return: None.
        """
        self._initial_task_accuracy = dict()
        self._current_task_accuracy = dict()

    def reset_current_accuracy(self) -> None:
        """
        Resets the current accuracy.

        This will preserve the initial accuracy value of each task. To be used
        at the beginning of each test step.

        :return: None.
        """
        self._current_task_accuracy = dict()

    def update(self, true_y: Tensor, predicted_y: Tensor, task_label: int) \
            -> None:
        """
        Updates the running accuracy of a task given the ground truth and
        predicted labels of a minibatch.

        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param predicted_y: The ground truth. Both labels and logit vectors
            are supported.
        :param task_label: The task label.
        :return: None.
        """
        if task_label not in self._current_task_accuracy:
            self._current_task_accuracy[task_label] = Accuracy()
        self._current_task_accuracy[task_label].update(true_y, predicted_y)

    def before_test(self, eval_data) -> None:
        self.reset_current_accuracy()

    def after_test_iteration(self, eval_data: OnTestIterationEnd) -> None:
        self.update(eval_data.ground_truth,
                    eval_data.prediction_logits,
                    eval_data.test_task_label)

    def after_test(self, eval_data: OnTestPhaseEnd) -> MetricResult:
        return self._package_result(eval_data.training_task_label)

    def result(self) -> Dict[int, float]:
        """
        Return the amount of forgetting for each task.

        The forgetting is computed as the accuracy difference between the
        initial task accuracy (when first encountered in the training stream)
        and the current accuracy. A positive value means that forgetting
        occurred. A negative value means that the accuracy on that task
        increased.

        :return: A dictionary in which keys are task labels and the values are
            the forgetting measures (as floats in range [-1, 1]).
        """
        prev_accuracies: Dict[int, float] = self._initial_task_accuracy
        task_accuracies: Dict[int, Accuracy] = self._current_task_accuracy
        all_task_labels: Set[int] = set(prev_accuracies.keys()) \
            .union(set(task_accuracies.keys()))
        task_forgetting: Dict[int, float] = dict()
        for task_id in all_task_labels:
            delta = 0.0
            if (task_id in task_accuracies) and \
                    (task_id in self._initial_task_accuracy):
                # Task already encountered in previous phases
                delta = self._initial_task_accuracy[task_id] - \
                        task_accuracies[task_id].result()
            # Other situations:
            # - A task that was not encountered before (forgetting == 0)
            # - A task that was encountered before, but has not been
            # encountered in the current test phase (forgetting == N.A. == 0)
            task_forgetting[task_id] = delta
        return task_forgetting

    def _package_result(self, train_task: int) \
            -> MetricResult:

        # The forgetting value is computed as the difference between the
        # accuracy obtained after training for the first time and the current
        # accuracy. Here we store the initial accuracy.
        if train_task not in self._initial_task_accuracy and \
                train_task in self._current_task_accuracy:
            initial_task_accuracy = self._current_task_accuracy[train_task].\
                result()
            self._initial_task_accuracy[train_task] = initial_task_accuracy

        metric_values = []
        for task_label, task_forgetting in self.result().items():
            metric_name = 'Task_Forgetting/Task{:03}'.format(task_label)
            plot_x_position = self._next_x_position(metric_name)

            metric_values.append(MetricValue(
                self, metric_name, MetricTypes.FORGETTING,
                task_forgetting, plot_x_position))
        return metric_values


__all__ = ['TaskForgetting']
