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

from typing import Dict, Optional, Set

from avalanche.evaluation import OnTestStepEnd, AggregatedMetric, OnTestPhaseEnd
from avalanche.evaluation.metric_definitions import TResult
from avalanche.evaluation.metric_results import MetricTypes, MetricValue, \
    MetricResult
from avalanche.evaluation.abstract_metric import AbstractMetric
from avalanche.evaluation.metric_units import AverageAccuracyUnit, MetricUnit
from avalanche.evaluation.metrics import TaskAccuracy


class TaskForgetting(AggregatedMetric[Dict[int, float], TaskAccuracy]):
    """
    The TaskForgetting metric, describing the accuracy loss detected for a
    certain task.

    This metric is computed separately for each task as the difference between
    the accuracy result obtained after training on a task and the accuracy
    result obtained on the same task at the end of successive steps.

    This metric is computed during the test phase only.
    TODO: doc
    """

    def __init__(self):
        """
        Creates an instance of the Catastrophic TaskForgetting metric.

        """
        super().__init__(TaskAccuracy())

        self._initial_task_accuracy: Dict[int, float] = dict()
        """
        The initial accuracy of each task.
        """

    def result(self) -> Dict[int, float]:
        task_accuracies: Dict[int, float] = self.base_metric.result()
        all_task_labels: Set[int] = set(self._initial_task_accuracy.keys())\
            .union(set(task_accuracies.keys()))
        task_forgetting: Dict[int, float] = dict()
        for task_id in all_task_labels:
            delta = 0.0
            if (task_id in task_accuracies) and \
                    (task_id in self._initial_task_accuracy):
                # Task already encountered in previous phases
                delta = self._initial_task_accuracy[task_id] - \
                        task_accuracies[task_id]
            # Other situations:
            # - A task that was not encountered before (forgetting == 0)
            # - A task that was encountered before, but has not been
            # encountered in the current test phase (forgetting == N.A. == 0)
            task_forgetting[task_id] = delta
        return task_forgetting

    def before_test(self, eval_data) -> None:
        super().before_test(eval_data)
        self.reset()

    def after_test(self, eval_data: OnTestPhaseEnd) -> MetricResult:
        super().after_test(eval_data)
        return self._consolidate_and_package_result(eval_data)

    def _consolidate_and_package_result(self, eval_data) -> MetricResult:
        for task_label, task_accuracy in self.base_metric.result().items():
            if task_label not in self._initial_task_accuracy:
                self._initial_task_accuracy[task_label] = task_accuracy

        metric_values = []
        for task_label, task_forgetting in self.result().items():
            metric_name = 'Task_Forgetting/Task{:03}'.format(task_label)
            plot_x_position = self._next_x_position(metric_name)

            metric_values.append(MetricValue(
                self, metric_name, MetricTypes.FORGETTING,
                task_forgetting, plot_x_position))
        return metric_values


__all__ = ['TaskForgetting']
