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

from typing import Dict

from avalanche.evaluation import OnTestStepEnd, MetricValue, MetricTypes
from avalanche.evaluation.abstract_metric import AbstractMetric
from avalanche.evaluation.metric_units import AverageAccuracyUnit, MetricUnit


class TaskForgetting(AbstractMetric):
    """
    The TaskForgetting metric, describing the accuracy loss detected for a
    certain task.

    This metric is computed separately for each task as the difference between
    the accuracy result obtained after training on a task and the accuracy
    result obtained on the same task at the end of successive steps.

    This metric is computed during the test phase only.
    """
    def __init__(self):
        """
        Creates an instance of the Catastrophic TaskForgetting metric.

        """
        super().__init__()

        self.best_accuracy: Dict[int, float] = dict()
        """
        The best accuracy of each task.
        """

        # Create accuracy unit
        self._accuracy_unit: MetricUnit = AverageAccuracyUnit(
            on_train_epochs=False, on_test_epochs=True)

        # Attach callbacks
        self._attach(self._accuracy_unit)\
            ._on(OnTestStepEnd, self.result_emitter)

    def result_emitter(self, eval_data):
        eval_data: OnTestStepEnd
        train_task_label = eval_data.training_task_label
        test_task_label = eval_data.test_task_label
        accuracy_value = self._accuracy_unit.value

        if test_task_label not in self.best_accuracy and \
                train_task_label == test_task_label:
            self.best_accuracy[test_task_label] = accuracy_value

        forgetting = 0.0

        if test_task_label in self.best_accuracy:
            forgetting = self.best_accuracy[test_task_label] - accuracy_value

        metric_name = 'TaskForgetting/Task{:03}'.format(test_task_label)
        plot_x_position = self._next_x_position(metric_name)

        return MetricValue(self, metric_name, MetricTypes.FORGETTING,
                           forgetting, plot_x_position)


__all__ = ['TaskForgetting']
