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

from typing import Union

from avalanche.evaluation import OnTrainEpochEnd, OnTestStepEnd, MetricValue, \
    MetricTypes
from avalanche.evaluation.abstract_metric import AbstractMetric
from avalanche.evaluation.metric_units import AverageAccuracyUnit
from avalanche.evaluation.metric_utils import filter_accepted_events, \
    get_task_label


class EpochAccuracy(AbstractMetric):
    """
    The average accuracy metric.

    This metric is computed separately for each task.

    The accuracy will be emitted after each epoch by aggregating minibatch
    values. Beware that the training accuracy is the "running" one.
    """
    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the EpochAccuracy metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to True.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')

        # Create accuracy unit
        self.accuracy_unit = AverageAccuracyUnit(on_train_epochs=train,
                                                 on_test_epochs=test)

        # When computing the accuracy metric we need to get EpochEnd events
        # to check if the epoch ended. The actual element in charge of
        # accumulating the running accuracy is the accuracy_unit.
        on_events = filter_accepted_events(
            [OnTrainEpochEnd, OnTestStepEnd], train=train, test=test)

        # Attach callbacks
        self._attach(self.accuracy_unit)\
            ._on(on_events, self.result_emitter)

    def result_emitter(self, eval_data):
        eval_data: Union[OnTrainEpochEnd, OnTestStepEnd]
        # This simply queries accuracy_unit for the accuracy value and
        # emits that value by labeling it with the appropriate name.
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        metric_value = self.accuracy_unit.value

        metric_name = 'Top1/{}/Task{:03}'.format(phase_name, task_label)
        plot_x_position = self._next_x_position(metric_name)

        return MetricValue(self, metric_name, MetricTypes.ACCURACY,
                           metric_value, plot_x_position)


__all__ = ['EpochAccuracy']
