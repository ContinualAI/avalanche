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

import time
from typing import Union

from avalanche.evaluation import OnTrainEpochStart, OnTestStepStart, \
    OnTrainEpochEnd, OnTestStepEnd, MetricValue, MetricTypes, \
    OnTrainStepStart, OnTrainStepEnd
from avalanche.evaluation.abstract_metric import AbstractMetric
from avalanche.evaluation.metric_utils import filter_accepted_events, \
    get_task_label


class EpochTime(AbstractMetric):
    """
    Time usage metric, measured in seconds.

    The time is measured between the start and end of an epoch.

    Beware that this metric logs a time value for each epoch! For the average
    epoch time use :class:`AverageEpochTime` instead, which logs the average
    the average epoch time for each step.

    By default this metric takes the time on training epochs only but this
    behaviour can be changed by passing test=True in the constructor.
    """

    def __init__(self, *, train=True, test=False):
        """
        Creates an instance of the Epoch Time metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the time will be taken on training epochs.
            Defaults to True.
        :param test: When True, the time will be taken on test epochs.
            Defaults to False.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')

        self._start_time = None

        on_start_events = filter_accepted_events(
            [OnTrainEpochStart, OnTestStepStart], train=train, test=test)

        on_end_events = filter_accepted_events(
            [OnTrainEpochEnd, OnTestStepEnd], train=train, test=test)

        # Attach callbacks
        self._on(on_start_events, self.time_start) \
            ._on(on_end_events, self.result_emitter)

    def time_start(self, _):
        # Epoch start
        self._start_time = time.perf_counter()

    def result_emitter(self, eval_data):
        eval_data: Union[OnTrainEpochEnd, OnTestStepEnd]
        # Epoch end
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        elapsed_time = time.perf_counter() - self._start_time

        metric_name = 'Epoch_Time/{}/Task{:03}'.format(phase_name,
                                                       task_label)
        plot_x_position = self._next_x_position(metric_name)

        return MetricValue(self, metric_name, MetricTypes.ELAPSED_TIME,
                           elapsed_time, plot_x_position)


class AverageEpochTime(AbstractMetric):
    """
    Time usage metric, measured in seconds.

    The time is measured as the average epoch time of a step.
    The average value is computed and emitted at the end of the train/test step.

    By default this metric takes the time of epochs in training steps only. This
    behaviour can be changed by passing test=True in the constructor.

    Consider that, when used on the test set, the epoch time is the same as the
    step time. This means that this metric and the :class:`EpochTime` metric
    will emit the same values when used for the test phase.
    """

    def __init__(self, *, train=True, test=False):
        """
        Creates an instance of the Average Epoch Time metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the time will be taken on training epochs.
            Defaults to True.
        :param test: When True, the time will be taken on test epochs.
            Defaults to False.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')
        self._epoch_start_time = None
        self._accumulated_time = 0.0
        self._n_epochs = 0

        on_step_start_events = filter_accepted_events(
            [OnTrainStepStart, OnTestStepStart], train=train, test=test)

        on_epoch_start_events = filter_accepted_events(
            [OnTrainEpochStart], train=train, test=test)

        on_epoch_end_events = filter_accepted_events(
            [OnTrainEpochEnd], train=train, test=test)

        on_step_end_events = filter_accepted_events(
            [OnTrainStepEnd, OnTestStepEnd], train=train, test=test)

        # Attach callbacks
        self._on(on_step_start_events, self.step_start) \
            ._on(on_epoch_start_events, self.epoch_start) \
            ._on(on_epoch_end_events, self.epoch_end) \
            ._on(on_step_end_events, self.result_emitter)

    def step_start(self, _):
        # Step start
        self._accumulated_time = 0.0
        self._n_epochs = 0
        # Used for timing during the test phase
        self._epoch_start_time = time.perf_counter()

    def epoch_start(self, _):
        # Epoch start (training phase)
        self._epoch_start_time = time.perf_counter()

    def epoch_end(self, _):
        # Epoch end  (training phase)
        self._accumulated_time = time.perf_counter() - self._epoch_start_time
        self._n_epochs += 1

    def result_emitter(self, eval_data):
        eval_data: Union[OnTrainEpochEnd, OnTestStepEnd]
        # Epoch end
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)

        if self._n_epochs == 0:
            # Test phase
            self._n_epochs = 1
            self._accumulated_time = \
                time.perf_counter() - self._epoch_start_time

        average_epoch_time = self._accumulated_time / self._n_epochs

        metric_name = 'Avg_Epoch_Time/{}/Task{:03}'.format(phase_name,
                                                           task_label)
        plot_x_position = self._next_x_position(metric_name)

        return MetricValue(
            self, metric_name, MetricTypes.ELAPSED_TIME,
            average_epoch_time, plot_x_position)


__all__ = [
    'AverageEpochTime',
    'EpochTime'
]
