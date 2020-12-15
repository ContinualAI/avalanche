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

from typing import Callable, Union

import torch
from PIL.Image import Image
from numpy import ndarray

from avalanche.evaluation import OnTrainStepEnd, OnTestStepEnd, \
    OnTrainEpochEnd, MetricValue, MetricTypes, AlternativeValues
from avalanche.evaluation.abstract_metric import AbstractMetric
from avalanche.evaluation.metric_units import ConfusionMatrixUnit, MetricUnit
from avalanche.evaluation.metric_utils import default_cm_image_creator, \
    filter_accepted_events, get_task_label


class ConfusionMatrix(AbstractMetric):
    """
    The Confusion Matrix metric.

    This matrix logs the Tensor and PIL Image representing the confusion
    matrix after each epoch.

    This metric is computed separately for each task.

    By default this metric computes the matrix on the test set only but this
    behaviour can be changed by passing train=True in the constructor.
    """
    def __init__(self, *,
                 train: bool = False,
                 test: bool = True,
                 num_classes: int = None,
                 normalize: str = None,
                 save_image: bool = True,
                 image_creator: Callable[[ndarray], Image] =
                 default_cm_image_creator):
        """
        Creates an instance of the Confusion Matrix metric.

        The train and test parameters can be True at the same time. However,
        at least one of them must be True.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to False.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to True.
        :param num_classes: When not None, is used to properly define the
            amount of rows/columns in the confusion matrix. When None, the
            matrix will have many rows/columns as the maximum value of the
            predicted and true pattern labels. Defaults to None.
        :param normalize: Normalizes confusion matrix over the true (rows),
            predicted (columns) conditions or all the population. If None,
            confusion matrix will not be normalized. Valid values are: 'true',
            'pred' and 'all'.
        :param save_image: If True, a graphical representation of the confusion
            matrix will be logged, too. If False, only the Tensor representation
            will be logged. Defaults to True.
        :param image_creator: A callable that, given the tensor representation
            of the confusion matrix, returns a graphical representation of the
            matrix as a PIL Image. Defaults to `default_cm_image_creator`.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             'time.')

        self._save_image = save_image

        if image_creator is None:
            image_creator = default_cm_image_creator
        self._image_creator = image_creator

        # Create CM unit
        self._cm_unit: MetricUnit = ConfusionMatrixUnit(
            num_classes=num_classes, normalize=normalize,
            on_train_epochs=train, on_test_epochs=test)

        on_events = filter_accepted_events(
            [OnTrainStepEnd, OnTestStepEnd], train=train, test=test)

        # Attach callbacks
        self._attach(self._cm_unit)._on(on_events, self.result_emitter)

    def result_emitter(self, eval_data):
        eval_data: Union[OnTrainEpochEnd, OnTestStepEnd]
        phase_name = 'Test' if eval_data.test_phase else 'Train'
        task_label = get_task_label(eval_data)
        metric_value = self._cm_unit.value

        metric_name = 'ConfusionMatrix/{}/Task{:03}'.format(phase_name,
                                                            task_label)
        plot_x_position = self._next_x_position(metric_name)

        metric_representation = MetricValue(
            self, metric_name, MetricTypes.CONFUSION_MATRIX,
            torch.as_tensor(metric_value), plot_x_position)

        if self._save_image:
            cm_image = self._image_creator(metric_value)
            metric_representation = MetricValue(
                self, metric_name, MetricTypes.CONFUSION_MATRIX,
                AlternativeValues(cm_image, metric_value), plot_x_position)

        return metric_representation


__all__ = ['ConfusionMatrix']
