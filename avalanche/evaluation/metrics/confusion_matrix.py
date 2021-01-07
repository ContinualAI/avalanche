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

from typing import Callable, Union, Optional

import torch
from PIL.Image import Image
from numpy import ndarray
from torch import Tensor
from torch.nn.functional import pad

from avalanche.evaluation import OnTrainStepEnd, OnTestStepEnd, \
    OnTrainEpochEnd, PluginMetric, Metric
from avalanche.evaluation.metric_results import AlternativeValues, MetricTypes, \
    MetricValue
from avalanche.evaluation.abstract_metric import AbstractMetric
from avalanche.evaluation.metric_units import ConfusionMatrixUnit, MetricUnit
from avalanche.evaluation.metric_utils import default_cm_image_creator, \
    filter_accepted_events, get_task_label


class ConfusionMatrix(Metric[Tensor]):

    def __init__(self, num_classes: int = None):
        self._cm_tensor: Optional[Tensor] = None
        self._num_classes: Optional[int] = num_classes

    @torch.no_grad()
    def update(self, true_y: Tensor, predicted_y: Tensor) -> None:
        if len(true_y) != len(predicted_y):
            raise ValueError('Size mismatch for true_y and predicted_y tensors')

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        # Initialize or enlarge the confusion matrix
        max_label = max(torch.max(predicted_y).item(),
                        torch.max(true_y).item())
        if self._num_classes is not None:
            max_label = max(max_label, self._num_classes)

        if self._cm_tensor is None:
            # Create the confusion matrix
            self._cm_tensor = torch.zeros((max_label+1, max_label+1),
                                          dtype=torch.long)
        elif max_label >= self._cm_tensor.shape[0]:
            # Enlarge the confusion matrix
            size_diff = 1 + max_label - self._cm_tensor.shape[0]
            self._cm_tensor = pad(self._cm_tensor,
                                  (0, size_diff, 0, size_diff))

        for pattern_idx in range(len(true_y)):
            self._cm_tensor[true_y[pattern_idx]][predicted_y[pattern_idx]] += 1

    def result(self) -> Tensor:
        if self._cm_tensor is None:
            return torch.zeros(0, dtype=torch.long)
        return self._cm_tensor

    def reset(self) -> None:
        self._cm_tensor = None


class TaskConfusionMatrix(PluginMetric[Tensor, ]):
    """
    The Confusion Matrix metric.

    This matrix logs the Tensor and PIL Image representing the confusion
    matrix after each phase. # TODO: ?

    This metric is computed separately for each task.

    By default this metric computes the matrix on the test set only but this
    behaviour can be changed by passing train=True in the constructor.
    """
    def __init__(self, *,
                 train: bool = False,
                 test: bool = True,
                 num_classes: int = None,  # TODO: or list of ints (1 per task)
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

        # TODO: continue

        # Create CM unit
        self._cm_unit: MetricUnit = ConfusionMatrixUnit(
            num_classes=num_classes, normalize=normalize,
            on_train_epochs=train, on_test_epochs=test)

        on_events = filter_accepted_events(
            [OnTrainStepEnd, OnTestStepEnd], train=train, test=test)

        # Attach callbacks
        self._attach(self._cm_unit)._on(on_events, self.result_emitter)

    def _package_result(self, eval_data):
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


class ConfusionMatrixOld(PluginMetric[Tensor, ]):
    """
    The Confusion Matrix metric.

    This matrix logs the Tensor and PIL Image representing the confusion
    matrix after each epoch. # TODO: ?

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
