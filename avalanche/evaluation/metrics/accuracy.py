#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from collections import defaultdict
from typing import Dict, TYPE_CHECKING, List

import torch
from torch import Tensor

from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics.mean import Mean

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class Accuracy(Metric[float]):
    """
    The accuracy metric.

    Instances of this metric compute the average accuracy by receiving a pair
    of "ground truth" and "prediction" Tensors describing the labels of a
    minibatch. Those two tensors can both contain plain labels or
    one-hot/logit vectors.

    The result is the running accuracy computed as the number of correct
    patterns divided by the overall amount of patterns.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an accuracy value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the accuracy metric.

        By default this metric in its initial state will return an accuracy
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """

        self._mean_accuracy = Mean()
        """
        The mean utility that will be used to store the running accuracy.
        """

    @torch.no_grad()
    def update(self, true_y: Tensor, predicted_y: Tensor) -> None:
        """
        Update the running accuracy given the true and predicted labels.

        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param predicted_y: The ground truth. Both labels and logit vectors
            are supported.
        :return: None.
        """
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

        true_positives = float(torch.sum(torch.eq(predicted_y, true_y)))
        total_patterns = len(true_y)

        self._mean_accuracy.update(true_positives / total_patterns,
                                   total_patterns)

    def result(self) -> float:
        """
        Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        :return: The running accuracy, as a float value between 0 and 1.
        """
        return self._mean_accuracy.result()

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._mean_accuracy.reset()


class MinibatchAccuracy(PluginMetric[float]):
    """
    The minibatch accuracy metric.

    This metric "logs" the accuracy value after each iteration. Beware that this
    metric will not average the accuracy across minibatches!

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracy` and/or :class:`TaskAccuracy` instead.
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the MinibatchAccuracy metric.

        The train and test parameters are used to control if this metric should
        compute and log values referred to the train phase, test phase or both.
        At least one of them must be True!

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             ' time.')
        self._minibatch_accuracy = Accuracy()
        self._compute_train_accuracy = train
        self._compute_test_accuracy = test

    def result(self) -> float:
        return self._minibatch_accuracy.result()

    def reset(self) -> None:
        self._minibatch_accuracy.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._compute_train_accuracy:
            return self._on_iteration(strategy)

    def after_test_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._compute_test_accuracy:
            return self._on_iteration(strategy)

    def _on_iteration(self, strategy: 'PluggableStrategy') -> MetricResult:
        self.reset()  # Because this metric computes the accuracy of a single mb
        self._minibatch_accuracy.update(strategy.mb_y,
                                        strategy.logits)
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'Top1_Acc_Minibatch/{}/Task{:03}'.format(phase_name,
                                                               task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class EpochAccuracy(PluginMetric[float]):
    """
    The average epoch accuracy metric.

    The accuracy will be logged after each epoch by computing the accuracy
    as the number of correctly predicted patterns divided by the overall
    number of patterns encountered in that epoch, which means that having
    unbalanced minibatch sizes will not affect the metric.
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the EpochAccuracy metric.

        The train and test parameters are used to control if this metric should
        compute and log values referred to the train phase, test phase or both.
        At least one of them must be True!

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to False.
        """
        super().__init__()

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             ' time.')

        self._accuracy_metric = Accuracy()
        self._compute_train_accuracy = train
        self._compute_test_accuracy = test

    def reset(self) -> None:
        self._accuracy_metric.reset()

    def result(self) -> float:
        return self._accuracy_metric.result()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        if self._compute_train_accuracy:
            self._accuracy_metric.update(strategy.mb_y,
                                         strategy.logits)

    def after_test_iteration(self, strategy: 'PluggableStrategy') -> None:
        if self._compute_test_accuracy:
            self._accuracy_metric.update(strategy.mb_y,
                                         strategy.logits)

    def before_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        if self._compute_train_accuracy:
            self.reset()

    def before_test_step(self, strategy: 'PluggableStrategy') -> None:
        if self._compute_test_accuracy:
            self.reset()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        if self._compute_train_accuracy:
            return self._package_result(strategy)

    def after_test_step(self, strategy: 'PluggableStrategy') -> MetricResult:
        if self._compute_test_accuracy:
            return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'Top1_Acc_Epoch/{}/Task{:03}'.format(phase_name,
                                                           task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class RunningEpochAccuracy(EpochAccuracy):
    """
    The running average accuracy metric.

    This metric behaves like :class:`EpochAccuracy` but, differently from it,
    this metric will log the running accuracy value after each iteration.
    """

    def __init__(self, *, train=True, test=True):
        """
        Creates an instance of the RunningEpochAccuracy metric.

        The train and test parameters are used to control if this metric should
        compute and log values referred to the train phase, test phase or both.
        At least one of them must be True!

        Beware that the test parameter defaults to False because logging
        the running test accuracy it's and uncommon practice.

        :param train: When True, the metric will be computed on the training
            phase. Defaults to True.
        :param test: When True, the metric will be computed on the test
            phase. Defaults to False.
        """
        super().__init__(train=train, test=test)

        if not train and not test:
            raise ValueError('train and test can\'t be both False at the same'
                             ' time.')

        self._compute_train_accuracy = train
        self._compute_test_accuracy = test

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        super().after_training_iteration(strategy)
        if self._compute_train_accuracy:
            return self._package_result(strategy)

    def after_test_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        super().after_test_iteration(strategy)
        if self._compute_test_accuracy:
            return self._package_result(strategy)

    def after_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        # Overrides the method from EpochAccuracy so that it doesn't
        # emit a metric value on epoch end!
        return None

    def after_test_step(self, strategy: 'PluggableStrategy') -> None:
        # Overrides the method from EpochAccuracy so that it doesn't
        # emit a metric value on epoch end!
        return None

    def _package_result(self, strategy: 'PluggableStrategy'):
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'Top1_Acc_Running/{}/Task{:03}'.format(phase_name,
                                                             task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class TaskAccuracy(PluginMetric[Dict[int, float]]):
    """
    The task accuracy metric.

    This is the most common metric used in the evaluation of a Continual
    Learning algorithm.

    Can be safely used when evaluation task-free scenarios, in which case the
    default task label "0" will be used.

    The task accuracies will be logged at the end of the test phase. This metric
    doesn't apply to the training phase.
    """

    def __init__(self):
        """
        Creates an instance of the TaskAccuracy metric.
        """
        super().__init__()

        self._task_accuracy: Dict[int, Accuracy] = defaultdict(Accuracy)
        """
        A dictionary used to store the accuracy for each task.
        """

    def reset(self) -> None:
        self._task_accuracy = defaultdict(Accuracy)

    def result(self) -> Dict[int, float]:
        result_dict = dict()
        for task_id in self._task_accuracy:
            result_dict[task_id] = self._task_accuracy[task_id].result()
        return result_dict

    def update(self, true_y: Tensor, predicted_y: Tensor, task_label: int) \
            -> None:
        self._task_accuracy[task_label].update(true_y, predicted_y)

    def before_test(self, strategy) -> None:
        self.reset()

    def after_test_iteration(self, strategy: 'PluggableStrategy') -> None:
        self.update(strategy.mb_y, strategy.logits, strategy.test_task_label)

    def after_test(self, strategy) -> MetricResult:
        return self._package_result()

    def _package_result(self) -> MetricResult:
        metric_values = []
        for task_label, task_accuracy in self.result().items():
            metric_name = 'Top1_Acc_Task/Task{:03}'.format(task_label)
            plot_x_position = self._next_x_position(metric_name)

            metric_values.append(MetricValue(
                self, metric_name, task_accuracy, plot_x_position))
        return metric_values


def accuracy_metrics(*, minibatch=False, epoch=False, epoch_running=False,
                     task=False, train=None, test=None) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of metric.

    :param minibatch: If True, will return a metric able to log the minibatch
        accuracy.
    :param epoch: If True, will return a metric able to log the epoch accuracy.
    :param epoch_running: If True, will return a metric able to log the running
        epoch accuracy.
    :param task: If True, will return a metric able to log the task accuracy.
        This metric applies to the test flow only. If the `test` parameter is
        False, an error will be raised.
    :param train: If True, metrics will log values for the train flow. Defaults
        to None, which means that the per-metric default value will be used.
    :param test: If True, metrics will log values for the test flow. Defaults
        to None, which means that the per-metric default value will be used.

    :return: A list of plugin metrics.
    """

    if (train is not None and not train) and (test is not None and not test):
        raise ValueError('train and test can\'t be both False at the same'
                         ' time.')
    if task and test is not None and not test:
        raise ValueError('The task accuracy metric only applies to the test '
                         'phase.')

    train_test_flags = dict()
    if train is not None:
        train_test_flags['train'] = train

    if test is not None:
        train_test_flags['test'] = test

    metrics = []
    if minibatch:
        metrics.append(MinibatchAccuracy(**train_test_flags))

    if epoch:
        metrics.append(EpochAccuracy(**train_test_flags))

    if epoch_running:
        metrics.append(RunningEpochAccuracy(**train_test_flags))

    if task:
        metrics.append(TaskAccuracy())

    return metrics


__all__ = [
    'Accuracy',
    'MinibatchAccuracy',
    'EpochAccuracy',
    'RunningEpochAccuracy',
    'TaskAccuracy',
    'accuracy_metrics'
]
