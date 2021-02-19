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


class RunningAverageAccuracy(Metric[float]):
    """
    The Running Average metric. This is a general metric
    used to compute more specific ones.

    Instances of this metric keeps the running average accuracy
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average accuracy
    across all predictions made since the last `reset`.
    \frac{1}{N} * \sum_{i=1}^N a_i,
    where N is the number of predictions made seen since last `reset` and
    a_i is 1 if the i-th prediction was correct,
    0 otherwise.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an accuracy value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the Running Average RunningAverageAccuracy metric.

        By default this metric in its initial state will return an accuracy
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """

        self._mean_accuracy = Mean()
        """
        The mean utility that will be used to store the running accuracy.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor) -> None:
        """
        Update the running accuracy given the true and predicted labels.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
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


class SingleMinibatchAccuracy(PluginMetric[float]):
    """
    The minibatch accuracy metric. This metric only works at training time.

    This metric computes the average accuracy over patterns from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`TrainEpochAccuracy` instead.
    """

    def __init__(self):
        """
        Creates an instance of the TrainMinibatchAccuracy metric.
        """

        super().__init__()

        self._minibatch_accuracy = RunningAverageAccuracy()

    def result(self) -> float:
        return self._minibatch_accuracy.result()

    def reset(self) -> None:
        self._minibatch_accuracy.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
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


class SingleEpochAccuracy(PluginMetric[float]):
    """
    The average accuracy over a single training epoch.
    This metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the TrainEpochAccuracy metric.
        """
        super().__init__()

        self._accuracy_metric = RunningAverageAccuracy()

    def reset(self) -> None:
        self._accuracy_metric.reset()

    def result(self) -> float:
        return self._accuracy_metric.result()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        self._accuracy_metric.update(strategy.mb_y,
                                     strategy.logits)

    def before_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'Top1_Acc_TrainEpoch/{}/Task{:03}'.format(phase_name,
                                                           task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class RunningMinibatchAccuracy(SingleEpochAccuracy):
    """
    The average accuracy across all minibatches up to the current
    epoch iteration.
    This metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningMinibatchAccuracy metric.
        """

        super().__init__()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        super().after_training_iteration(strategy)
        return self._package_result(strategy)

    def after_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        # Overrides the method from EpochAccuracy so that it doesn't
        # emit a metric value on epoch end!
        return None

    def _package_result(self, strategy: 'PluggableStrategy'):
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'Top1_Acc_WithinEpoch/{}/Task{:03}'.format(phase_name,
                                                             task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class StepAccuracy(PluginMetric[float]):
    """
    At the end of each step, this metric reports the average accuracy over all patterns seen
    in that step.
    The metric can emit results during train and evaluation phases or only during
     one of the two, depending on the user choice.
    """

    def __init__(self, *, train=True, eval=True):
        """
        At least one of `train` and `eval` must be true.

        :param train: if True, reports the accuracy after each training step
        :param eval: if True, reports the accuracy after each evaluation step
        """
        super().__init__()
        if not train and not eval:
            raise ValueError(
                'StepAccuracy cannot have both train and'
                ' eval parameters set to False')

        self.train_mode = train
        self.eval_mode = eval
        self._accuracy_metric = RunningAverageAccuracy()

    def reset(self) -> None:
        self._accuracy_metric.reset()

    def result(self) -> float:
        return self._accuracy_metric.result()

    def before_training_step(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        self._accuracy_metric.update(strategy.mb_y,
                                     strategy.logits)

    def after_training_step(self, strategy: 'PluggableStrategy') -> \
            'MetricResult':
        if self.train_mode:
            return self._package_result(strategy)

    def before_eval_step(self, strategy: 'PluggableStrategy') -> None:
        self.reset()

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        self._accuracy_metric.update(strategy.mb_y,
                                     strategy.logits)

    def after_eval_step(self, strategy: 'PluggableStrategy') -> \
            'MetricResult':
        if self.eval_mode:
            return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> \
            MetricResult:
        phase_name, task_label = phase_and_task(strategy)
        metric_value = self.result()

        metric_name = 'Top1_Acc_Step/{}/Task{:03}'.format(phase_name,
                                                           task_label)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]


class EvalTaskAccuracy(PluginMetric[Dict[int, float]]):
    """
    The task accuracy metric.
    This metric works only on evaluation phase.

    This metric computes the average accuracy for each task in the evaluation stream.
    It returns a dictionary mapping each task label to the corresponding
    average accuracy.
    Patterns belonging to a task are not required to be presented in a single evaluation step,
    but they may be distributed over different evaluation steps.

    Can be safely used when evaluation task-free scenarios, in which case the
    default task label "0" will be used.

    The task accuracies will be logged at the end of the eval phase.
    """

    def __init__(self):
        """
        Creates an instance of the EvalTaskAccuracy metric.
        """
        super().__init__()

        self._task_accuracy: Dict[int, RunningAverageAccuracy] = defaultdict(RunningAverageAccuracy)
        """
        A dictionary used to store the accuracy for each task.
        """

    def reset(self) -> None:
        self._task_accuracy = defaultdict(RunningAverageAccuracy)

    def result(self) -> Dict[int, float]:
        result_dict = dict()
        for task_id in self._task_accuracy:
            result_dict[task_id] = self._task_accuracy[task_id].result()
        return result_dict

    def update(self, true_y: Tensor, predicted_y: Tensor, task_label: int) \
            -> None:
        self._task_accuracy[task_label].update(true_y, predicted_y)

    def before_eval(self, strategy) -> None:
        self.reset()

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        self.update(strategy.mb_y, strategy.logits, strategy.eval_task_label)

    def after_eval(self, strategy) -> MetricResult:
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
                     step=False, task=False, train=None, eval=None) -> \
        List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of metric.

    :param minibatch: If True, will return a metric able to log the train minibatch
        accuracy.
    :param epoch: If True, will return a metric able to log the train epoch accuracy.
    :param epoch_running: If True, will return a metric able to log the running
        epoch accuracy.
    :param step: If True, will return a metric able to log the accuracy on each step.
    :param task: If True, will return a metric able to log the task accuracy.
        This metric applies to the eval flow only.
    :param train: If True, metrics will log values for the train flow. Defaults
        to None, which means that the per-metric default value will be used.
    :param eval: If True, metrics will log values for the eval flow. Defaults
        to None, which means that the per-metric default value will be used.

    :return: A list of plugin metrics.
    """

    if (train is not None and not train) and (eval is not None and not eval):
        raise ValueError('train and eval can\'t be both False at the same'
                         ' time.')
    if step and eval is not None and not eval:
        raise ValueError('The task accuracy metric only applies to the eval '
                         'phase.')

    train_eval_flags = dict()
    if train is not None:
        train_eval_flags['train'] = train

    if eval is not None:
        train_eval_flags['eval'] = eval

    metrics = []
    if minibatch:
        metrics.append(SingleMinibatchAccuracy())

    if epoch:
        metrics.append(SingleEpochAccuracy())

    if epoch_running:
        metrics.append(RunningMinibatchAccuracy())

    if step:
        metrics.append(StepAccuracy(**train_eval_flags))

    if task:
        metrics.append(EvalTaskAccuracy())

    return metrics


__all__ = [
    'RunningAverageAccuracy',
    'SingleMinibatchAccuracy',
    'SingleEpochAccuracy',
    'RunningMinibatchAccuracy',
    'StepAccuracy',
    'EvalTaskAccuracy',
    'accuracy_metrics'
]
