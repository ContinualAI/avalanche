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

from typing import Dict, Set, TYPE_CHECKING

from torch import Tensor

from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metrics import Accuracy
if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class Forgetting(PluginMetric[Dict[int, float]]):
    """
    The Forgetting metric, describing the accuracy loss detected for a
    certain task or step.

    This metric, computed separately for each task/step
    is the difference between the accuracy result obtained after
    first training on a task/step and the accuracy result obtained
    on the same task/step at the end of successive steps.

    This metric is computed during the eval phase only.
    """

    def __init__(self, compute_for_step=False):
        """
        Creates an instance of the Forgetting metric.

        :param compute_for_step: if True, compute the metric at a step level.
            If False, compute the metric at task level. Default to False.
        """

        super().__init__()

        self.compute_for_step = compute_for_step

        self._initial_accuracy: Dict[int, float] = dict()
        """
        The initial accuracy of each task/step.
        """

        self._current_accuracy: Dict[int, Accuracy] = dict()
        """
        The current accuracy of each task/step.
        """

    def reset(self) -> None:
        """
        Resets the metric.

        Beware that this will also reset the initial accuracy of each task/step!

        :return: None.
        """
        self._initial_accuracy = dict()
        self._current_accuracy = dict()

    def reset_current_accuracy(self) -> None:
        """
        Resets the current accuracy.

        This will preserve the initial accuracy value of each task/step.
        To be used at the beginning of each eval step.

        :return: None.
        """
        self._current_accuracy = dict()

    def update(self, true_y: Tensor, predicted_y: Tensor, label: int) \
            -> None:
        """
        Updates the running accuracy of a task/step given the ground truth and
        predicted labels of a minibatch.

        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param predicted_y: The ground truth. Both labels and logit vectors
            are supported.
        :param label: The task or step label.
        :return: None.
        """
        if label not in self._current_accuracy:
            self._current_accuracy[label] = Accuracy()
        self._current_accuracy[label].update(true_y, predicted_y)

    def before_eval(self, strategy) -> None:
        self.reset_current_accuracy()

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        label = strategy.eval_step_id if self.compute_for_step \
                else strategy.eval_task_label
        self.update(strategy.mb_y,
                    strategy.logits,
                    label)

    def after_eval(self, strategy: 'PluggableStrategy') -> MetricResult:
        label = strategy.training_step_counter if self.compute_for_step \
                else strategy.train_task_label
        return self._package_result(label)

    def result(self) -> Dict[int, float]:
        """
        Return the amount of forgetting for each task/step.

        The forgetting is computed as the accuracy difference between the
        initial task/step accuracy (when first encountered
        in the training stream) and the current accuracy.
        A positive value means that forgetting occurred. A negative value
        means that the accuracy on that task/step increased.

        :return: A dictionary in which keys are task/step labels and the
                 values are the forgetting measures
                 (as floats in range [-1, 1]).
        """
        prev_accuracies: Dict[int, float] = self._initial_accuracy
        accuracies: Dict[int, Accuracy] = self._current_accuracy
        all_labels: Set[int] = set(prev_accuracies.keys()) \
            .union(set(accuracies.keys()))
        forgetting: Dict[int, float] = dict()
        for label in all_labels:
            delta = 0.0
            if (label in accuracies) and \
                    (label in self._initial_accuracy):
                # Task / step already encountered in previous phases
                delta = self._initial_accuracy[label] - \
                        accuracies[label].result()
            # Other situations:
            # - A task/step that was not encountered before (forgetting == 0)
            # - A task/step that was encountered before, but has not been
            # encountered in the current eval phase (forgetting == N.A. == 0)
            forgetting[label] = delta
        return forgetting

    def _package_result(self, label: int) -> MetricResult:

        # The forgetting value is computed as the difference between the
        # accuracy obtained after training for the first time and the current
        # accuracy. Here we store the initial accuracy.
        if label not in self._initial_accuracy and \
                label in self._current_accuracy:
            initial_accuracy = self._current_accuracy[label].\
                result()
            self._initial_accuracy[label] = initial_accuracy

        metric_values = []
        string_print = 'Step' if self.compute_for_step else 'Task'
        for label, forgetting in self.result().items():
            metric_name = '{}_Forgetting/{}{:03}'.format(
                string_print, string_print, label)
            plot_x_position = self._next_x_position(metric_name)

            metric_values.append(MetricValue(
                self, metric_name, forgetting, plot_x_position))
        return metric_values


__all__ = ['Forgetting']
