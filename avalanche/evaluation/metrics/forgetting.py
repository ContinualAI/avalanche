################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
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
from avalanche.evaluation.metric_utils import get_metric_name

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class StepForgetting(PluginMetric[Dict[int, float]]):
    """
    The Forgetting metric, describing the accuracy loss detected for a
    certain step.

    This metric, computed separately for each step,
    is the difference between the accuracy result obtained after
    first training on a step and the accuracy result obtained
    on the same step at the end of successive steps.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the StepForgetting metric.
        """

        super().__init__()

        self._initial_accuracy: Dict[int, float] = dict()
        """
        The initial accuracy of each step.
        """

        self._current_accuracy: Dict[int, Accuracy] = dict()
        """
        The current accuracy of each step.
        """

        self.eval_step_id = None
        """
        The current evaluation step id
        """

    def reset(self) -> None:
        """
        Resets the metric.

        Beware that this will also reset the initial accuracy of each step!

        :return: None.
        """
        self._initial_accuracy = dict()
        self._current_accuracy = dict()

    def reset_current_accuracy(self) -> None:
        """
        Resets the current accuracy.

        This will preserve the initial accuracy value of each step.
        To be used at the beginning of each eval step.

        :return: None.
        """
        self._current_accuracy = dict()

    def update(self, true_y: Tensor, predicted_y: Tensor, label: int) \
            -> None:
        """
        Updates the running accuracy of a step given the ground truth and
        predicted labels of a minibatch.

        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param predicted_y: The ground truth. Both labels and logit vectors
            are supported.
        :param label: The step label.
        :return: None.
        """
        if label not in self._current_accuracy:
            self._current_accuracy[label] = Accuracy()
        self._current_accuracy[label].update(true_y, predicted_y)

    def before_eval(self, strategy) -> None:
        self.reset_current_accuracy()

    def before_eval_step(self, strategy: 'PluggableStrategy') -> None:
        self.eval_step_id = strategy.eval_step_id

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        label = strategy.eval_step_id
        self.update(strategy.mb_y,
                    strategy.logits,
                    label)

    def after_eval_step(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        # eval step never encountered during training
        # or eval step is the current training step
        # forgetting not reported in both cases
        if self.eval_step_id not in self._initial_accuracy:
            train_label = strategy.training_step_counter
            # the test accuracy on the training step we have just
            # trained on. This is the initial accuracy.
            if train_label not in self._initial_accuracy:
                self._initial_accuracy[train_label] = \
                    self._current_accuracy[train_label].result()
            return None

        # eval step previously encountered during training
        # which is not the most recent training step
        # return forgetting
        return self._package_result(strategy)

    def result(self) -> float:
        """
        Return the amount of forgetting for the eval step
        associated to `eval_label`.

        The forgetting is computed as the accuracy difference between the
        initial step accuracy (when first encountered
        in the training stream) and the current accuracy.
        A positive value means that forgetting occurred. A negative value
        means that the accuracy on that step increased.

        :param eval_label: integer label describing the eval step
                of which measuring the forgetting
        :return: The amount of forgetting on `eval_step` step
                 (as float in range [-1, 1]).
        """

        prev_accuracy: float = self._initial_accuracy[self.eval_step_id]
        accuracy: Accuracy = self._current_accuracy[self.eval_step_id]
        forgetting = prev_accuracy - accuracy.result()
        return forgetting

    def _package_result(self, strategy: 'PluggableStrategy') \
            -> MetricResult:

        forgetting = self.result()
        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        metric_values = [MetricValue(
            self, metric_name, forgetting, plot_x_position)]
        return metric_values

    def __str__(self):
        return "StepForgetting"


__all__ = ['StepForgetting']
