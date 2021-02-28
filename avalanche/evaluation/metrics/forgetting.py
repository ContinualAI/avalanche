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


class ExperienceForgetting(PluginMetric[Dict[int, float]]):
    """
    The Forgetting metric, describing the accuracy loss detected for a
    certain experience.

    This metric, computed separately for each experience,
    is the difference between the accuracy result obtained after
    first training on a experience and the accuracy result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the ExperienceForgetting metric.
        """

        super().__init__()

        self._initial_accuracy: Dict[int, float] = dict()
        """
        The initial accuracy of each experience.
        """

        self._current_accuracy: Dict[int, Accuracy] = dict()
        """
        The current accuracy of each experience.
        """

        self.eval_exp_id = None
        """
        The current evaluation experience id
        """

    def reset(self) -> None:
        """
        Resets the metric.

        Beware that this will also reset the initial accuracy of each
        experience!

        :return: None.
        """
        self._initial_accuracy = dict()
        self._current_accuracy = dict()

    def reset_current_accuracy(self) -> None:
        """
        Resets the current accuracy.

        This will preserve the initial accuracy value of each experience.
        To be used at the beginning of each eval experience.

        :return: None.
        """
        self._current_accuracy = dict()

    def update(self, true_y: Tensor, predicted_y: Tensor, label: int) \
            -> None:
        """
        Updates the running accuracy of a experience given the ground truth and
        predicted labels of a minibatch.

        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param predicted_y: The ground truth. Both labels and logit vectors
            are supported.
        :param label: The experience label.
        :return: None.
        """
        if label not in self._current_accuracy:
            self._current_accuracy[label] = Accuracy()
        self._current_accuracy[label].update(true_y, predicted_y)

    def before_eval(self, strategy) -> None:
        self.reset_current_accuracy()

    def before_eval_exp(self, strategy: 'PluggableStrategy') -> None:
        self.eval_exp_id = strategy.eval_exp_id

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        label = strategy.eval_exp_id
        self.update(strategy.mb_y,
                    strategy.logits,
                    label)

    def after_eval_exp(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        # eval experience never encountered during training
        # or eval experience is the current training experience
        # forgetting not reported in both cases
        if self.eval_exp_id not in self._initial_accuracy:
            train_label = strategy.training_exp_counter
            # the test accuracy on the training experience we have just
            # trained on. This is the initial accuracy.
            if train_label not in self._initial_accuracy:
                self._initial_accuracy[train_label] = \
                    self._current_accuracy[train_label].result()
            return None

        # eval experience previously encountered during training
        # which is not the most recent training experience
        # return forgetting
        return self._package_result(strategy)

    def result(self) -> float:
        """
        Return the amount of forgetting for the eval experience
        associated to `eval_label`.

        The forgetting is computed as the accuracy difference between the
        initial experience accuracy (when first encountered
        in the training stream) and the current accuracy.
        A positive value means that forgetting occurred. A negative value
        means that the accuracy on that experience increased.

        :param eval_label: integer label describing the eval experience
                of which measuring the forgetting
        :return: The amount of forgetting on `eval_exp` experience
                 (as float in range [-1, 1]).
        """

        prev_accuracy: float = self._initial_accuracy[self.eval_exp_id]
        accuracy: Accuracy = self._current_accuracy[self.eval_exp_id]
        forgetting = prev_accuracy - accuracy.result()
        return forgetting

    def _package_result(self, strategy: 'PluggableStrategy') \
            -> MetricResult:

        forgetting = self.result()
        metric_name = get_metric_name(self, strategy, add_experience=True)
        plot_x_position = self._next_x_position(metric_name)

        metric_values = [MetricValue(
            self, metric_name, forgetting, plot_x_position)]
        return metric_values

    def __str__(self):
        return "ExperienceForgetting"


__all__ = ['ExperienceForgetting']
