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

from typing import Dict, TYPE_CHECKING, Union

from avalanche.evaluation.metric_definitions import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metrics import Accuracy
from avalanche.evaluation.metric_utils import get_metric_name

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class Forgetting(Metric[Union[float, None, Dict[int, float]]]):
    """
    The Forgetting metric.
    This metric returns the forgetting relative to a specific key.
    Alternatively, this metric returns a dict in which each key is associated
    to the forgetting.
    Forgetting is computed as the difference between the first value recorded
    for a specific key and the last value recorded for that key.
    The value associated to a key can be update with the `update` method.

    At initialization, this metric returns an empty dictionary.
    """

    def __init__(self):
        """
        Creates an instance of the Forgetting metric
        """

        super().__init__()

        self.initial: Dict[int, float] = dict()
        """
        The initial value for each key.
        """

        self.last: Dict[int, float] = dict()
        """
        The last value detected for each key
        """

    def update_initial(self, k, v):
        self.initial[k] = v

    def update_last(self, k, v):
        self.last[k] = v

    def update(self, k, v, initial=False):
        if initial:
            self.update_initial(k, v)
        else:
            self.update_last(k, v)

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        Forgetting is returned only for keys encountered twice.

        :param k: the key for which returning forgetting. If k has not
            updated at least twice it returns None. If k is None,
            forgetting will be returned for all keys encountered at least
            twice.

        :return: the difference between the first and last value encountered
            for k, if k is not None. It returns None if k has not been updated
            at least twice. If k is None, returns a dictionary
            containing keys whose value has been updated at least twice. The
            associated value is the difference between the first and last
            value recorded for that key.
        """

        forgetting = {}
        if k is not None:
            if k in self.initial and k in self.last:
                return self.initial[k] - self.last[k]
            else:
                return None

        ik = set(self.initial.keys())
        both_keys = list(ik.intersection(set(self.last.keys())))

        for k in both_keys:
            forgetting[k] = self.initial[k] - self.last[k]

        return forgetting

    def reset_last(self) -> None:
        self.last: Dict[int, float] = dict()

    def reset(self) -> None:
        self.initial: Dict[int, float] = dict()
        self.last: Dict[int, float] = dict()


class ExperienceForgetting(PluginMetric[Dict[int, float]]):
    """
    The ExperienceForgetting metric, describing the accuracy loss
    detected for a certain experience.

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

        self.forgetting = Forgetting()
        """
        The general metric to compute forgetting
        """

        self._last_accuracy = Accuracy()
        """
        The average accuracy over the current evaluation experience
        """

        self.eval_exp_id = None
        """
        The current evaluation experience id
        """

        self.train_exp_id = None
        """
        The last encountered training experience id
        """

    def reset(self) -> None:
        """
        Resets the metric.

        Beware that this will also reset the initial accuracy of each
        experience!

        :return: None.
        """
        self.forgetting.reset()

    def reset_last_accuracy(self) -> None:
        """
        Resets the last accuracy.

        This will preserve the initial accuracy value of each experience.
        To be used at the beginning of each eval experience.

        :return: None.
        """
        self.forgetting.reset_last()

    def update(self, k, v, initial=False):
        """
        Update forgetting metric.
        See `Forgetting` for more detailed information.

        :param k: key to update
        :param v: value associated to k
        :param initial: update initial value. If False, update
            last value.
        """
        self.forgetting.update(k, v, initial=initial)

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        See `Forgetting` documentation for more detailed information.

        k: optional key from which compute forgetting.
        """
        return self.forgetting.result(k=k)

    def before_training_exp(self, strategy: 'PluggableStrategy') -> None:
        self.train_exp_id = strategy.experience.current_experience

    def before_eval(self, strategy) -> None:
        self.reset_last_accuracy()

    def before_eval_exp(self, strategy: 'PluggableStrategy') -> None:
        self._last_accuracy.reset()

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        self.eval_exp_id = strategy.experience.current_experience
        self._last_accuracy.update(strategy.mb_y,
                                   strategy.logits)

    def after_eval_exp(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        # update experience on which training just ended
        if self.train_exp_id == self.eval_exp_id:
            self.update(self.eval_exp_id,
                        self._last_accuracy.result(),
                        initial=True)
        else:
            # update other experiences
            # if experience has not been encountered in training
            # its value will not be considered in forgetting
            self.update(self.eval_exp_id,
                        self._last_accuracy.result())

        # this checks if the evaluation experience has been
        # already encountered at training time
        # before the last training.
        # If not, forgetting should not be returned.
        if self.result(k=self.eval_exp_id) is not None:
            return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') \
            -> MetricResult:

        forgetting = self.result(k=self.eval_exp_id)
        metric_name = get_metric_name(self, strategy, add_experience=True)
        plot_x_position = self._next_x_position(metric_name)

        metric_values = [MetricValue(
            self, metric_name, forgetting, plot_x_position)]
        return metric_values

    def __str__(self):
        return "ExperienceForgetting"


__all__ = [
    'Forgetting',
    'ExperienceForgetting'
]
