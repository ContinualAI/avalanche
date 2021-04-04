################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 4-02-2021                                                              #
# Author(s): Ryan Lindeborg                                                    #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import Dict, Union

from avalanche.evaluation.metric_definitions import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metrics import Accuracy, Mean
from avalanche.evaluation.metric_utils import get_metric_name, \
    phase_and_task, stream_type

class ForwardTransfer(Metric[Union[float, None, Dict[int, float]]]):
    """
        The standalone Forward Transfer metric.
        This metric returns the forward transfer relative to a specific key.
        Alternatively, this metric returns a dict in which each key is associated
        to the forward transfer.
        Forward transfer is computed as the difference between the value recorded for a specific key after the previous experience has been trained on, and random initialization before training
        The value associated to a key can be updated with the `update` method.

        At initialization, this metric returns an empty dictionary.
        """

    def __init__(self):
        """
        Creates an instance of the standalone Forward Transfer metric
        """

        super().__init__()

        self.initial: Dict[int, float] = dict()
        """
        The initial value for each key. This is the accuracy at random initialization.
        """

        self.previous: Dict[int, float] = dict()
        """
        The previous experience value detected for each key
        """

    def update_initial(self, k, v):
        self.initial[k] = v

    def update_previous(self, k, v):
        self.previous[k] = v

    def update(self, k, v, initial=False):
        if initial:
            self.update_initial(k, v)
        else:
            self.update_previous(k, v)

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        :param k: the key for which returning forward transfer. If k is None,
            forward transfer will be returned for all keys where the previous experience has been trained on.

        :return: the difference between the key value after training on the previous experience,
            and the key at random initialization.
        """

        forward_transfer = {}
        if k is not None:
            if k in self.previous:
                return self.previous[k] - self.initial[k]
            else:
                return None

        previous_keys = set(self.previous.keys())
        for k in previous_keys:
            forward_transfer[k] = self.previous[k] - self.initial[k]

        return forward_transfer

    def reset(self) -> None:
        # Resets previous and initial accuracy dictionaries
        #TODO: Need to implement random initialization accuracy evaluation into self.initial
        self.initial: Dict[int, float] = dict()
        self.previous: Dict[int, float] = dict()

class ExperienceForwardTransfer(PluginMetric[Dict[int, float]]):
    """
    The ExperienceForwardMetric metric, describing the forward transfer
    detected after a certain experience.

    This plugin metric, computed separately for each experience,
    is the difference between the accuracy result obtained after the previous experience
    and at random initialization.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the ExperienceForwardTransfer metric.
        """

        super().__init__()

        self.forward_transfer = ForwardTransfer()
        """
        The general metric to compute forward transfer
        """

        self._current_accuracy = Accuracy()
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

        Note that this will reset the previous and initial accuracy of each
        experience.

        :return: None.
        """
        self.forward_transfer.reset()

    def update(self, k, v, initial=False):
        """
        Update forward transfer metric.
        See `ForwardTransfer` for more detailed information.

        :param k: key to update
        :param v: value associated to k
        :param initial: update initial value. If False, update
            previous value.
        """
        self.forward_transfer.update(k, v, initial=initial)

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        Result for experience defined by a key.
        See `ForwardTransfer` documentation for more detailed information.

        k: optional key from which to compute forward transfer.
        """
        return self.forward_transfer.result(k=k)

    def before_training_exp(self, strategy: 'BaseStrategy') -> None:
        self.train_exp_id = strategy.experience.current_experience

    def before_eval_exp(self, strategy: 'BaseStrategy') -> None:
        self._current_accuracy.reset()

    def after_eval_iteration(self, strategy: 'BaseStrategy') -> None:
        self.eval_exp_id = strategy.experience.current_experience
        self._current_accuracy.update(strategy.mb_y,
                                      strategy.logits)

    def after_eval_exp(self, strategy: 'BaseStrategy') \
            -> MetricResult:
        # Only if eval experience is one after the train experience, do we update the previous accuracy dictionary
        if self.train_exp_id == self.eval_exp_id - 1:
            self.update(self.eval_exp_id,
                        self._current_accuracy.result())

        # Only after the previous experience was trained on can we return the forward transfer metric for this experience.
        if self.result(k=self.eval_exp_id) is not None:
            return self._package_result(strategy)

    def _package_result(self, strategy: 'BaseStrategy') \
            -> MetricResult:

        forward_transfer = self.result(k=self.eval_exp_id)
        metric_name = get_metric_name(self, strategy, add_experience=True)
        plot_x_position = self.get_global_counter()

        metric_values = [MetricValue(
            self, metric_name, forward_transfer, plot_x_position)]
        return metric_values

    def __str__(self):
        return "ExperienceForwardTransfer"

class StreamForwardTransfer(PluginMetric[Dict[int, float]]):
    """
        The StreamForwardTransfer metric, describing the average evaluation forward transfer
        detected over all experiences observed during training.

        This plugin metric, computed over all observed experiences during training,
        is the average over the difference between the accuracy result obtained
        after the previous experience and the accuracy result obtained
        on random initialization.

        This metric is computed during the eval phase only.
        """

    def __init__(self):
        """
        Creates an instance of the StreamForwardTransfer metric.
        """

        super().__init__()

        self.stream_forward_transfer = Mean()
        """
        The average forward transfer over all experiences
        """

        self.forward_transfer = ForwardTransfer()
        """
        The general metric to compute forward transfer
        """

        self._current_accuracy = Accuracy()
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
        Resets the forward transfer metrics.

        Note that this will reset the previous and initial accuracy of each
        experience.

        :return: None.
        """
        self.forward_transfer.reset()
        self.stream_forward_transfer.reset()

    def exp_update(self, k, v, initial=False):
        """
        Update forward transfer metric.
        See `Forward Transfer` for more detailed information.

        :param k: key to update
        :param v: value associated to k
        :param initial: update initial value. If False, update
            previous value.
        """
        self.forward_transfer.update(k, v, initial=initial)

    def exp_result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        Result for experience defined by a key.
        See `ForwardTransfer` documentation for more detailed information.

        k: optional key from which to compute forward transfer.
        """
        return self.forward_transfer.result(k=k)

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        The average forward transfer over all experiences.

        k: optional key from which to compute forward transfer.
        """
        return self.stream_forward_transfer.result()

    def before_training_exp(self, strategy: 'BaseStrategy') -> None:
        self.train_exp_id = strategy.experience.current_experience

    def before_eval(self, strategy) -> None:
        self.stream_forward_transfer.reset()

    def before_eval_exp(self, strategy: 'BaseStrategy') -> None:
        self._current_accuracy.reset()

    def after_eval_iteration(self, strategy: 'BaseStrategy') -> None:
        self.eval_exp_id = strategy.experience.current_experience
        self._current_accuracy.update(strategy.mb_y,
                                      strategy.logits)

    def after_eval_exp(self, strategy: 'BaseStrategy') -> None:
        # Only if eval experience is one after the train experience, do we update the previous accuracy dictionary
        if self.train_exp_id == self.eval_exp_id - 1:
            self.update(self.eval_exp_id,
                        self._current_accuracy.result())

        # Only after the previous experience was trained on can we return the forward transfer metric for this experience.
        if self.result(k=self.eval_exp_id) is not None:
            exp_forward_transfer = self.exp_result(k=self.eval_exp_id)
            self.stream_forward_transfer.update(exp_forward_transfer, weight=1)

    def after_eval(self, strategy: 'BaseStrategy') -> \
            'MetricResult':
        return self._package_result(strategy)

    def _package_result(self, strategy: 'BaseStrategy') -> \
            MetricResult:
        metric_value = self.result()

        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = '{}/{}_phase/{}_stream' \
            .format(str(self),
                    phase_name,
                    stream)
        plot_x_position = self.get_global_counter()

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "StreamForwardTransfer"

__all__ = [
'ForwardTransfer',
'ExperienceForwardTransfer',
'StreamForwardTransfer'
]