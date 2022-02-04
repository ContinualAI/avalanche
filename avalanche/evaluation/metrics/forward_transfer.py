################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 4-02-2021                                                              #
# Author(s): Ryan Lindeborg, Andrea Cossu                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import Dict, TYPE_CHECKING, Union

from avalanche.evaluation.metric_definitions import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metrics import Accuracy, Mean
from avalanche.evaluation.metric_utils import (
    get_metric_name,
    phase_and_task,
    stream_type,
)

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class ForwardTransfer(Metric[Union[float, None, Dict[int, float]]]):
    """
    The standalone Forward Transfer metric.
    This metric returns the forward transfer relative to a specific key.
    Alternatively, this metric returns a dict in which each key is
    associated to the forward transfer.
    Forward transfer is computed as the difference between the value
    recorded for a specific key after the previous experience has
    been trained on, and random initialization before training.
    The value associated to a key can be updated with the `update` method.

    At initialization, this metric returns an empty dictionary.
    """

    def __init__(self):
        """
        Creates an instance of the standalone Forward Transfer metric
        """

        self.initial: Dict[int, float] = dict()
        """
        The initial value for each key. This is the accuracy at 
        random initialization.
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
            forward transfer will be returned for all keys
            where the previous experience has been trained on.

        :return: the difference between the key value after training on the
            previous experience, and the key at random initialization.
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
        self.previous: Dict[int, float] = dict()


class GenericExperienceForwardTransfer(PluginMetric[Dict[int, float]]):
    """
    The GenericExperienceForwardMetric metric, describing the forward transfer
    detected after a certain experience. The user should
    subclass this and provide the desired metric.

    In particular, the user should override:
    * __init__ by calling `super` and instantiating the `self.current_metric`
    property as a valid avalanche metric
    * `metric_update`, to update `current_metric`
    * `metric_result` to get the result from `current_metric`.
    * `__str__` to define the experience forward transfer  name.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the GenericExperienceForwardTransfer metric.
        """

        super().__init__()

        self.forward_transfer = ForwardTransfer()
        """
        The general metric to compute forward transfer
        """

        self._current_metric = None
        """
        The metric the user should override
        """

        self.eval_exp_id = None
        """
        The current evaluation experience id
        """

        self.train_exp_id = None
        """
        The last encountered training experience id
        """

        self.at_init = True

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

    def before_training_exp(self, strategy: "SupervisedTemplate") -> None:
        self.train_exp_id = strategy.experience.current_experience

    def after_eval(self, strategy):
        if self.at_init:
            assert (
                strategy.eval_every > -1
            ), "eval every > -1 to compute forward transfer"
            self.at_init = False

    def before_eval_exp(self, strategy: "SupervisedTemplate") -> None:
        self._current_metric.reset()

    def after_eval_iteration(self, strategy: "SupervisedTemplate") -> None:
        super().after_eval_iteration(strategy)
        self.eval_exp_id = strategy.experience.current_experience
        self.metric_update(strategy)

    def after_eval_exp(self, strategy: "SupervisedTemplate") -> MetricResult:
        if self.at_init:
            self.update(
                self.eval_exp_id, self.metric_result(strategy), initial=True
            )
        else:
            if self.train_exp_id == self.eval_exp_id - 1:
                self.update(self.eval_exp_id, self.metric_result(strategy))

                return self._package_result(strategy)

    def _package_result(self, strategy: "SupervisedTemplate") -> MetricResult:
        # Only after the previous experience was trained on can we return the
        # forward transfer metric for this experience.
        result = self.result(k=self.eval_exp_id)
        if result is not None:
            metric_name = get_metric_name(self, strategy, add_experience=True)
            plot_x_position = strategy.clock.train_iterations

            metric_values = [
                MetricValue(self, metric_name, result, plot_x_position)
            ]
            return metric_values

    def metric_update(self, strategy):
        raise NotImplementedError

    def metric_result(self, strategy):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class ExperienceForwardTransfer(GenericExperienceForwardTransfer):
    """
    The Forward Transfer computed on each experience separately.
    The transfer is computed based on the accuracy metric.
    """

    def __init__(self):
        super().__init__()

        self._current_metric = Accuracy()
        """
        The average accuracy over the current evaluation experience
        """

    def metric_update(self, strategy):
        self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)

    def metric_result(self, strategy):
        return self._current_metric.result(0)[0]

    def __str__(self):
        return "ExperienceForwardTransfer"


class GenericStreamForwardTransfer(GenericExperienceForwardTransfer):
    """
    The GenericStreamForwardTransfer metric, describing the average evaluation
    forward transfer detected over all experiences observed during training.

    In particular, the user should override:
    * __init__ by calling `super` and instantiating the `self.current_metric`
    property as a valid avalanche metric
    * `metric_update`, to update `current_metric`
    * `metric_result` to get the result from `current_metric`.
    * `__str__` to define the experience forgetting  name.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the GenericStreamForwardTransfer metric.
        """

        super().__init__()

        self.stream_forward_transfer = Mean()
        """
        The average forward transfer over all experiences
        """

    def reset(self) -> None:
        """
        Resets the forward transfer metrics.

        Note that this will reset the previous and initial accuracy of each
        experience.

        :return: None.
        """
        super().reset()
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
        super().update(k, v, initial=initial)

    def exp_result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        Result for experience defined by a key.
        See `ForwardTransfer` documentation for more detailed information.

        k: optional key from which to compute forward transfer.
        """
        return super().result(k=k)

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        The average forward transfer over all experiences.

        k: optional key from which to compute forward transfer.
        """
        return self.stream_forward_transfer.result()

    def before_eval(self, strategy) -> None:
        super().before_eval(strategy)
        self.stream_forward_transfer.reset()

    def after_eval_exp(self, strategy: "SupervisedTemplate") -> None:
        if self.at_init:
            self.update(
                self.eval_exp_id, self.metric_result(strategy), initial=True
            )
        else:
            if self.train_exp_id == self.eval_exp_id - 1:
                self.update(self.eval_exp_id, self.metric_result(strategy))
            exp_forward_transfer = self.exp_result(k=self.eval_exp_id)
            if exp_forward_transfer is not None:
                self.stream_forward_transfer.update(
                    exp_forward_transfer, weight=1
                )

    def after_eval(self, strategy: "SupervisedTemplate") -> "MetricResult":
        super().after_eval(strategy)
        return self._package_result(strategy)

    def _package_result(self, strategy: "SupervisedTemplate") -> MetricResult:
        metric_value = self.result()

        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = "{}/{}_phase/{}_stream".format(
            str(self), phase_name, stream
        )
        plot_x_position = strategy.clock.train_iterations

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def metric_update(self, strategy):
        raise NotImplementedError

    def metric_result(self, strategy):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class StreamForwardTransfer(GenericStreamForwardTransfer):
    """
    The Forward Transfer averaged over all the evaluation experiences.

    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the accuracy result obtained
    after the previous experience and the accuracy result obtained
    on random initialization.
    """

    def __init__(self):
        super().__init__()
        self._current_metric = Accuracy()
        """
        The average accuracy over the current evaluation experience
        """

    def metric_update(self, strategy):
        self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)

    def metric_result(self, strategy):
        return self._current_metric.result(0)[0]

    def __str__(self):
        return "StreamForwardTransfer"


def forward_transfer_metrics(*, experience=False, stream=False):
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the forward transfer on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the forward transfer averaged over the evaluation stream experiences,
        which have been observed during training.

    :return: A list of plugin metrics.
    """

    metrics = []

    if experience:
        metrics.append(ExperienceForwardTransfer())

    if stream:
        metrics.append(StreamForwardTransfer())

    return metrics


__all__ = [
    "ForwardTransfer",
    "GenericExperienceForwardTransfer",
    "ExperienceForwardTransfer",
    "GenericStreamForwardTransfer",
    "StreamForwardTransfer",
    "forward_transfer_metrics",
]
