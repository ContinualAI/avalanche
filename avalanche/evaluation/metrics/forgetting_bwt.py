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

from abc import ABC, abstractmethod
from typing import (
    Dict,
    TYPE_CHECKING,
    Generic,
    Optional,
    TypeVar,
    Union,
    List,
    overload,
)

from avalanche.evaluation.metric_definitions import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metrics import TaskAwareAccuracy, Mean
from avalanche.evaluation.metric_utils import (
    get_metric_name,
    phase_and_task,
    stream_type,
)

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

TResult_co = TypeVar("TResult_co", covariant=True)
TResultKey_co = TypeVar("TResultKey_co", covariant=True)
TMetric = TypeVar("TMetric", bound=Metric)


class Forgetting(Metric[Dict[int, float]]):
    """
    The standalone Forgetting metric.
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
        Creates an instance of the standalone Forgetting metric
        """

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

    def result_key(self, k: int) -> Optional[float]:
        """
        Compute the forgetting for a specific key.

        :param k: the key for which returning forgetting.

        :return: the difference between the first and last value encountered
            for k, if k is not None. It returns None if k has not been updated
            at least twice.
        """
        assert k is not None
        if k in self.initial and k in self.last:
            return self.initial[k] - self.last[k]
        else:
            return None

    def result(self) -> Dict[int, float]:
        """
        Compute the forgetting for all keys.

        :return: A dictionary containing keys whose value has been updated
            at least twice. The associated value is the difference between
            the first and last value recorded for that key.
        """

        ik = set(self.initial.keys())
        both_keys = list(ik.intersection(set(self.last.keys())))

        forgetting: Dict[int, float] = {}
        for k in both_keys:
            forgetting[k] = self.initial[k] - self.last[k]

        return forgetting

    def reset_last(self) -> None:
        self.last = dict()

    def reset(self) -> None:
        self.initial = dict()
        self.last = dict()


class GenericExperienceForgetting(
    PluginMetric[TResult_co], Generic[TMetric, TResult_co, TResultKey_co], ABC
):
    """
    The GenericExperienceForgetting metric, describing the change in
    a metric detected for a certain experience. The user should
    subclass this and provide the desired metric.

    In particular, the user should override:
    * __init__ by calling `super` and instantiating the `self.current_metric`
    property as a valid avalanche metric
    * `metric_update`, to update `current_metric`
    * `metric_result` to get the result from `current_metric`.
    * `__str__` to define the experience forgetting  name.

    This plugin metric, computed separately for each experience,
    is the difference between the metric result obtained after
    first training on a experience and the metric result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self, current_metric: TMetric):
        """
        Creates an instance of the GenericExperienceForgetting metric.
        """

        super().__init__()

        self.forgetting: Forgetting = Forgetting()
        """
        The general metric to compute forgetting
        """

        self._current_metric: TMetric = current_metric
        """
        The metric the user should override
        """

        self.eval_exp_id: int = -1
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

        Beware that this will also reset the initial metric of each
        experience!

        :return: None.
        """
        self.forgetting.reset()

    def reset_last(self) -> None:
        """
        Resets the last metric value.

        This will preserve the initial metric value of each experience.
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

    @abstractmethod
    def result_key(self, k: int) -> TResultKey_co:
        pass

    @abstractmethod
    def result(self) -> TResult_co:
        pass

    def before_training_exp(self, strategy: "SupervisedTemplate") -> None:
        assert strategy.experience is not None
        self.train_exp_id = strategy.experience.current_experience

    def before_eval(self, strategy) -> None:
        self.reset_last()

    def before_eval_exp(self, strategy: "SupervisedTemplate") -> None:
        self._current_metric.reset()

    def after_eval_iteration(self, strategy: "SupervisedTemplate") -> None:
        super().after_eval_iteration(strategy)
        assert strategy.experience is not None
        self.eval_exp_id = strategy.experience.current_experience
        self.metric_update(strategy)

    def after_eval_exp(self, strategy: "SupervisedTemplate") -> MetricResult:
        # update experience on which training just ended
        self._check_eval_exp_id()
        if self.train_exp_id == self.eval_exp_id:
            self.update(self.eval_exp_id, self.metric_result(strategy), initial=True)
        else:
            # update other experiences
            # if experience has not been encountered in training
            # its value will not be considered in forgetting
            self.update(self.eval_exp_id, self.metric_result(strategy))

        return self._package_result(strategy)

    def after_eval(self, strategy: "SupervisedTemplate") -> MetricResult:
        self.eval_exp_id = -1  # reset the last experience ID
        return super().after_eval(strategy)

    def _package_result(self, strategy: "SupervisedTemplate") -> MetricResult:
        # this checks if the evaluation experience has been
        # already encountered at training time
        # before the last training.
        # If not, forgetting should not be returned.
        self._check_eval_exp_id()

        forgetting = self.result_key(k=self.eval_exp_id)
        if forgetting is not None:
            metric_name = get_metric_name(self, strategy, add_experience=True)
            plot_x_position = strategy.clock.train_iterations

            metric_values = [
                MetricValue(self, metric_name, forgetting, plot_x_position)
            ]
            return metric_values
        return None

    @abstractmethod
    def metric_update(self, strategy):
        pass

    @abstractmethod
    def metric_result(self, strategy):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def _check_eval_exp_id(self):
        assert self.eval_exp_id >= 0, (
            "The evaluation loop executed 0 iterations. "
            "This is not suported while using this metric"
        )


class ExperienceForgetting(
    GenericExperienceForgetting[TaskAwareAccuracy, Dict[int, float], Optional[float]]
):
    """
    The ExperienceForgetting metric, describing the accuracy loss
    detected for a certain experience.

    This plugin metric, computed separately for each experience,
    is the difference between the accuracy result obtained after
    first training on a experience and the accuracy result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the ExperienceForgetting metric.
        """

        super().__init__(TaskAwareAccuracy())

    def result_key(self, k: int) -> Optional[float]:
        """
        Forgetting for an experience defined by its key.

        See :class:`Forgetting` documentation for more detailed information.

        :param k: key from which to compute the forgetting.
        :return: the difference between the first and last value encountered
            for k.
        """
        return self.forgetting.result_key(k=k)

    def result(self) -> Dict[int, float]:
        """
        Forgetting for all experiences.

        See :class:`Forgetting` documentation for more detailed information.

        :return: A dictionary containing keys whose value has been updated
            at least twice. The associated value is the difference between
            the first and last value recorded for that key.
        """
        return self.forgetting.result()

    def metric_update(self, strategy):
        self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)

    def metric_result(self, strategy):
        return self._current_metric.result(0)[0]

    def __str__(self):
        return "ExperienceForgetting"


class GenericStreamForgetting(
    GenericExperienceForgetting[TMetric, float, Optional[float]]
):
    """
    The GenericStreamForgetting metric, describing the average evaluation
    change in the desired metric detected over all experiences observed
    during training.

    In particular, the user should override:
    * __init__ by calling `super` and instantiating the `self.current_metric`
    property as a valid avalanche metric
    * `metric_update`, to update `current_metric`
    * `metric_result` to get the result from `current_metric`.
    * `__str__` to define the experience forgetting  name.

    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the metric result obtained
    after first training on a experience and the metric result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self, current_metric: TMetric):
        """
        Creates an instance of the GenericStreamForgetting metric.
        """

        super().__init__(current_metric)

        self.stream_forgetting = Mean()
        """
        The average forgetting over all experiences
        """

    def reset(self) -> None:
        """
        Resets the forgetting metrics.

        Beware that this will also reset the initial metric value of each
        experience!

        :return: None.
        """
        super().reset()
        self.stream_forgetting.reset()

    def exp_update(self, k, v, initial=False):
        """
        Update forgetting metric.
        See `Forgetting` for more detailed information.

        :param k: key to update
        :param v: value associated to k
        :param initial: update initial value. If False, update
            last value.
        """
        super().update(k, v, initial=initial)

    def exp_result(self, k: int) -> Optional[float]:
        """
        Result for experience defined by a key.
        See `Forgetting` documentation for more detailed information.

        k: optional key from which compute forgetting.
        """
        return self.result_key(k=k)

    def result_key(self, k: int) -> Optional[float]:
        """
        Result for experience defined by a key.
        See `Forgetting` documentation for more detailed information.

        k: optional key from which compute forgetting.
        """
        return self.forgetting.result_key(k=k)

    def result(self) -> float:
        """
        The average forgetting over all experience.
        """
        return self.stream_forgetting.result()

    def before_eval(self, strategy) -> None:
        super().before_eval(strategy)
        self.stream_forgetting.reset()

    def after_eval_exp(self, strategy: "SupervisedTemplate") -> None:
        # update experience on which training just ended
        self._check_eval_exp_id()
        if self.train_exp_id == self.eval_exp_id:
            self.exp_update(
                self.eval_exp_id, self.metric_result(strategy), initial=True
            )
        else:
            # update other experiences
            # if experience has not been encountered in training
            # its value will not be considered in forgetting
            self.exp_update(self.eval_exp_id, self.metric_result(strategy))

        # this checks if the evaluation experience has been
        # already encountered at training time
        # before the last training.
        # If not, forgetting should not be returned.
        exp_forgetting = self.exp_result(k=self.eval_exp_id)
        if exp_forgetting is not None:
            self.stream_forgetting.update(exp_forgetting, weight=1)

    def after_eval(self, strategy: "SupervisedTemplate") -> "MetricResult":
        return self._package_result(strategy)

    def _package_result(self, strategy: "SupervisedTemplate") -> MetricResult:
        assert strategy.experience is not None
        metric_value = self.result()

        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        metric_name = "{}/{}_phase/{}_stream".format(str(self), phase_name, stream)
        plot_x_position = strategy.clock.train_iterations

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def metric_update(self, strategy):
        raise NotImplementedError

    def metric_result(self, strategy):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class StreamForgetting(GenericStreamForgetting[TaskAwareAccuracy]):
    """
    The StreamForgetting metric, describing the average evaluation accuracy loss
    detected over all experiences observed during training.

    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the accuracy result obtained
    after first training on a experience and the accuracy result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the StreamForgetting metric.
        """

        super().__init__(TaskAwareAccuracy())

    def metric_update(self, strategy):
        self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)

    def metric_result(self, strategy):
        return self._current_metric.result(0)[0]

    def __str__(self):
        return "StreamForgetting"


def forgetting_metrics(*, experience=False, stream=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the forgetting on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the forgetting averaged over the evaluation stream experiences,
        which have been observed during training.

    :return: A list of plugin metrics.
    """

    metrics: List[PluginMetric] = []

    if experience:
        metrics.append(ExperienceForgetting())

    if stream:
        metrics.append(StreamForgetting())

    return metrics


@overload
def forgetting_to_bwt(f: float) -> float:
    ...


@overload
def forgetting_to_bwt(f: Dict[int, float]) -> Dict[int, float]:
    ...


@overload
def forgetting_to_bwt(f: None) -> None:
    ...


def forgetting_to_bwt(f: Optional[Union[float, Dict[int, float]]]):
    """
    Convert forgetting to backward transfer.
    BWT = -1 * forgetting
    """
    if f is None:
        return f
    if isinstance(f, dict):
        return {k: -1 * v for k, v in f.items()}
    elif isinstance(f, float):
        return -1 * f
    else:
        raise ValueError(
            "Forgetting data type not recognized when converting"
            "to backward transfer."
        )


class BWT(Forgetting):
    """
    The standalone Backward Transfer metric.
    This metric returns the backward transfer relative to a specific key.
    Alternatively, this metric returns a dict in which each key is associated
    to the backward transfer.
    Backward transfer is computed as the difference between the last value
    recorded for a specific key and the first value recorded for that key.
    The value associated to a key can be update with the `update` method.

    At initialization, this metric returns an empty dictionary.
    """

    def result_key(self, k: int) -> Optional[float]:
        """
        Backward Transfer is returned only for keys encountered twice.
        Backward Transfer is the negative forgetting.

        :param k: the key for which returning backward transfer. If k has not
            updated at least twice it returns None.

        :return: the difference between the last value encountered for k
            and its first value.
            It returns None if k has not been updated
            at least twice.
        """

        forgetting = super().result_key(k)
        bwt = forgetting_to_bwt(forgetting)
        return bwt

    def result(self) -> Dict[int, float]:
        """
        Backward Transfer is returned only for keys encountered twice.
        Backward Transfer is the negative forgetting.

        Backward transfer will be returned for all keys encountered at
        least twice.

        :return: A dictionary containing keys whose value has been
            updated at least twice. The associated value is the difference
            between the last and first value recorded for that key.
        """

        forgetting = super().result()
        bwt = forgetting_to_bwt(forgetting)
        return bwt


class ExperienceBWT(ExperienceForgetting):
    """
    The Experience Backward Transfer metric.

    This plugin metric, computed separately for each experience,
    is the difference between the last accuracy result obtained on a certain
    experience and the accuracy result obtained when first training on that
    experience.

    This metric is computed during the eval phase only.
    """

    def result_key(self, k=None) -> Optional[float]:
        """
        See `Forgetting` documentation for more detailed information.

        k: key from which compute forgetting.
        """
        forgetting = super().result_key(k)
        return forgetting_to_bwt(forgetting)

    def result(self) -> Dict[int, float]:
        """
        See `Forgetting` documentation for more detailed information.
        """
        forgetting = super().result()
        return forgetting_to_bwt(forgetting)

    def __str__(self):
        return "ExperienceBWT"


class StreamBWT(StreamForgetting):
    """
    The StreamBWT metric, emitting the average BWT across all experiences
    encountered during training.

    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the last accuracy result
    obtained on an experience and the accuracy result obtained when first
    training on that experience.

    This metric is computed during the eval phase only.
    """

    def exp_result(self, k: int) -> Optional[float]:
        """
        Result for experience defined by a key.
        See `BWT` documentation for more detailed information.

        k: optional key from which compute backward transfer.
        """
        forgetting = super().exp_result(k)
        return forgetting_to_bwt(forgetting)

    def __str__(self):
        return "StreamBWT"


def bwt_metrics(*, experience=False, stream=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the backward transfer on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the backward transfer averaged over the evaluation stream experiences
        which have been observed during training.
    :return: A list of plugin metrics.
    """

    metrics: List[PluginMetric] = []

    if experience:
        metrics.append(ExperienceBWT())

    if stream:
        metrics.append(StreamBWT())

    return metrics


__all__ = [
    "Forgetting",
    "GenericExperienceForgetting",
    "GenericStreamForgetting",
    "ExperienceForgetting",
    "StreamForgetting",
    "forgetting_metrics",
    "BWT",
    "ExperienceBWT",
    "StreamBWT",
    "bwt_metrics",
]
