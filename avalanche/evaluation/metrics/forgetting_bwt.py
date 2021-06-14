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

from typing import Dict, TYPE_CHECKING, Union, List

from avalanche.evaluation.metric_definitions import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metrics import Accuracy, Mean
from avalanche.evaluation.metric_utils import get_metric_name, \
    phase_and_task, stream_type

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy


class Forgetting(Metric[Union[float, None, Dict[int, float]]]):
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


class GenericExperienceForgetting(PluginMetric[Dict[int, float]]):
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

    def __init__(self):
        """
        Creates an instance of the GenericExperienceForgetting metric.
        """

        super().__init__()

        self.forgetting = Forgetting()
        """
        The general metric to compute forgetting
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

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        See `Forgetting` documentation for more detailed information.

        k: optional key from which compute forgetting.
        """
        return self.forgetting.result(k=k)

    def before_training_exp(self, strategy: 'BaseStrategy') -> None:
        self.train_exp_id = strategy.experience.current_experience

    def before_eval(self, strategy) -> None:
        self.reset_last()

    def before_eval_exp(self, strategy: 'BaseStrategy') -> None:
        self._current_metric.reset()

    def after_eval_iteration(self, strategy: 'BaseStrategy') -> None:
        super().after_eval_iteration(strategy)
        self.eval_exp_id = strategy.experience.current_experience
        self.metric_update(strategy)

    def after_eval_exp(self, strategy: 'BaseStrategy') \
            -> MetricResult:
        # update experience on which training just ended
        if self.train_exp_id == self.eval_exp_id:
            self.update(self.eval_exp_id,
                        self.metric_result(strategy),
                        initial=True)
        else:
            # update other experiences
            # if experience has not been encountered in training
            # its value will not be considered in forgetting
            self.update(self.eval_exp_id,
                        self.metric_result(strategy))

        return self._package_result(strategy)

    def _package_result(self, strategy: 'BaseStrategy') \
            -> MetricResult:
        # this checks if the evaluation experience has been
        # already encountered at training time
        # before the last training.
        # If not, forgetting should not be returned.
        forgetting = self.result(k=self.eval_exp_id)
        if forgetting is not None:
            metric_name = get_metric_name(self, strategy, add_experience=True)
            plot_x_position = self.get_global_counter()

            metric_values = [MetricValue(
                self, metric_name, forgetting, plot_x_position)]
            return metric_values

    def metric_update(self, strategy):
        raise NotImplementedError

    def metric_result(self, strategy):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class ExperienceForgetting(GenericExperienceForgetting):
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

        super().__init__()

        self._current_metric = Accuracy()
        """
        The average accuracy over the current evaluation experience
        """

    def metric_update(self, strategy):
        self._current_metric.update(strategy.mb_y,
                                    strategy.mb_output, 0)

    def metric_result(self, strategy):
        return self._current_metric.result(0)[0]

    def __str__(self):
        return "ExperienceForgetting"


class GenericStreamForgetting(GenericExperienceForgetting):
    """
    The GenericStreamForgetting metric, describing the average evaluation change
    in the desired metric detected over all experiences observed during training.

    In particular, the user should override __init__ by calling `super`
    and instantiating the `self.current_metric` property as a valid
    avalanche metric. The user should also override `after_eval_iteration`,
    call `super` and the `update` method of the chosen metric.
    In addition, the method `__str__` must be defined.

    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the metric result obtained
    after first training on a experience and the metric result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the GenericStreamForgetting metric.
        """

        super().__init__()

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

    def exp_result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        Result for experience defined by a key.
        See `Forgetting` documentation for more detailed information.

        k: optional key from which compute forgetting.
        """
        return super().result(k)

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        The average forgetting over all experience.

        k: optional key from which compute forgetting.
        """
        return self.stream_forgetting.result()

    def before_eval(self, strategy) -> None:
        super().before_eval(strategy)
        self.stream_forgetting.reset()

    def after_eval_exp(self, strategy: 'BaseStrategy') -> None:
        # update experience on which training just ended
        if self.train_exp_id == self.eval_exp_id:
            self.exp_update(self.eval_exp_id,
                            self.metric_result(strategy),
                            initial=True)
        else:
            # update other experiences
            # if experience has not been encountered in training
            # its value will not be considered in forgetting
            self.exp_update(self.eval_exp_id,
                            self.metric_result(strategy))

        # this checks if the evaluation experience has been
        # already encountered at training time
        # before the last training.
        # If not, forgetting should not be returned.
        exp_forgetting = self.exp_result(k=self.eval_exp_id)
        if exp_forgetting is not None:
            self.stream_forgetting.update(exp_forgetting, weight=1)

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

    def metric_update(self, strategy):
        raise NotImplementedError

    def metric_result(self, strategy):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class StreamForgetting(GenericStreamForgetting):
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

        super().__init__()

        self._current_metric = Accuracy()
        """
        The average accuracy over the current evaluation experience
        """

    def metric_update(self, strategy):
        self._current_metric.update(strategy.mb_y,
                                    strategy.mb_output, 0)

    def metric_result(self, strategy):
        return self._current_metric.result(0)[0]

    def __str__(self):
        return "StreamForgetting"


class GenericTaskForgetting(PluginMetric[Dict[int, float]]):
    def __init__(self):
        super().__init__()
        self.forgetting = Forgetting()
        self._current_train_metric = None
        self._current_eval_metric = None

    def reset(self, **kwargs) -> None:
        self.forgetting.reset()

    def result(self, **kwargs):
        return self.forgetting.result()

    def update(self, k, v, initial):
        self.forgetting.update(k, v, initial=initial)

    def before_training(self, strategy: 'BaseStrategy'):
        self._current_train_metric.reset()

    def after_training_iteration(self, strategy: 'BaseStrategy'):
        super().__init__()
        try:
            for t in strategy.mb_task_id:
                self._current_train_metric.reset(t)
        except AssertionError:
            self._current_train_metric.reset()
        self.metric_update(strategy, train=True)

    def before_eval(self, strategy: 'BaseStrategy'):
        self.forgetting.reset_last()
        for k, v in self._current_train_metric.result().items():
            self.update(k, v, initial=True)
        self._current_eval_metric.reset()

    def after_eval_iteration(self, strategy: 'BaseStrategy'):
        super().after_eval_iteration(strategy)
        self.metric_update(strategy, train=False)

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        for k, v in self._current_eval_metric.result().items():
            self.update(k, v, initial=False)
        return self._package_result(strategy)

    def _package_result(self, strategy: 'BaseStrategy') -> \
            MetricResult:
        metric_value = self.result()
        plot_x_position = self.get_global_counter()
        results = []
        for k, v in metric_value.items():
            metric_name = get_metric_name(self, strategy, add_experience=False,
                                          add_task=k)
            results.append(MetricValue(self, metric_name, v, plot_x_position))

        return results

    def metric_update(self, strategy, train):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class TaskForgetting(GenericTaskForgetting):

    def __init__(self):
        super().__init__()
        self._current_train_metric = Accuracy()
        self._current_eval_metric = Accuracy()

    def metric_update(self, strategy, train):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        if train:
            self._current_train_metric.update(strategy.mb_output, strategy.mb_y,
                                              task_labels)
        else:
            self._current_eval_metric.update(strategy.mb_output, strategy.mb_y,
                                             task_labels)

    def __str__(self):
        return "TaskForgetting"


def forgetting_metrics(*, experience=False, stream=False, task=False) \
        -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the forgetting on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the forgetting averaged over the evaluation stream experiences,
        which have been observed during training.
    :param task: If True, will return a metric able to log the forgetting
        across each task encountered during training and evaluation.

    :return: A list of plugin metrics.
    """

    metrics = []

    if experience:
        metrics.append(ExperienceForgetting())

    if stream:
        metrics.append(StreamForgetting())

    if task:
        metrics.append(TaskForgetting())

    return metrics


def forgetting_to_bwt(f):
    """
    Convert forgetting to backward transfer.
    BWT = -1 * forgetting
    """
    if f is None:
        return f
    if isinstance(f, dict):
        bwt = {k: -1 * v for k, v in f.items()}
    elif isinstance(f, float):
        bwt = -1 * f
    else:
        raise ValueError("Forgetting data type not recognized when converting"
                         "to backward transfer.")
    return bwt


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

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        Backward Transfer is returned only for keys encountered twice.
        Backward Transfer is the negative forgetting.

        :param k: the key for which returning backward transfer. If k has not
            updated at least twice it returns None. If k is None,
            backward transfer will be returned for all keys encountered at
            least twice.

        :return: the difference between the last value encountered for k
            and its first value, if k is not None.
            It returns None if k has not been updated
            at least twice. If k is None, returns a dictionary
            containing keys whose value has been updated at least twice. The
            associated value is the difference between the last and first
            value recorded for that key.
        """

        forgetting = super().result(k)
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

    def result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        See `Forgetting` documentation for more detailed information.

        k: optional key from which compute forgetting.
        """
        forgetting = super().result(k)
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

    def exp_result(self, k=None) -> Union[float, None, Dict[int, float]]:
        """
        Result for experience defined by a key.
        See `BWT` documentation for more detailed information.

        k: optional key from which compute backward transfer.
        """
        forgetting = super().exp_result(k)
        return forgetting_to_bwt(forgetting)

    def __str__(self):
        return "StreamBWT"


class TaskBWT(TaskForgetting):
    """
    The TaskBWT metric, emitting the average BWT task-wise.

    This plugin metric, computed over all observed tasks during training,
    is the average over the difference between the last accuracy result
    obtained on a task and the accuracy result obtained when
    training on that task.

    This metric is computed during the eval phase only.
    """

    def result(self) -> Union[float, None, Dict[int, float]]:
        """
        Result for experience defined by a key.
        See `BWT` documentation for more detailed information.

        """
        forgetting = super().result()
        return forgetting_to_bwt(forgetting)

    def __str__(self):
        return "TaskBWT"


def bwt_metrics(*, experience=False, stream=False, task=False) \
        -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the backward transfer on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the backward transfer averaged over the evaluation stream experiences
        which have been observed during training.
    :param task: If True, will return a metric able to log
        the backward transfer for each task in the evaluation stream
    :return: A list of plugin metrics.
    """

    metrics = []

    if experience:
        metrics.append(ExperienceBWT())

    if stream:
        metrics.append(StreamBWT())

    if task:
        metrics.append(TaskBWT())

    return metrics


__all__ = [
    'Forgetting',
    'GenericExperienceForgetting',
    'GenericStreamForgetting',
    'GenericTaskForgetting',
    'ExperienceForgetting',
    'StreamForgetting',
    'TaskForgetting',
    'forgetting_metrics',
    'BWT',
    'ExperienceBWT',
    'StreamBWT',
    'TaskBWT',
    'bwt_metrics'
]
