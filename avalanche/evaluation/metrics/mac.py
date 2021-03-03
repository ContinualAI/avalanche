################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-01-2021                                                             #
# Author(s): Vincenzo Lomonaco, Lorenzo Pellegrini                             #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from torch.nn import Module
from typing import TYPE_CHECKING, List, Optional
from torch import Tensor

from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, \
    phase_and_task, stream_type

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy


class MAC(Metric[int]):
    """
    Multiply-and-accumulate metric. Provides a lower bound of the
    computational cost of a model in a hardware-independent way by
    computing the number of multiplications. Currently supports only
    Linear or Conv2d modules. Other operations are ignored.
    """
    def __init__(self):
        """
        Creates an instance of the MAC metric.
        """
        self.hooks = []
        self._compute_cost: Optional[int] = None

    def update(self, model: Module, dummy_input: Tensor):
        """
        Computes the MAC metric.

        :param model: current model.
        :param dummy_input: A tensor of the correct size to feed as input
            to model. It includes batch size
        :return: MAC metric.
        """

        for mod in model.modules():
            if MAC.is_recognized_module(mod):
                def foo(a, b, c):
                    return self.update_compute_cost(a, b, c)
                handle = mod.register_forward_hook(foo)
                self.hooks.append(handle)

        self._compute_cost = 0
        model(dummy_input)  # trigger forward hooks

        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def result(self) -> Optional[int]:
        """
        Return the number of MAC operations as computed in the previous call
        to the `update` method.

        :return: The number of MAC operations or None if `update` has not been
            called yet.
        """
        return self._compute_cost

    def update_compute_cost(self, module, dummy_input, output):
        modname = module.__class__.__name__
        if modname == 'Linear':
            self._compute_cost += dummy_input[0].shape[1] * output.shape[1]
        elif modname == 'Conv2d':
            n, cout, hout, wout = output.shape  # Batch, Channels, Height, Width
            ksize = module.kernel_size[0] * module.kernel_size[1]
            self._compute_cost += cout * hout * wout * ksize

    @staticmethod
    def is_recognized_module(mod):
        modname = mod.__class__.__name__
        return modname == 'Linear' or modname == 'Conv2d'


class MinibatchMAC(PluginMetric[float]):
    """
    The minibatch MAC metric.
    This metric only works at training time.

    This metric computes the MAC over 1 pattern
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochMAC` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchMAC metric.
        """

        super().__init__()

        self._minibatch_MAC = MAC()

    def reset(self) -> None:
        pass

    def result(self) -> float:
        return self._minibatch_MAC.result()

    def after_training_iteration(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        self._minibatch_MAC.update(strategy.model,
                                   strategy.mb_x[0].unsqueeze(0))
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        metric_value = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "MAC_MB"


class EpochMAC(PluginMetric[float]):
    """
    The MAC at the end of each epoch computed on a
    single pattern.
    This metric only works at training time.

    The MAC will be logged after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochMAC metric.
        """
        super().__init__()

        self._MAC_metric = MAC()

    def reset(self) -> None:
        pass

    def result(self) -> float:
        return self._MAC_metric.result()

    def after_training_epoch(self, strategy: 'PluggableStrategy') \
            -> MetricResult:
        self._MAC_metric.update(strategy.model,
                                strategy.mb_x[0].unsqueeze(0))
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> MetricResult:
        metric_value = self.result()

        metric_name = get_metric_name(self, strategy)
        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "MAC_Epoch"


class ExperienceMAC(PluginMetric[float]):
    """
    At the end of each experience, this metric reports the
    MAC computed on a single pattern.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceMAC metric
        """
        super().__init__()

        self._MAC_metric = MAC()

    def reset(self) -> None:
        pass

    def result(self) -> float:
        return self._MAC_metric.result()

    def after_eval_exp(self, strategy: 'PluggableStrategy') -> \
            'MetricResult':
        self._MAC_metric.update(strategy.model,
                                strategy.mb_x[0].unsqueeze(0))
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy') -> \
            MetricResult:
        metric_value = self.result()

        metric_name = get_metric_name(self, strategy, add_experience=True)

        plot_x_position = self._next_x_position(metric_name)

        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "MAC_Exp"


def MAC_metrics(*, minibatch=False, epoch=False, experience=False) \
        -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of metric.

    :param minibatch: If True, will return a metric able to log
        the MAC after each iteration at training time.
    :param epoch: If True, will return a metric able to log
        the MAC after each epoch at training time.
    :param experience: If True, will return a metric able to log
        the MAC after each eval experience.

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchMAC())

    if epoch:
        metrics.append(EpochMAC())

    if experience:
        metrics.append(ExperienceMAC())

    return metrics


__all__ = [
    'MAC',
    'MinibatchMAC',
    'EpochMAC',
    'ExperienceMAC',
    'MAC_metrics'
]
