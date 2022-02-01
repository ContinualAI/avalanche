################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Diganta Misra                                                     #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

import copy
from typing import TYPE_CHECKING

from torch import Tensor

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class WeightCheckpoint(PluginMetric[Tensor]):
    """
    The WeightCheckpoint Metric. This is a standalone metric.

    Instances of this metric keeps the weight checkpoint tensor of the
    model at each experience.

    Each time `result` is called, this metric emits the latest experience's
    weight checkpoint tensor since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return None.
    """

    def __init__(self):
        """
        Creates an instance of the WeightCheckpoint Metric.

        By default this metric in its initial state will return None.
        The metric can be updated by using the `update` method
        while the current experience's weight checkpoint tensor can be
        retrieved using the `result` method.
        """
        super().__init__()
        self.weights = None

    def update(self, weights) -> Tensor:
        """
        Update the weight checkpoint at the current experience.

        :param weights: the weight tensor at current experience
        :return: None.
        """
        self.weights = weights

    def result(self) -> Tensor:
        """
        Retrieves the weight checkpoint at the current experience.

        :return: The weight checkpoint as a tensor.
        """
        return self.weights

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self.weights = None

    def _package_result(self, strategy) -> "MetricResult":
        weights = self.result()
        metric_name = get_metric_name(
            self, strategy, add_experience=True, add_task=False
        )
        return [
            MetricValue(
                self, metric_name, weights, strategy.clock.train_iterations
            )
        ]

    def after_eval_exp(self, strategy: "SupervisedTemplate") -> "MetricResult":
        model_params = copy.deepcopy(strategy.model.parameters())
        self.update(model_params)

    def __str__(self):
        return "WeightCheckpoint"


__all__ = ["WeightCheckpoint"]
