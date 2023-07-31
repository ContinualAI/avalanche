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
import io
from typing import TYPE_CHECKING, Optional

from torch import Tensor
import torch

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class WeightCheckpoint(PluginMetric[Tensor]):
    """
    The WeightCheckpoint Metric.

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
        self.weights: Optional[bytes] = None

    def update(self, weights: bytes):
        """
        Update the weight checkpoint at the current experience.

        :param weights: the weight tensor at current experience
        :return: None.
        """
        self.weights = weights

    def result(self) -> Optional[bytes]:
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
        if weights is None:
            return None

        metric_name = get_metric_name(
            self, strategy, add_experience=True, add_task=False
        )
        return [
            MetricValue(self, metric_name, weights, strategy.clock.train_iterations)
        ]

    def after_training_exp(self, strategy: "SupervisedTemplate") -> "MetricResult":
        buff = io.BytesIO()
        model_params = copy.deepcopy(strategy.model).to("cpu")
        torch.save(model_params, buff)
        buff.seek(0)
        self.update(buff.read())

        return self._package_result(strategy)

    def __str__(self):
        return "WeightCheckpoint"


__all__ = ["WeightCheckpoint"]
