################################################################################
# Copyright (c) 2025 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-01-2025                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

from typing import Dict, Optional, List

from avalanche.evaluation.metrics import (
    GenericExperienceForgetting,
    GenericStreamForgetting,
)
from avalanche.evaluation.metric_definitions import PluginMetric

from avalanche.evaluation.metrics import TaskAwareRMSE, TaskAwareR2


class ExperienceRMSEForgetting(
    GenericExperienceForgetting[TaskAwareRMSE, Dict[int, float], Optional[float]]
):
    """
    The ExperienceRMSEForgetting metric, describing the RMSE loss
    detected for a certain experience.

    This plugin metric, computed separately for each experience,
    is the difference between the RMSE result obtained after
    first training on a experience and the RMSE result obtained
    on the same experience at the end of successive experiences.

    Since RMSE is to be minimized, forgetting is observed when
    the result is negative.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the ExperienceRMSEForgetting metric.
        """

        super().__init__(TaskAwareRMSE())

    def result_key(self, k: int) -> Optional[float]:
        """
        RMSEForgetting for an experience defined by its key.

        :param k: key from which to compute the forgetting.
        :return: the difference between the first and last value encountered
            for k.
        """
        return self.forgetting.result_key(k=k)

    def result(self) -> Dict[int, float]:
        """
        RMSEForgetting for all experiences.

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
        return "ExperienceRMSEForgetting"


class StreamRMSEForgetting(GenericStreamForgetting[TaskAwareRMSE]):
    """
    The StreamRMSEForgetting metric, describing the average evaluation RMSE loss
    detected over all experiences observed during training.

    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the RMSE result obtained
    after first training on a experience and the RMSE result obtained
    on the same experience at the end of successive experiences.

    Since RMSE is to be minimized, forgetting is observed when
    the result is negative.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the StreamRMSEForgetting metric.
        """

        super().__init__(TaskAwareRMSE())

    def metric_update(self, strategy):
        self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)

    def metric_result(self, strategy):
        return self._current_metric.result(0)[0]

    def __str__(self):
        return "StreamRMSEForgetting"


def rmse_forgetting_metrics(*, experience=False, stream=False) -> List[PluginMetric]:
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
        metrics.append(ExperienceRMSEForgetting())

    if stream:
        metrics.append(StreamRMSEForgetting())

    return metrics


class ExperienceR2Forgetting(
    GenericExperienceForgetting[TaskAwareR2, Dict[int, float], Optional[float]]
):
    """
    The ExperienceR2Forgetting metric, describing the R2 loss
    detected for a certain experience.

    This plugin metric, computed separately for each experience,
    is the difference between the R2 result obtained after
    first training on a experience and the R2 result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the ExperienceR2Forgetting metric.
        """

        super().__init__(TaskAwareR2())

    def result_key(self, k: int) -> Optional[float]:
        """
        R2Forgetting for an experience defined by its key.

        :param k: key from which to compute the forgetting.
        :return: the difference between the first and last value encountered
            for k.
        """
        return self.forgetting.result_key(k=k)

    def result(self) -> Dict[int, float]:
        """
        R2Forgetting for all experiences.

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
        return "ExperienceR2Forgetting"


class StreamR2Forgetting(GenericStreamForgetting[TaskAwareR2]):
    """
    The StreamR2Forgetting metric, describing the average evaluation R2 loss
    detected over all experiences observed during training.

    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the R2 result obtained
    after first training on a experience and the R2 result obtained
    on the same experience at the end of successive experiences.

    This metric is computed during the eval phase only.
    """

    def __init__(self):
        """
        Creates an instance of the StreamR2Forgetting metric.
        """

        super().__init__(TaskAwareR2())

    def metric_update(self, strategy):
        self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)

    def metric_result(self, strategy):
        return self._current_metric.result(0)[0]

    def __str__(self):
        return "StreamR2Forgetting"


def r2_forgetting_metrics(*, experience=False, stream=False) -> List[PluginMetric]:
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
        metrics.append(ExperienceR2Forgetting())

    if stream:
        metrics.append(StreamR2Forgetting())

    return metrics


__all__ = [
    "ExperienceRMSEForgetting",
    "StreamRMSEForgetting",
    "rmse_forgetting_metrics",
    "ExperienceR2Forgetting",
    "StreamR2Forgetting",
    "r2_forgetting_metrics",
]
