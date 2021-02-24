################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-01-2021                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import SupportsFloat

from avalanche.evaluation import Metric


class Sum(Metric[float]):
    """
    The sum metric.

    This utility metric is a general purpose metric that can be used to keep
    track of the sum of a sequence of values.

    Beware that this metric only supports summing numbers and the result is
    always a float value, even when `update` is called by passing `int`s only.
    """

    def __init__(self):
        """
        Creates an instance of the sum metric.

        This metric in its initial state will return a sum value of 0.
        The metric can be updated by using the `update` method while the sum
        can be retrieved using the `result` method.
        """
        super().__init__()
        self.summed: float = 0.0

    def update(self, value: SupportsFloat) -> None:
        """
        Update the running sum given the value.

        :param value: The value to be used to update the sum.
        :return: None.
        """
        self.summed += float(value)

    def result(self) -> float:
        """
        Retrieves the sum.

        Calling this method will not change the internal state of the metric.

        :return: The sum, as a float.
        """
        return self.summed

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self.summed = 0.0


__all__ = ['Sum']
