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

from avalanche.evaluation.metric_definitions import Metric

class ForwardTransfer(Metric[Union[float, None, Dict[int, float]]]):
    """
        The standalone Forward Transfer metric.
        This metric returns the forward transfer relative to a specific key.
        Alternatively, this metric returns a dict in which each key is associated
        to the forward transfer.
        Forward transfer is computed as the difference between the value recorded for a specific key up until the immediately preceding task, and random initialization of the model before training
        The value associated to a key can be update with the `update` method.

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
        The previous task value detected for each key
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
        Forward transfer is not returned for the last task.

        :param k: the key for which returning forward transfer. If k is None,
            forward transfer will be returned for all keys except the last one.

        :return: the difference between the previous task key value and the key at random initialization.
        """

        forward_transfer = {}
        if k is not None:
            if k in self.previous:
                return self.previous[k] - self.initial[k]
            else:
                return None

        previous_keys = set(self.previous.keys())
        for k in self.previous.keys():
            forward_transfer[k] = self.previous[k] - self.initial[k]

        return forward_transfer

    def reset_previous(self) -> None:
        self.previous: Dict[int, float] = dict()

    def reset(self) -> None:
        self.initial: Dict[int, float] = dict()
        self.previous: Dict[int, float] = dict()