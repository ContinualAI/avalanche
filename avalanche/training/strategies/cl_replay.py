#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-06-2020                                                              #
# Author(s): Andrea Cossu                                                      #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

from __future__ import absolute_import
from __future__ import division
# Python 2-3 compatible
from __future__ import print_function

from typing import Optional, Sequence

from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.strategies import Naive
from avalanche.training.plugins import ReplayPlugin

from avalanche.evaluation import EvalProtocol
from avalanche.training.skeletons import StrategySkeleton


class Replay(Naive):
    """
    The replay strategy is built on top of the Replay plugin and Naive strategy.
    It mix patterns from the current step with patterns from the previous steps
    in order to mitigate forgetting.
    """

    def __init__(self, model: Module, classifier_field: str,
                 optimizer: Optimizer, criterion: Module,
                 mem_size: int,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 evaluation_protocol: Optional[EvalProtocol] = None,
                 plugins: Optional[Sequence[StrategySkeleton]] = None):
        """
        Creates an instance of the Replay strategy.

        :param model: The model.
        :param classifier_field: The name of the classifier field. Used when
            managing heads in Multi-Task scenarios.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: Replay memory size
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param evaluation_protocol: The evaluation protocol. Defaults to None.
        :param plugins: Plugins to be added 
            (Replay plugin will be added by default). Defaults to None.
        """

        replay_plugin = ReplayPlugin(mem_size=mem_size)
        if plugins is None:
            plugins = [replay_plugin]
        else:
            plugins.append(replay_plugin)

        super(Replay, self).__init__(
            model, classifier_field, optimizer, criterion, train_mb_size,
            train_epochs, test_mb_size, device, evaluation_protocol, plugins)


__all__ = ['Replay']
