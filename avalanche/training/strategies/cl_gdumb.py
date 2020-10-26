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

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.strategies import Naive
from avalanche.training.plugins import GDumbPlugin

from avalanche.training.skeletons import TrainingFlow

from avalanche.evaluation import EvalProtocol
from avalanche.training.skeletons import StrategySkeleton


class GDumb(Naive):
    """
    The GDumb strategy is built on top of the GDumb plugin and Naive strategy.
    It retrain at each step the model with patterns taken solely from
    replay memory. It is up to the user logic to decide whether to 
    reinitialize model at each step or proceed with finetuning.
    Original paper retrain from scratch at each step.
    """

    def __init__(self, model: Module, classifier_field: str,
                 optimizer: Optimizer, criterion: Module,
                 mem_size: int, reinit_model_before_step: bool = True,
                 reinit_function: object = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 evaluation_protocol: Optional[EvalProtocol] = None,
                 plugins: Optional[Sequence[StrategySkeleton]] = None):
        """
        Creates an instance of the GDumb strategy.

        :param model: The model.
        :param classifier_field: The name of the classifier field. Used when
            managing heads in Multi-Task scenarios.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: Replay memory size
        :param reinit_model_before_step: if True, model will be reinitialized 
            before each step according to `reinit_function` or uniformly if
            `reinit_function` is not provided. Default True.
        :param reinit_function: a function which takes as input the model 
            and return the model with weights initialized.
            Used only if `reinit_model_before_step` is True. Default None.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param evaluation_protocol: The evaluation protocol. Defaults to None.
        :param plugins: Plugins to be added 
            (GDumb plugin will be added by default). Defaults to None.
        """

        gdumb_plugin = GDumbPlugin(mem_size=mem_size)
        if plugins is None:
            plugins = [gdumb_plugin]
        else:
            plugins.append(gdumb_plugin)

        super(GDumb, self).__init__(
            model, classifier_field, optimizer, criterion, train_mb_size,
            train_epochs, test_mb_size, device, evaluation_protocol, plugins)

        self.reinit_model_before_step = reinit_model_before_step
        self.reinit_function = reinit_function


    @TrainingFlow
    def before_training(self):
        """
        Reinitialize model before each step, if needed.
        """

        if self.reinit_model_before_step:
            if self.reinit_function is not None:
                self.reinit_function(self.model)
            else:
                self._init_weights()


    def _init_weights(self):
        """
        Default init function
        """
        with torch.no_grad():
            for p in self.model.parameters():
                torch.nn.init.uniform_(p, -0.1, 0.1)


__all__ = ['GDumb']
