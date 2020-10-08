#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-06-2020                                                              #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

from __future__ import absolute_import
from __future__ import division
# Python 2-3 compatible
from __future__ import print_function

from typing import Optional

from torch.nn import Module
from torch.optim.optimizer import Optimizer

from avalanche.evaluation import EvalProtocol
from avalanche.training.skeletons.deep_learning_strategy import \
    MTDeepLearningStrategy
from avalanche.training.skeletons.strategy_flow import TrainingFlow, TestingFlow


class Naive(MTDeepLearningStrategy):
    """
    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(self, model: Module, classifier_field: str,
                 optimizer: Optimizer, criterion: Module,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 evaluation_protocol: Optional[EvalProtocol] = None):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param classifier_field: The name of the classifier field. Used when
            managing heads in Multi-Task scenarios.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param evaluation_protocol: The evaluation protocol. Defaults to None.
        """
        super(Naive, self).__init__(
            model, classifier_field, train_mb_size=train_mb_size,
            train_epochs=train_epochs, test_mb_size=test_mb_size,
            evaluation_protocol=evaluation_protocol, device=device)

        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model

    @TrainingFlow
    def training_epoch(self, model: Module, train_data_loader,
                       optimizer: Optimizer, criterion: Module, device=None):
        # Number of iterations to run
        epoch_iterations = len(train_data_loader)
        self.update_namespace(epoch_iterations=epoch_iterations)

        # Move model to device
        self.model = model.to(device)
        model = self.model

        # Run an epoch
        for iteration, (train_mb_x, train_mb_y) in enumerate(train_data_loader):
            # Publish some relevant data to the global namespace
            self.update_namespace(iteration=iteration,
                                  train_mb_x=train_mb_x,
                                  train_mb_y=train_mb_y)
            optimizer.zero_grad()

            # Iteration begins
            self.before_training_iteration()

            # Move mini-batch data to device
            train_mb_x = train_mb_x.to(device)
            train_mb_y = train_mb_y.to(device)

            # Forward
            self.before_forward()
            logits = model(train_mb_x)
            self.update_namespace(logits=logits.detach().cpu())
            self.after_forward()

            # Loss
            loss = criterion(logits, train_mb_y)
            self.update_namespace(loss=loss.detach().cpu())

            # Backward
            self.before_backward()
            loss.backward()
            self.after_backward()

            # Update
            self.before_update()
            optimizer.step()
            self.after_update()

            # Iteration end
            self.after_training_iteration()

    @TestingFlow
    def testing_epoch(self, model: Module, test_data_loader,
                      criterion: Module, device=None):
        epoch_iterations = len(test_data_loader)
        self.model = model = model.to(device)
        self.update_namespace(epoch_iterations=epoch_iterations)

        for iteration, (test_mb_x, test_mb_y) in enumerate(test_data_loader):
            self.update_namespace(iteration=iteration,
                                  test_mb_x=test_mb_x,
                                  test_mb_y=test_mb_y)
            # Iteration begins
            self.before_test_iteration()

            # Move mini-batch data to device
            test_mb_x = test_mb_x.to(device)
            test_mb_y = test_mb_y.to(device)

            # Forward
            self.before_test_forward()
            test_logits = model(test_mb_x)
            self.update_namespace(test_logits=test_logits.detach().cpu())
            self.after_test_forward()

            # Loss
            test_loss = criterion(test_logits, test_mb_y)
            self.update_namespace(test_loss=test_loss.detach().cpu())

            # Iteration end
            self.after_test_iteration()


__all__ = ['Naive']
