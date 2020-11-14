#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-06-2020                                                              #
# Author(s): Andrea Cossu                                                #
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

from avalanche.evaluation import EvalProtocol
from avalanche.training.skeletons import TrainingFlow
from avalanche.training.old_strategies import Naive
from avalanche.benchmarks.scenarios.generic_definitions import IStepInfo
from avalanche.training.utils import ConcatDatasetWithTargets
from avalanche.training.skeletons import StrategySkeleton


class Cumulative(Naive):
    """
    A Cumulative strategy in which, at each step (or task), the model
    is trained with all the data encountered so far. Therefore, at each step,
    the model is trained in a MultiTask scenario.
    The strategy has a high memory and computational cost.
    """

    def __init__(self, model: Module, classifier_field: str,
                 optimizer: Optimizer, criterion: Module,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 evaluation_protocol: Optional[EvalProtocol] = None,
                 plugins: Optional[Sequence[StrategySkeleton]] = None):

        super(Cumulative, self).__init__(
            model, classifier_field, optimizer, criterion, train_mb_size,
            train_epochs, test_mb_size, device, evaluation_protocol, plugins)

    @TrainingFlow
    def make_train_dataset(self, step_info: IStepInfo):
        """
        Returns the training dataset, given the step_info instance.
        The dataset is composed by all datasets encountered so far.

        This is a part of the training flow. Sets the train_dataset namespace
        value.

        :param step_info: The step info instance, as returned from the CL
            scenario.
        :return: The training dataset.
        """

        train_dataset = step_info.cumulative_training_sets()
        
        train_dataset = ConcatDatasetWithTargets(
            [el[0] for el in train_dataset]
        )

        self.update_namespace(train_dataset=train_dataset)
        return train_dataset


__all__ = ['Cumulative']
