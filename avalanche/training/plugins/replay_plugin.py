#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 09-10-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

import copy

import torch
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from avalanche.training.skeletons.cl_strategy import StrategySkeleton
from avalanche.training.skeletons.strategy_flow import TrainingFlow


class ReplayPlugin(StrategySkeleton):
    """
    An experience replay plugin that can be plugged in a strategy.

    Instances of this class should be used as strategy plugins.

    This simply handles an external memory filled with randomly selected
    patterns and implements the "adapt_train_dataset" callback to add them to
    the training set.

    The :mem_size: params controls the number of patterns to be stored in the
    external memory. We assume the training set to contain at least this
    number of training data points.
    """
    def __init__(self, mem_size=200):

        super().__init__()

        self.mem_size = mem_size
        self.ext_mem = None

    @TrainingFlow
    def adapt_train_dataset(self, step_id, train_dataset, ext_mem):
        """ Before training we make sure to publish in the namespace a copy
            of :mem_size: randomly selected patterns to be used for replay
            in the next batch and we expand the current training set to
            contain also the data from the external memory. """

        # Additional set of the current batch to be concatenated to the ext.
        # memory at the end of the training
        rm_add = None

        # how many patterns to save for next iter
        h = min(self.mem_size // (step_id + 1), len(train_dataset))

        # We recover it as a mini-batch from the shuffled dataset
        # and we publish it in the namespace
        data_loader = DataLoader(
            train_dataset, batch_size=h, shuffle=True
        )
        rm_add = next(iter(data_loader))
        rm_add = TensorDataset(rm_add[0], rm_add[1])
        self.update_namespace(rm_add=rm_add)

        if step_id > 0:
            # We update the train_dataset concatenating the external memory.
            # We assume the user will shuffle the data when creating the data
            # loader.
            train_dataset = ConcatDataset([train_dataset, ext_mem])
            self.update_namespace(train_dataset=train_dataset)

    @TrainingFlow
    def after_training(self, step_id, rm_add):
        """ After training we update the external memory with the patterns of
         the current training batch/task. """

        # replace patterns in random memory
        ext_mem = self.ext_mem
        if step_id == 0:
            ext_mem = copy.deepcopy(rm_add)
        else:
            idxs_2_replace = torch.randperm(
                len(ext_mem.tensors[0]))[:len(rm_add.tensors[0])]
            for j, idx in enumerate(idxs_2_replace):
                ext_mem.tensors[0][idx] = rm_add.tensors[0][j]
                ext_mem.tensors[1][idx] = rm_add.tensors[1][j]
        self.ext_mem = ext_mem


__all__ = ['ReplayPlugin']
