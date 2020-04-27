#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. ContinualAI. All rights reserved.                        #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-07-2019                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" Rehearsal Strategy Implementation """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from avalanche.training.strategies.skeletons.strategy \
    import Strategy
from avalanche.evaluation.eval_protocol import EvalProtocol
from avalanche.evaluation.metrics import ACC
from avalanche.training.utils import pad_data, shuffle_in_unison
import torch
import numpy as np
import copy


class Rehearsal(Strategy):
    """
    Naive Strategy: PyTorch implementation.
    """

    def __init__(self, model, optimizer=None,
                 criterion=torch.nn.CrossEntropyLoss(), mb_size=256,
                 train_ep=2, multi_head=False, use_cuda=True, preproc=None,
                 eval_protocol=EvalProtocol(metrics=[ACC]), rm_sz=1500,
                 replace=True):

        super(Rehearsal, self).__init__(
            model, optimizer, criterion, mb_size, train_ep, multi_head,
            use_cuda, preproc, eval_protocol
        )

        # rehearsal parameters
        self.rm_sz = rm_sz
        self.replace = replace

        # to be updated
        self.rm = None
        self.rm_add = None
        self.h = 0

    def before_train(self):
        pass

    def preproc_batch_data(self, x, y, t):

        if self.preproc:
            x = self.preproc(x)

        # saving patterns for next iter
        h = min(self.rm_sz // (self.batch_processed + 1), x.shape[0])
        idxs_cur = np.random.choice(
            x.shape[0], h, replace=False
        )
        self.rm_add = [x[idxs_cur], y[idxs_cur]]

        print("rm_add size", self.rm_add[0].shape[0])

        # adding eventual replay patterns to the current batch
        if self.batch_processed > 0:
            print("rm size", self.rm[0].shape[0])
            x = np.concatenate((x, self.rm[0]))
            y = np.concatenate((y, self.rm[1]))

        (x, y), it_x_ep = pad_data([x, y], self.mb_size) 
        shuffle_in_unison([x, y], in_place=True)

        return  x, y, it_x_ep

    def before_epoch(self):
        pass

    def before_iteration(self):
        pass

    def before_weights_update(self):
        pass

    def after_iter_ended(self):
        pass

    def after_epoch_ended(self):
        pass

    def after_train(self):

        if self.replace:
            # replace patterns in random memory
            if self.batch_processed == 0:
                self.rm = copy.deepcopy(self.rm_add)
            else:
                idxs_2_replace = np.random.choice(
                    self.rm[0].shape[0], self.h, replace=False
                )
                for j, idx in enumerate(idxs_2_replace):
                    self.rm[0][idx] = copy.deepcopy(self.rm_add[0][j])
                    self.rm[1][idx] = copy.deepcopy(self.rm_add[1][j])
        else:
            self.rm[0] = np.concatenate((self.rm_add[0], self.rm[0]))
            self.rm[1] = np.concatenate((self.rm_add[1], self.rm[1]))

    def before_test(self):
        pass

    def after_test(self):
        pass

    def before_task_test(self):
        pass

    def after_task_test(self):
        pass
