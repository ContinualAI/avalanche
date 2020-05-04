#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

""" Common training utils. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from avalanche.training.strategies.skeletons.strategy\
    import Strategy
from avalanche.evaluation.eval_protocol import EvalProtocol
from avalanche.evaluation.metrics import ACC
import copy
import torch


class Naive(Strategy):
    """
    Naive Strategy: PyTorch implementation.
    """

    def __init__(self, model, optimizer=None,
                 criterion=torch.nn.CrossEntropyLoss(), mb_size=256,
                 train_ep=2, multi_head=False, use_cuda=True, preproc=None,
                 eval_protocol=EvalProtocol(metrics=[ACC])):

        super(Naive, self).__init__(
            model, optimizer, criterion, mb_size, train_ep, multi_head,
            use_cuda, preproc, eval_protocol
        )

        # to be filled with {t:params}
        self.heads = {}

    def before_train(self):
        pass

    def before_epoch(self):

        if self.multi_head:
            if self.cur_ep == 0:
                shape = self.model.classifier.weight.data.size()
                self.model.classifier = torch.nn.Linear(shape[1], shape[0])

                # here eventual zero-reinit
                # weight_init.xavier_uniform(self.model.classifier.weight)
                # weight_init.uniform(self.model.classifier.weight, 0.0, 0.1)

    def before_iteration(self):
        pass

    def before_weights_update(self):
        pass

    def after_iter_ended(self):
        pass

    def after_epoch_ended(self):

        if self.cur_ep == self.train_ep-1:
            # we have finished training
            if self.multi_head:
                w, b = self.model.classifier.weight, self.model.classifier.bias
                self.heads[self.cur_train_t] = copy.deepcopy((w, b))
                print("multi-head used: ", self.heads.keys())

    def after_train(self):
        pass

    def before_test(self):

        if self.multi_head:
            # save training head for later use
            w, b = self.model.classifier.weight, self.model.classifier.bias
            self.heads['train'] = copy.deepcopy((w, b))

    def after_test(self):

        if self.multi_head:
            # reposition train head
            w, b = self.heads['train']
            self.model.classifier.weight = w
            self.model.classifier.bias = b

    def before_task_test(self):

        if self.multi_head:
            # reposition right head
            if self.cur_test_t in self.heads.keys():
                w, b = self.heads[self.cur_test_t]
                self.model.classifier.weight = w
                self.model.classifier.bias = b
            else:
                shape = self.model.classifier.weight.data.size()
                self.model.classifier = torch.nn.Linear(shape[1], shape[0])

    def after_task_test(self):
        pass
