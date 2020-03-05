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

""" Common training utils. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from avalanche.training.strategies.skeletons.strategy_caffe\
    import StrategyCaffe
import copy
from avalanche.training.utils.caffe_utils import \
    reset_classifier, set_classifier, get_classifier


class NaiveCaffe(StrategyCaffe):
    """
    Naive Strategy: PyTorch implementation.
    """

    def __init__(self, model, optimizer=None, mb_size=256,
                 train_ep=2, multi_head=False, use_cuda=True, preproc=None):

        super(NaiveCaffe, self).__init__(
            model, optimizer, mb_size, train_ep, multi_head, use_cuda, preproc
        )

        # to be filled with {t:params}
        self.heads = {}

    def before_epoch(self):

        pass
        if self.multi_head:
            if self.cur_ep == 0:
                reset_classifier(self.model)

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
                w, b = get_classifier(self.model)
                self.heads[self.cur_train_t] = copy.deepcopy((w, b))
                print("multi-head used: ", self.heads.keys())

    def before_test(self):

        if self.multi_head:
            # save training head for later use
            w, b = get_classifier(self.model)
            self.heads['train'] = copy.deepcopy((w, b))

    def after_test(self):

        if self.multi_head:
            # reposition train head
            w, b = self.heads['train']
            set_classifier(self.model, w, b)

    def before_task_test(self):

        if self.multi_head:
            # reposition right head
            if self.cur_test_t in self.heads.keys():
                w, b = self.heads[self.cur_test_t]
                set_classifier(self.model, w, b)
            else:
                reset_classifier(self.model)

    def after_task_test(self):
        pass
