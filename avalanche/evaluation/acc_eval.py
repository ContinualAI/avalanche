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

""" Common metrics for CL. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from .metrics import ACC


class AccEvalProtocol(object):

    def __init__(self):

        self.metric = ACC()
        # to be updated
        self.cur_acc = {}
        self.global_step = 0
        self.cur_classes = None
        self.prev_acc_x_class = {}

    def get_results(self, true_y, y_hat, train_t, test_t):
        """ Compute results based on accuracy """

        results = {}
        results[ACC] = self.metric.compute(true_y, y_hat)

        self.global_step += 1

        return results

    def update_tb_test(self, res, step):
        """ Function to update tensorboard """
        pass

    def update_tb_train(self, loss, acc, step, encountered_class, t):
        """ Function to update tensorboard """
        pass
