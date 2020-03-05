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

""" Common metrics for CL. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from .metrics import ACC, CF, RAMU


class EvalProtocol(object):

    def __init__(self, metrics=[ACC]):

        self.metrics = []
        for metric in metrics:
            self.metrics.append(metric())
        self.cur_acc = {}

    def get_results(self, true_y, y_hat, train_t, test_t):
        """ Compute results based on accuracy """

        results = {}

        for metric in self.metrics:
            if isinstance(metric, ACC):
                results[ACC] = metric.compute(true_y, y_hat)
            elif isinstance(metric, CF):
                results[CF] = metric.compute(true_y, y_hat, train_t, test_t)
            elif isinstance(metric, RAMU):
                results[RAMU] = metric.compute(train_t)

        return results


