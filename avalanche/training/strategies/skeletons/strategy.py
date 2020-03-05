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


class Strategy(object):

    def __init__(self):
        pass

    def train(self, x, y, t):
        raise NotImplemented

    def test(self, test_set):
        raise NotImplemented

    def before_epoch(self):
        raise NotImplemented

    def before_iteration(self):
        raise NotImplemented

    def before_weights_update(self):
        raise NotImplemented

    def after_iter_ended(self):
        raise NotImplemented

    def after_epoch_ended(self):
        raise NotImplemented

    def before_test(self):
        raise NotImplemented

    def after_test(self):
        raise NotImplemented

    def before_task_test(self):
        raise NotImplemented

    def after_task_test(self):
        raise NotImplemented


