#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

""" Tensorboard Object Class to control logging over it. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from torch.utils.tensorboard import SummaryWriter


class TensorboardLogging(object):

    def __init__(self, tb_logdir=""):
        """
        TensorboardLogging is a simple class to handle the interface with
        the tensorboard API offered in Pytorch.

        """
        self.writer = SummaryWriter(tb_logdir)
