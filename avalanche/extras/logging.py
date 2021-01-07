#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 25-11-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" This module handles all the functionalities related to the logging of
Avalanche experiments. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import logging

from PIL.Image import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from avalanche.evaluation.metric_results import AlternativeValues, MetricValue


class TensorboardLogging(object):

    def __init__(self, tb_logdir=""):
        """
        TensorboardLogging is a simple class to handle the interface with
        the tensorboard API offered in Pytorch.

        """
        self.writer = SummaryWriter(tb_logdir)


class Logger:
    """ Main Logger class. """

    log = None
    tb_logging = None

    def __init__(self, log_dir="./logs/", tb_logdir_name="tb_data",
                 logging_level=logging.INFO, additional_handlers=None):

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger = logging.getLogger("avalanche")
        logger.setLevel(logging_level)
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.FileHandler(
            os.path.join(log_dir, 'logfile.log'))
        )

        if additional_handlers is not None:
            for hl in additional_handlers:
                logger.addHandler(hl)

        self.tb_logging = TensorboardLogging(
            tb_logdir=os.path.join(log_dir, tb_logdir_name)
        )

        self.log = logger

    def log_metric(self, metric_value: MetricValue):
        name = metric_value.name
        value = metric_value.value

        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(Image, float, int)

        if not isinstance(value, (Image, float, int)):
            # Unsupported type
            return

        if isinstance(value, Image):
            self.tb_logging.writer.add_image(
                name, ToTensor()(value),
                global_step=metric_value.x_plot)
        elif isinstance(value, (float, int)):
            self.tb_logging.writer.add_scalar(
                name, value, global_step=metric_value.x_plot)
