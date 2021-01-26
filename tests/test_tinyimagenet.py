#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" TinyImagenet Tests"""
import logging
import unittest

from avalanche.benchmarks.classic import SplitTinyImageNet
from avalanche.benchmarks.scenarios.generic_definitions import IStepInfo


class TinyImagenetTest(unittest.TestCase):
    def test_tinyimagenet_default_loader(self):

        logger = logging.getLogger("avalanche")
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())

        scenario = SplitTinyImageNet()
        for task_info in scenario.train_stream:
            self.assertIsInstance(task_info, IStepInfo)

        for task_info in scenario.test_stream:
            self.assertIsInstance(task_info, IStepInfo)


if __name__ == '__main__':
    unittest.main()
