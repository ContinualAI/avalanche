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

import unittest

from avalanche.benchmarks.classic import CTinyImageNet
from avalanche.benchmarks.scenarios.generic_definitions import IStepInfo


class TinyImagenetTest(unittest.TestCase):
    def test_tinyimagenet_default_loader(self):

        scenario = CTinyImageNet()
        for task_info in scenario:
            self.assertIsInstance(task_info, IStepInfo)


if __name__ == '__main__':
    unittest.main()
