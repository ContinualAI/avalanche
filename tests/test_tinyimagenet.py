################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
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
from avalanche.benchmarks.scenarios.generic_definitions import Experience


class TinyImagenetTest(unittest.TestCase):
    def test_tinyimagenet_default_loader(self):

        logger = logging.getLogger("avalanche")
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())

        benchmark = SplitTinyImageNet()
        for task_info in benchmark.train_stream:
            self.assertIsInstance(task_info, Experience)

        for task_info in benchmark.test_stream:
            self.assertIsInstance(task_info, Experience)


if __name__ == "__main__":
    unittest.main()
