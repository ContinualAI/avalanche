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

""" CORe50 Tests"""

import unittest

from avalanche.benchmarks.classic import CORe50
from avalanche.benchmarks.scenarios.generic_definitions import Experience


class CORe50Test(unittest.TestCase):
    def test_core50_ni_scenario(self):

        # for now we disable it as it takes a while to download the CORe50
        # dataset.
        pass

        # scenario = CORe50(scenario="ni")
        # for task_info in scenario:
        #     self.assertIsInstance(task_info, IExperience)


if __name__ == '__main__':
    unittest.main()
