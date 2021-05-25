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
import os


from avalanche.benchmarks.classic import CORe50


class CORe50Test(unittest.TestCase):
    def test_core50_ni_scenario(self):

        if "FAST_TEST" in os.environ:
            pass
        else:
            scenario = CORe50(scenario="ni")
            for task_info in scenario:
                pass

    def test_core50_nc_scenario(self):
        if "FAST_TEST" in os.environ:
            pass
        else:
            benchmark_instance = CORe50(scenario='nc')
            self.assertEqual(1, len(benchmark_instance.test_stream))

            classes_in_test = benchmark_instance.\
                classes_in_experience['test'][0]
            self.assertSetEqual(
                set(range(50)),
                set(classes_in_test))


if __name__ == '__main__':
    unittest.main()
