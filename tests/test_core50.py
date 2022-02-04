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
from tests.unit_tests_utils import FAST_TEST, is_github_action


class CORe50Test(unittest.TestCase):
    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_core50_ni_benchmark(self):
        benchmark = CORe50(scenario="ni")
        for experience in benchmark.train_stream:
            pass

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_core50_nc_benchmark(self):
        benchmark_instance = CORe50(scenario="nc")
        self.assertEqual(1, len(benchmark_instance.test_stream))

        classes_in_test = benchmark_instance.classes_in_experience["test"][0]
        self.assertSetEqual(set(range(50)), set(classes_in_test))


if __name__ == "__main__":
    unittest.main()
