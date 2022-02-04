################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-06-2021                                                             #
# Author(s): Timm Hess                                                         #
# E-mail: hess@ccc.cs.uni-frankfurt.de                                         #
# Website: www.continualai.org                                                 #
################################################################################


"""Endless Continual Learning Simulator Dataset Tests"""

import unittest
import os

from avalanche.benchmarks.classic import EndlessCLSim
from tests.unit_tests_utils import FAST_TEST, is_github_action


class EndlessCLSimTest(unittest.TestCase):
    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_endless_cl_classification(self):

        if "FAST_TEST" in os.environ:
            pass
        else:
            # "Classes"
            scenario = EndlessCLSim(
                scenario="Classes",
                sequence_order=None,
                task_order=None,
                semseg=False,
                dataset_root=None,
            )
            for experience in scenario.train_stream:
                pass

            # Illumination
            scenario = EndlessCLSim(
                scenario="Illumination",
                sequence_order=None,
                task_order=None,
                semseg=False,
                dataset_root=None,
            )
            for experience in scenario.train_stream:
                pass

            # Weather
            scenario = EndlessCLSim(
                scenario="Weather",
                sequence_order=None,
                task_order=None,
                semseg=False,
                dataset_root=None,
            )
            for experience in scenario.train_stream:
                pass
        return

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_endless_cl_video(self):
        if "FAST_TEST" in os.environ:
            pass
        else:
            # "Classes"
            scenario = EndlessCLSim(
                scenario="Classes",
                sequence_order=None,
                task_order=None,
                semseg=True,
                dataset_root="/data/avalanche",
            )
            for experience in scenario.train_stream:
                pass

            # Illumination
            scenario = EndlessCLSim(
                scenario="Illumination",
                sequence_order=None,
                task_order=None,
                semseg=True,
                dataset_root=None,
            )
            for experience in scenario.train_stream:
                pass

            # Weather
            scenario = EndlessCLSim(
                scenario="Weather",
                sequence_order=None,
                task_order=None,
                semseg=True,
                dataset_root=None,
            )
            for experience in scenario.train_stream:
                pass
        return


if __name__ == "__main__":
    unittest.main()
