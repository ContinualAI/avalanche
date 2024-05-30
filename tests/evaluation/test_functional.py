""" Metrics Tests"""

import unittest

import torch

from avalanche.evaluation.functional import forgetting
import numpy as np


class FunctionalMetricTests(unittest.TestCase):
    def test_diag_forgetting(self):
        vals = [[1.0, 0.0, 0.0], [0.5, 0.8, 1.0], [0.2, 1.0, 0.5]]
        target = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.8, -0.2, 0.0]]
        fm = forgetting(vals)
        np.testing.assert_array_almost_equal(fm, target)

    def test_non_diag_forgetting(self):
        """non-diagonal matrix requires explicit boundary indices"""
        vals = [[1.0, 0.0, 0.0], [0.5, 0.8, 1.0], [0.5, 0.5, 1.0], [0.2, 1.0, 0.5]]
        target = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.3, 0.0], [0.8, -0.2, 0.0]]
        fm = forgetting(vals, boundary_indices=[0, 1, 3])
        np.testing.assert_array_almost_equal(fm, target)


if __name__ == "__main__":
    unittest.main()
