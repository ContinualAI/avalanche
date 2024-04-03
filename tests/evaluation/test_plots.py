""" Metrics Tests"""

import unittest

from avalanche.evaluation.plot_utils import plot_metric_matrix


class PlotMetricTests(unittest.TestCase):
    def test_basic(self):
        """just checking that the code runs without errors."""
        mvals = [
            [0.76370887, 0.43307481, 0.30893094, 0.25093633, 0.2075],
            [0.74177468, 0.82286715, 0.55876494, 0.42234707, 0.341],
            [0.57178465, 0.65048787, 0.7563081, 0.56941323, 0.4562],
            [0.65802592, 0.52339254, 0.58947543, 0.68339576, 0.5474],
            [0.65802592, 0.52339254, 0.58947543, 0.68339576, 0.5474],
            [0.4995015, 0.58819114, 0.57320717, 0.62322097, 0.6925],
        ]
        plot_metric_matrix(mvals, title="Accuracy - Train")


if __name__ == "__main__":
    unittest.main()
