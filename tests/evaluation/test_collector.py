""" Metrics Tests"""

import unittest
from types import SimpleNamespace

import torch

from avalanche.evaluation.collector import MetricCollector
from avalanche.evaluation.functional import forgetting
import numpy as np


class MetricCollectorTests(unittest.TestCase):
    def test_time_reduce(self):
        # we just need an object with __len__
        fake_stream = [
            SimpleNamespace(dataset=list(range(1))),
            SimpleNamespace(dataset=list(range(2))),
        ]
        # m = MockMetric([0, 1, 2, 3, 4, 5])
        mc = MetricCollector(fake_stream)
        stream_of_mvals = [
            [1, 3],
            [5, 7],
            [11, 13]
        ]
        for vvv in stream_of_mvals:
            mc.update({"FakeMetric": vvv})

        # time_reduce = None
        v = mc.get("FakeMetric", time_reduce=None, exp_reduce=None)
        np.testing.assert_array_almost_equal(v, stream_of_mvals)
        v = mc.get("FakeMetric", time_reduce=None, exp_reduce="sample_mean")
        np.testing.assert_array_almost_equal(v, [(1+3*2)/3, (5+7*2)/3, (11+13*2)/3])
        v = mc.get("FakeMetric", time_reduce=None, exp_reduce="experience_mean")
        np.testing.assert_array_almost_equal(v, [(1+3)/2, (5+7)/2, (11+13)/2])

        # time = "last"
        v = mc.get("FakeMetric", time_reduce="last", exp_reduce=None)
        np.testing.assert_array_almost_equal(v, [11, 13])
        v = mc.get("FakeMetric", time_reduce="last", exp_reduce="sample_mean")
        self.assertAlmostEqual(v, (11+13*2)/3)
        v = mc.get("FakeMetric", time_reduce="last", exp_reduce="experience_mean")
        self.assertAlmostEqual(v, (11+13)/2)

        # time_reduce = "mean"
        v = mc.get("FakeMetric", time_reduce="mean", exp_reduce=None)
        np.testing.assert_array_almost_equal(v, [(1+5+11)/3, (3+7+13)/3])
        v = mc.get("FakeMetric", time_reduce="mean", exp_reduce="sample_mean")
        self.assertAlmostEqual(v, ((1+5+11)/3 + (3+7+13)/3*2) / 3)
        v = mc.get("FakeMetric", time_reduce="mean", exp_reduce="experience_mean")
        self.assertAlmostEqual(v, ((1+5+11)/3 + (3+7+13)/3) / 2)


if __name__ == "__main__":
    unittest.main()
