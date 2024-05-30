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
        class ToyStream:
            def __init__(self):
                self.els = [
                    SimpleNamespace(dataset=list(range(1))),
                    SimpleNamespace(dataset=list(range(2))),
                ]
                self.name = "toy"

            def __getitem__(self, item):
                return self.els[item]

            def __len__(self):
                return len(self.els)

        fake_stream = ToyStream()
        # m = MockMetric([0, 1, 2, 3, 4, 5])
        mc = MetricCollector()
        stream_of_mvals = [[1, 3], [5, 7], [11, 13]]
        for vvv in stream_of_mvals:
            mc.update({"toy/FakeMetric": vvv})

        # time_reduce = None
        v = mc.get("FakeMetric", time_reduce=None, exp_reduce=None, stream=fake_stream)
        np.testing.assert_array_almost_equal(v, stream_of_mvals)
        v = mc.get(
            "FakeMetric", time_reduce=None, exp_reduce="sample_mean", stream=fake_stream
        )
        np.testing.assert_array_almost_equal(
            v, [(1 + 3 * 2) / 3, (5 + 7 * 2) / 3, (11 + 13 * 2) / 3]
        )
        v = mc.get(
            "FakeMetric",
            time_reduce=None,
            exp_reduce="experience_mean",
            stream=fake_stream,
        )
        np.testing.assert_array_almost_equal(
            v, [(1 + 3) / 2, (5 + 7) / 2, (11 + 13) / 2]
        )
        v = mc.get(
            "FakeMetric",
            time_reduce=None,
            exp_reduce="weighted_sum",
            weights=[1, 2],
            stream=fake_stream,
        )
        np.testing.assert_array_almost_equal(
            v, [(1 + 3 * 2), (5 + 7 * 2), (11 + 13 * 2)]
        )

        # time = "last"
        v = mc.get(
            "FakeMetric", time_reduce="last", exp_reduce=None, stream=fake_stream
        )
        np.testing.assert_array_almost_equal(v, [11, 13])
        v = mc.get(
            "FakeMetric",
            time_reduce="last",
            exp_reduce="sample_mean",
            stream=fake_stream,
        )
        self.assertAlmostEqual(v, (11 + 13 * 2) / 3)
        v = mc.get(
            "FakeMetric",
            time_reduce="last",
            exp_reduce="experience_mean",
            stream=fake_stream,
        )
        self.assertAlmostEqual(v, (11 + 13) / 2)
        v = mc.get(
            "FakeMetric",
            time_reduce="last",
            exp_reduce="weighted_sum",
            stream=fake_stream,
            weights=[1, 2],
        )
        self.assertAlmostEqual(v, 11 + 13 * 2)

        # time_reduce = "mean"
        v = mc.get(
            "FakeMetric", time_reduce="mean", exp_reduce=None, stream=fake_stream
        )
        np.testing.assert_array_almost_equal(v, [(1 + 5 + 11) / 3, (3 + 7 + 13) / 3])
        v = mc.get(
            "FakeMetric",
            time_reduce="mean",
            exp_reduce="sample_mean",
            stream=fake_stream,
        )
        self.assertAlmostEqual(v, ((1 + 5 + 11) / 3 + (3 + 7 + 13) / 3 * 2) / 3)
        v = mc.get(
            "FakeMetric",
            time_reduce="mean",
            exp_reduce="experience_mean",
            stream=fake_stream,
        )
        self.assertAlmostEqual(v, ((1 + 5 + 11) / 3 + (3 + 7 + 13) / 3) / 2)
        v = mc.get(
            "FakeMetric",
            time_reduce="mean",
            exp_reduce="weighted_sum",
            stream=fake_stream,
            weights=[1, 2],
        )
        self.assertAlmostEqual(v, ((1 + 3 * 2) + (5 + 7 * 2) + (11 + 13 * 2)) / 3)


if __name__ == "__main__":
    unittest.main()
