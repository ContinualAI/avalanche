import unittest

from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from tests.unit_tests_utils import get_fast_benchmark


class OCLTests(unittest.TestCase):
    def test_ocl_scenario(self):
        benchmark = get_fast_benchmark()
        batch_streams = benchmark.streams.values()
        ocl_benchmark = OnlineCLScenario(batch_streams)

        for s in ocl_benchmark.streams():
            print(s.name)
