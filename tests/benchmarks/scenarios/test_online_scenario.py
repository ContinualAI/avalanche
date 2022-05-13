import unittest

from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from tests.unit_tests_utils import get_fast_benchmark


class OCLTests(unittest.TestCase):
    def test_ocl_scenario_stream(self):
        benchmark = get_fast_benchmark()
        batch_streams = benchmark.streams.values()
        ocl_benchmark = OnlineCLScenario(batch_streams)

        for s in ocl_benchmark.streams.values():
            print(s.name)

    def test_ocl_scenario_experience(self):
        benchmark = get_fast_benchmark()
        batch_streams = benchmark.streams.values()

        for exp in benchmark.train_stream:
            ocl_benchmark = OnlineCLScenario(batch_streams, exp)

            for s in ocl_benchmark.streams.values():
                print(s.name)
