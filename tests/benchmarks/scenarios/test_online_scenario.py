import unittest

from avalanche.benchmarks.scenarios.online import OnlineCLScenario
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from tests.unit_tests_utils import dummy_classification_datasets, get_fast_benchmark
from avalanche.benchmarks.scenarios.online import split_online_stream


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

    def test_split_online_stream(self):
        num_exp, num_classes = 5, 10
        d1, d2 = dummy_classification_datasets(n_classes=num_classes)
        bm = class_incremental_benchmark({'train': d1, 'test': d2}, num_experiences=num_exp)
        online_train_stream = split_online_stream(bm.train_stream, experience_size=10, drop_last=True)
        for exp in online_train_stream:
            assert len(exp.dataset) == 10


if __name__ == "__main__":
    unittest.main()
