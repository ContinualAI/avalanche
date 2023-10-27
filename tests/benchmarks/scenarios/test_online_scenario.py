import unittest

from avalanche.benchmarks.scenarios.online import OnlineCLScenario
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from tests.unit_tests_utils import dummy_classification_datasets, get_fast_benchmark
from avalanche.benchmarks.scenarios.online import (
    split_online_stream,
    split_continuous_linear_decay_stream,
)


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

    def test_split_online_stream_drop_last(self):
        num_exp, num_classes = 5, 10
        d1, d2 = dummy_classification_datasets(n_classes=num_classes)
        bm = class_incremental_benchmark(
            {"train": d1, "test": d2}, num_experiences=num_exp
        )
        online_train_stream = split_online_stream(
            bm.train_stream, experience_size=10, drop_last=True
        )
        cnt_is_first = 0
        cnt_is_last = 0
        for exp in online_train_stream:
            if exp.is_first_subexp:
                cnt_is_first += 1
            if exp.is_last_subexp:
                cnt_is_last += 1
            assert len(exp.dataset) == 10

        assert exp.is_last_subexp  # final exp should have is_last_subexp == True
        assert cnt_is_last == len(bm.train_stream)
        assert cnt_is_first == len(bm.train_stream)

    def test_split_online_stream_not_drop(self):
        num_exp, num_classes = 5, 10
        d1, d2 = dummy_classification_datasets(n_classes=num_classes)
        bm = class_incremental_benchmark(
            {"train": d1, "test": d2}, num_experiences=num_exp
        )
        online_train_stream = split_online_stream(
            bm.train_stream, experience_size=10, drop_last=False
        )
        cnt_is_first = 0
        cnt_is_last = 0
        for exp in online_train_stream:
            if exp.is_first_subexp:
                cnt_is_first += 1
            if exp.is_last_subexp:
                cnt_is_last += 1

            if not exp.is_last_subexp:
                assert len(exp.dataset) == 10
            else:
                assert len(exp.dataset) <= 10

        assert exp.is_last_subexp  # final exp should have is_last_subexp == True
        assert cnt_is_last == len(bm.train_stream)
        assert cnt_is_first == len(bm.train_stream)

    def test_split_online_stream_continuous_linear_decay(self):
        num_exp, num_classes = 5, 10
        d1, d2 = dummy_classification_datasets(n_classes=num_classes)
        bm = class_incremental_benchmark(
            {"train": d1, "test": d2}, num_experiences=num_exp
        )

        iter_per_virt_epoch = 100
        ocl_stream = split_continuous_linear_decay_stream(
            bm.train_stream,
            experience_size=10,
            iters_per_virtual_epoch=100,
            beta=0.5,
            shuffle=True,
        )

        expected_length = len(bm.train_stream) * iter_per_virt_epoch
        n_exps = 0
        all_classes = set()
        for exp in ocl_stream:
            assert len(exp.dataset) == 10
            n_exps += 1
            all_classes.update(set(exp.dataset.targets))

        # Check the length of the stream
        assert n_exps == expected_length

        # Check if all classes in the dataset are present in the stream
        assert len(all_classes) == num_classes


if __name__ == "__main__":
    unittest.main()
