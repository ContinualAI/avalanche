import unittest

import torch

from avalanche.benchmarks.scenarios import _split_dataset_by_attribute
from avalanche.benchmarks.scenarios.supervised import (
    class_incremental_benchmark,
    new_instances_benchmark,
    with_classes_timeline,
)
from avalanche.benchmarks.utils import _make_taskaware_tensor_classification_dataset
from tests.unit_tests_utils import dummy_classification_datasets


class ClassIncrementalBenchmark(unittest.TestCase):
    def test_class_incremental_benchmark_basic_num_experiences(self):
        tx = torch.rand(10, 17)
        ty = torch.tensor([1 for _ in range(10)])
        data = _make_taskaware_tensor_classification_dataset(tx, ty)

        tx = torch.rand(10, 15)
        ty = torch.tensor([0 for _ in range(10)])
        data = data.concat(_make_taskaware_tensor_classification_dataset(tx, ty))

        tx = torch.rand(10, 17)
        ty = torch.tensor([2 for _ in range(10)])
        data = data.concat(_make_taskaware_tensor_classification_dataset(tx, ty))

        bm = class_incremental_benchmark(
            datasets_dict={"train": data, "test": data},
            class_order=[2, 1, 0],
            num_experiences=3,
        )
        bm = with_classes_timeline(bm)
        assert len(bm.streams) == 2

        # train_stream - class order is [11, 3, 4]
        stream = bm.train_stream
        assert len(stream) == 3
        assert stream[0].classes_in_this_experience == {2}
        assert len(stream[0].dataset) == 10
        assert stream[1].classes_in_this_experience == {1}
        assert len(stream[1].dataset) == 10
        assert stream[2].classes_in_this_experience == {0}
        assert len(stream[2].dataset) == 10
        assert len(stream[0].dataset) + len(stream[1].dataset) + len(
            stream[2].dataset
        ) == len(data)

        # test_stream - class order is [11, 3, 4]
        stream = bm.test_stream
        assert len(stream) == 3
        assert stream[0].classes_in_this_experience == {2}
        assert len(stream[0].dataset) == 10
        assert stream[1].classes_in_this_experience == {1}
        assert len(stream[1].dataset) == 10
        assert stream[2].classes_in_this_experience == {0}
        assert len(stream[2].dataset) == 10
        assert len(stream[0].dataset) + len(stream[1].dataset) + len(
            stream[2].dataset
        ) == len(data)

    def test_class_incremental_benchmark_basic_num_classes_per_exp(self):
        tx = torch.rand(10, 17)
        ty = torch.tensor([1 for _ in range(10)])
        data = _make_taskaware_tensor_classification_dataset(tx, ty)

        tx = torch.rand(10, 17)
        ty = torch.tensor([0 for _ in range(10)])
        data = data.concat(_make_taskaware_tensor_classification_dataset(tx, ty))

        tx = torch.rand(10, 17)
        ty = torch.tensor([2 for _ in range(10)])
        data = data.concat(_make_taskaware_tensor_classification_dataset(tx, ty))

        bm = class_incremental_benchmark(
            datasets_dict={"train": data, "test": data},
            class_order=[2, 1, 0],
            num_classes_per_exp=[1, 1, 1],
        )
        bm = with_classes_timeline(bm)
        assert len(bm.streams) == 2

        # train_stream - class order is [11, 3, 4]
        stream = bm.train_stream
        assert len(stream) == 3
        assert stream[0].classes_in_this_experience == {2}
        assert stream[1].classes_in_this_experience == {1}
        assert stream[2].classes_in_this_experience == {0}
        assert len(stream[0].dataset) + len(stream[1].dataset) + len(
            stream[2].dataset
        ) == len(data)

        # test_stream - class order is [11, 3, 4]
        stream = bm.test_stream
        assert len(stream) == 3
        assert stream[0].classes_in_this_experience == {2}
        assert stream[1].classes_in_this_experience == {1}
        assert stream[2].classes_in_this_experience == {0}
        assert len(stream[0].dataset) + len(stream[1].dataset) + len(
            stream[2].dataset
        ) == len(data)


class NewInstancesTests(unittest.TestCase):
    def test_ni_balanced(self):
        num_exp, num_classes = 5, 10
        d1, d2 = dummy_classification_datasets(n_classes=num_classes)
        bm = new_instances_benchmark(
            d1,
            d2,
            num_experiences=num_exp,
            shuffle=True,
            seed=1234,
            balance_experiences=True,
        )
        bm = with_classes_timeline(bm)

        data_by_class = _split_dataset_by_attribute(d1, "targets")
        nums_per_class = {}
        for k, v in data_by_class.items():
            nums_per_class[k] = len(v)

        self.assertEqual(5, len(list(bm.train_stream)))
        for exp in bm.train_stream:
            self.assertEqual(10, len(exp.classes_in_this_experience))
            # check class balancing
            data_by_class = _split_dataset_by_attribute(exp.dataset, "targets")
            for k, dd in data_by_class.items():
                assert len(dd) >= (nums_per_class[k] // num_exp)

        _, unique_count = torch.unique(torch.as_tensor(d1.targets), return_counts=True)

        min_exp_size = torch.sum(unique_count // num_exp).item()
        max_exp_size = min_exp_size + num_classes

        pattern_count = 0
        for batch_id, exp in enumerate(bm.train_stream):
            cur_train_set = exp.dataset
            self.assertEqual(batch_id, exp.current_experience)
            self.assertGreaterEqual(len(cur_train_set), min_exp_size)
            self.assertLessEqual(len(cur_train_set), max_exp_size)
            pattern_count += len(cur_train_set)
        self.assertEqual(len(d1), pattern_count)

        self.assertEqual(1, len(bm.test_stream))
        pattern_count = 0
        for batch_id, exp in enumerate(bm.test_stream):
            cur_test_set = exp.dataset
            self.assertEqual(batch_id, exp.current_experience)
            pattern_count += len(cur_test_set)
        self.assertEqual(len(d2), pattern_count)

    def test_ni_unbalanced(self):
        num_exp, num_classes = 5, 10
        d1, d2 = dummy_classification_datasets(n_classes=num_classes)
        bm = new_instances_benchmark(
            d1,
            d2,
            num_experiences=num_exp,
            shuffle=True,
            seed=1234,
            balance_experiences=False,
        )
        bm = with_classes_timeline(bm)

        # TODO: fix pycharm types?
        self.assertEqual(5, len(list(bm.train_stream)))
        for exp in bm.train_stream:
            self.assertEqual(10, len(exp.classes_in_this_experience))

        _, unique_count = torch.unique(torch.as_tensor(d1.targets), return_counts=True)

        min_exp_size = torch.sum(unique_count // num_exp).item()
        max_exp_size = min_exp_size + num_classes

        pattern_count = 0
        for batch_id, exp in enumerate(bm.train_stream):
            cur_train_set = exp.dataset
            self.assertEqual(batch_id, exp.current_experience)
            self.assertGreaterEqual(len(cur_train_set), min_exp_size)
            self.assertLessEqual(len(cur_train_set), max_exp_size)
            pattern_count += len(cur_train_set)
        self.assertEqual(len(d1), pattern_count)

        self.assertEqual(1, len(bm.test_stream))
        pattern_count = 0
        for batch_id, exp in enumerate(bm.test_stream):
            cur_test_set = exp.dataset
            self.assertEqual(batch_id, exp.current_experience)
            pattern_count += len(cur_test_set)
        self.assertEqual(len(d2), pattern_count)
