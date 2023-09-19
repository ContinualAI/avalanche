import unittest

import torch

from avalanche.benchmarks import with_classes_timeline, ni_benchmark
from avalanche.benchmarks.scenarios.supervised_scenario import class_incremental_benchmark, new_instances_benchmark
from avalanche.benchmarks.utils import make_tensor_classification_dataset
from unit_tests_utils import dummy_classification_datasets


class ClassIncrementalBenchmark(unittest.TestCase):
    def test_class_incremental_benchmark_basic_num_experiences(self):
        tx = torch.rand(10, 17)
        ty = torch.tensor([1 for _ in range(10)])
        data = make_tensor_classification_dataset(tx, ty)

        tx = torch.rand(10, 15)
        ty = torch.tensor([0 for _ in range(10)])
        data = data.concat(make_tensor_classification_dataset(tx, ty))

        tx = torch.rand(10, 17)
        ty = torch.tensor([2 for _ in range(10)])
        data = data.concat(make_tensor_classification_dataset(tx, ty))

        bm = class_incremental_benchmark(
            datasets_dict={"train": data, "test": data},
            class_order=[2, 1, 0],
            num_experiences=3
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
        assert len(stream[0].dataset) + len(stream[1].dataset) + len(stream[2].dataset) == len(data)

        # test_stream - class order is [11, 3, 4]
        stream = bm.test_stream
        assert len(stream) == 3
        assert stream[0].classes_in_this_experience == {2}
        assert len(stream[0].dataset) == 10
        assert stream[1].classes_in_this_experience == {1}
        assert len(stream[1].dataset) == 10
        assert stream[2].classes_in_this_experience == {0}
        assert len(stream[2].dataset) == 10
        assert len(stream[0].dataset) + len(stream[1].dataset) + len(stream[2].dataset) == len(data)

    def test_class_incremental_benchmark_basic_num_classes_per_exp(self):
        tx = torch.rand(10, 17)
        ty = torch.tensor([1 for _ in range(10)])
        data = make_tensor_classification_dataset(tx, ty)

        tx = torch.rand(10, 17)
        ty = torch.tensor([0 for _ in range(10)])
        data = data.concat(make_tensor_classification_dataset(tx, ty))

        tx = torch.rand(10, 17)
        ty = torch.tensor([2 for _ in range(10)])
        data = data.concat(make_tensor_classification_dataset(tx, ty))

        bm = class_incremental_benchmark(
            datasets_dict={"train": data, "test": data},
            class_order=[2, 1, 0],
            num_classes_per_exp=[1, 1, 1]
        )
        bm = with_classes_timeline(bm)
        assert len(bm.streams) == 2

        # train_stream - class order is [11, 3, 4]
        stream = bm.train_stream
        assert len(stream) == 3
        assert stream[0].classes_in_this_experience == {2}
        assert stream[1].classes_in_this_experience == {1}
        assert stream[2].classes_in_this_experience == {0}
        assert len(stream[0].dataset) + len(stream[1].dataset) + len(stream[2].dataset) == len(data)

        # test_stream - class order is [11, 3, 4]
        stream = bm.test_stream
        assert len(stream) == 3
        assert stream[0].classes_in_this_experience == {2}
        assert stream[1].classes_in_this_experience == {1}
        assert stream[2].classes_in_this_experience == {0}
        assert len(stream[0].dataset) + len(stream[1].dataset) + len(stream[2].dataset) == len(data)


class NewInstancesTests(unittest.TestCase):
    def test_ni_sit_single_dataset(self):
        d1, d2 = dummy_classification_datasets()
        bm = new_instances_benchmark(
            d1, d2,
            num_experiences=5,
            shuffle=True,
            seed=1234,
            balance_experiences=True,
        )

        # TODO: fix pycharm types?
        self.assertEqual(5, len(bm.train_stream))
        for exp in bm.train_stream:
            self.assertEqual(10, len(exp.classes_in_this_experience))

        _, unique_count = torch.unique(
            torch.as_tensor(d1.targets), return_counts=True
        )

        min_batch_size = torch.sum(unique_count // bm.n_experiences).item()
        max_batch_size = min_batch_size + bm.n_classes

        pattern_count = 0
        batch_info: NIExperience
        for batch_id, batch_info in enumerate(bm.train_stream):
            cur_train_set = batch_info.dataset
            t = batch_info.task_label
            self.assertEqual(0, t)
            self.assertEqual(batch_id, batch_info.current_experience)
            self.assertGreaterEqual(len(cur_train_set), min_batch_size)
            self.assertLessEqual(len(cur_train_set), max_batch_size)
            pattern_count += len(cur_train_set)
        self.assertEqual(len(d1), pattern_count)

        self.assertEqual(1, len(bm.test_stream))
        pattern_count = 0
        for batch_id, batch_info in enumerate(bm.test_stream):
            cur_test_set = batch_info.dataset
            t = batch_info.task_label
            self.assertEqual(0, t)
            self.assertEqual(batch_id, batch_info.current_experience)
            pattern_count += len(cur_test_set)
        self.assertEqual(len(d2), pattern_count)

    def test_ni_sit_single_dataset_fixed_assignment(self):
        mnist_train = MNIST(
            root=default_dataset_location("mnist"),
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=default_dataset_location("mnist"),
            train=False,
            download=True,
        )
        ni_benchmark_reference = ni_benchmark(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234
        )

        reference_assignment = ni_benchmark_reference.train_exps_patterns_assignment

        my_ni_benchmark = ni_benchmark(
            mnist_train,
            mnist_test,
            5,
            shuffle=True,
            seed=4321,
            fixed_exp_assignment=reference_assignment,
        )

        self.assertEqual(
            ni_benchmark_reference.n_experiences, my_ni_benchmark.n_experiences
        )

        self.assertEqual(
            ni_benchmark_reference.train_exps_patterns_assignment,
            my_ni_benchmark.train_exps_patterns_assignment,
        )

        self.assertEqual(
            ni_benchmark_reference.exp_structure, my_ni_benchmark.exp_structure
        )

    def test_ni_sit_multi_dataset_merge(self):
        split_mapping = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        mnist_train = MNIST(
            root=default_dataset_location("mnist"),
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=default_dataset_location("mnist"),
            train=False,
            download=True,
        )

        train_part1 = make_nc_transformation_subset(mnist_train, None, None, range(5))
        train_part2 = make_nc_transformation_subset(
            mnist_train, None, None, range(5, 10)
        )
        train_part2 = classification_subset(train_part2, class_mapping=split_mapping)

        test_part1 = make_nc_transformation_subset(mnist_test, None, None, range(5))
        test_part2 = make_nc_transformation_subset(mnist_test, None, None, range(5, 10))
        test_part2 = classification_subset(test_part2, class_mapping=split_mapping)
        my_ni_benchmark = ni_benchmark(
            [train_part1, train_part2],
            [test_part1, test_part2],
            5,
            shuffle=True,
            seed=1234,
            balance_experiences=True,
        )

        self.assertEqual(5, my_ni_benchmark.n_experiences)
        self.assertEqual(10, my_ni_benchmark.n_classes)
        for batch_id in range(5):
            self.assertEqual(
                10,
                len(my_ni_benchmark.classes_in_experience["train"][batch_id]),
            )

        all_classes = set()
        for batch_id in range(5):
            all_classes.update(my_ni_benchmark.classes_in_experience["train"][batch_id])

        self.assertEqual(10, len(all_classes))

    def test_ni_sit_slicing(self):
        mnist_train = MNIST(
            root=default_dataset_location("mnist"),
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=default_dataset_location("mnist"),
            train=False,
            download=True,
        )
        my_ni_benchmark = ni_benchmark(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234
        )

        experience: NIExperience
        for batch_id, experience in enumerate(my_ni_benchmark.train_stream):
            self.assertEqual(batch_id, experience.current_experience)
            self.assertIsInstance(experience, NIExperience)

        self.assertEqual(1, len(my_ni_benchmark.test_stream))
        for batch_id, experience in enumerate(my_ni_benchmark.test_stream):
            self.assertEqual(batch_id, experience.current_experience)
            self.assertIsInstance(experience, NIExperience)

        iterable_slice = [3, 4, 1]
        sliced_stream = my_ni_benchmark.train_stream[iterable_slice]
        self.assertIsInstance(sliced_stream, ClassificationStream)
        self.assertEqual(len(iterable_slice), len(sliced_stream))
        self.assertEqual("train", sliced_stream.name)

        for batch_id, experience in enumerate(sliced_stream):
            self.assertEqual(iterable_slice[batch_id], experience.current_experience)
            self.assertIsInstance(experience, NIExperience)

        with self.assertRaises(IndexError):
            # The test stream only has one element (the complete test set)
            sliced_stream = my_ni_benchmark.test_stream[iterable_slice]

        iterable_slice = [0, 0, 0]
        sliced_stream = my_ni_benchmark.test_stream[iterable_slice]
        self.assertIsInstance(sliced_stream, ClassificationStream)
        self.assertEqual(len(iterable_slice), len(sliced_stream))
        self.assertEqual("test", sliced_stream.name)

        for batch_id, experience in enumerate(sliced_stream):
            self.assertEqual(iterable_slice[batch_id], experience.current_experience)
            self.assertIsInstance(experience, NIExperience)
