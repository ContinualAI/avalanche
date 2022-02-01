import unittest

from os.path import expanduser

import torch
from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios import NIExperience, GenericScenarioStream
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.benchmarks.scenarios.new_classes.nc_utils import (
    make_nc_transformation_subset,
)
from avalanche.benchmarks import ni_benchmark


class NISITTests(unittest.TestCase):
    def test_ni_sit_single_dataset(self):
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
        )
        my_ni_benchmark = ni_benchmark(
            mnist_train,
            mnist_test,
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

        _, unique_count = torch.unique(
            torch.as_tensor(mnist_train.targets), return_counts=True
        )

        min_batch_size = torch.sum(
            unique_count // my_ni_benchmark.n_experiences
        ).item()
        max_batch_size = min_batch_size + my_ni_benchmark.n_classes

        pattern_count = 0
        batch_info: NIExperience
        for batch_id, batch_info in enumerate(my_ni_benchmark.train_stream):
            cur_train_set = batch_info.dataset
            t = batch_info.task_label
            self.assertEqual(0, t)
            self.assertEqual(batch_id, batch_info.current_experience)
            self.assertGreaterEqual(len(cur_train_set), min_batch_size)
            self.assertLessEqual(len(cur_train_set), max_batch_size)
            pattern_count += len(cur_train_set)
        self.assertEqual(len(mnist_train), pattern_count)

        self.assertEqual(1, len(my_ni_benchmark.test_stream))
        pattern_count = 0
        for batch_id, batch_info in enumerate(my_ni_benchmark.test_stream):
            cur_test_set = batch_info.dataset
            t = batch_info.task_label
            self.assertEqual(0, t)
            self.assertEqual(batch_id, batch_info.current_experience)
            pattern_count += len(cur_test_set)
        self.assertEqual(len(mnist_test), pattern_count)

    def test_ni_sit_single_dataset_fixed_assignment(self):
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
        )
        ni_benchmark_reference = ni_benchmark(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234
        )

        reference_assignment = (
            ni_benchmark_reference.train_exps_patterns_assignment
        )

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

    def test_ni_sit_single_dataset_reproducibility_data(self):
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
        )
        ni_benchmark_reference = ni_benchmark(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234
        )

        rep_data = ni_benchmark_reference.get_reproducibility_data()

        my_ni_benchmark = ni_benchmark(
            mnist_train, mnist_test, 0, reproducibility_data=rep_data
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
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
        )

        train_part1 = make_nc_transformation_subset(
            mnist_train, None, None, range(5)
        )
        train_part2 = make_nc_transformation_subset(
            mnist_train, None, None, range(5, 10)
        )
        train_part2 = AvalancheSubset(train_part2, class_mapping=split_mapping)

        test_part1 = make_nc_transformation_subset(
            mnist_test, None, None, range(5)
        )
        test_part2 = make_nc_transformation_subset(
            mnist_test, None, None, range(5, 10)
        )
        test_part2 = AvalancheSubset(test_part2, class_mapping=split_mapping)
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
            all_classes.update(
                my_ni_benchmark.classes_in_experience["train"][batch_id]
            )

        self.assertEqual(10, len(all_classes))

    def test_ni_sit_slicing(self):
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
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
        self.assertIsInstance(sliced_stream, GenericScenarioStream)
        self.assertEqual(len(iterable_slice), len(sliced_stream))
        self.assertEqual("train", sliced_stream.name)

        for batch_id, experience in enumerate(sliced_stream):
            self.assertEqual(
                iterable_slice[batch_id], experience.current_experience
            )
            self.assertIsInstance(experience, NIExperience)

        with self.assertRaises(IndexError):
            # The test stream only has one element (the complete test set)
            sliced_stream = my_ni_benchmark.test_stream[iterable_slice]

        iterable_slice = [0, 0, 0]
        sliced_stream = my_ni_benchmark.test_stream[iterable_slice]
        self.assertIsInstance(sliced_stream, GenericScenarioStream)
        self.assertEqual(len(iterable_slice), len(sliced_stream))
        self.assertEqual("test", sliced_stream.name)

        for batch_id, experience in enumerate(sliced_stream):
            self.assertEqual(
                iterable_slice[batch_id], experience.current_experience
            )
            self.assertIsInstance(experience, NIExperience)


if __name__ == "__main__":
    unittest.main()
