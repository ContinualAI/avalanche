import unittest

from os.path import expanduser

from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios.new_classes import NCExperience
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.benchmarks.scenarios.new_classes.nc_utils import \
    make_nc_transformation_subset
from avalanche.benchmarks import nc_benchmark, GenericScenarioStream


class SITTests(unittest.TestCase):
    def test_sit_single_dataset(self):
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True, download=True)
        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False, download=True)
        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=False, shuffle=True,
            seed=1234)

        self.assertEqual(5, my_nc_benchmark.n_experiences)
        self.assertEqual(10, my_nc_benchmark.n_classes)
        for batch_id in range(my_nc_benchmark.n_experiences):
            self.assertEqual(
                2, len(my_nc_benchmark.classes_in_experience[batch_id]))

        all_classes = set()
        for batch_id in range(5):
            all_classes.update(my_nc_benchmark.classes_in_experience[batch_id])

        self.assertEqual(10, len(all_classes))

    def test_sit_single_dataset_fixed_order(self):
        order = [2, 3, 5, 7, 8, 9, 0, 1, 4, 6]
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True)
        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=False,
            fixed_class_order=order)

        all_classes = []
        for batch_id in range(5):
            all_classes.extend(my_nc_benchmark.classes_in_experience[batch_id])

        self.assertEqual(order, all_classes)

    def test_sit_single_dataset_fixed_order_subset(self):
        order = [2, 3, 5, 8, 9, 1, 4, 6]
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True)
        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, 4, task_labels=False,
            fixed_class_order=order)

        self.assertEqual(4, len(my_nc_benchmark.classes_in_experience))

        all_classes = set()
        for batch_id in range(4):
            self.assertEqual(
                2, len(my_nc_benchmark.classes_in_experience[batch_id]))
            all_classes.update(my_nc_benchmark.classes_in_experience[batch_id])

        self.assertEqual(set(order), all_classes)

    def test_sit_single_dataset_remap_indexes(self):
        order = [2, 3, 5, 8, 9, 1, 4, 6]
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True)
        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, 4, task_labels=False,
            fixed_class_order=order, class_ids_from_zero_from_first_exp=True)

        self.assertEqual(4, len(my_nc_benchmark.classes_in_experience))

        all_classes = []
        for batch_id in range(4):
            self.assertEqual(
                2, len(my_nc_benchmark.classes_in_experience[batch_id]))
            all_classes.extend(my_nc_benchmark.classes_in_experience[batch_id])
        self.assertEqual(list(range(8)), all_classes)

        # Regression test for issue #258
        for i, experience in enumerate(my_nc_benchmark.train_stream):
            unique_dataset_classes = sorted(set(experience.dataset.targets))
            expected_dataset_classes = list(range(2 * i, 2 * (i+1)))

            self.assertListEqual(expected_dataset_classes,
                                 unique_dataset_classes)
            self.assertListEqual(
                sorted(order[2 * i:2 * (i+1)]),
                sorted(my_nc_benchmark.original_classes_in_exp[i]))
        # End regression test for issue #258

    def test_sit_single_dataset_remap_indexes_each_exp(self):
        order = [2, 3, 5, 8, 9, 1, 4, 6]
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True)

        with self.assertRaises(ValueError):
            # class_ids_from_zero_* are mutually exclusive
            nc_benchmark(
                mnist_train, mnist_test, 4, task_labels=False,
                fixed_class_order=order,
                class_ids_from_zero_from_first_exp=True,
                class_ids_from_zero_in_each_exp=True)

        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, 4, task_labels=False,
            fixed_class_order=order,
            class_ids_from_zero_in_each_exp=True)

        self.assertEqual(4, len(my_nc_benchmark.classes_in_experience))

        all_classes = []
        for batch_id in range(4):
            self.assertEqual(
                2, len(my_nc_benchmark.classes_in_experience[batch_id]))
            all_classes.extend(my_nc_benchmark.classes_in_experience[batch_id])
        self.assertEqual(8, len(all_classes))
        self.assertListEqual([0, 1], sorted(set(all_classes)))

        # Regression test for issue #258
        for i, experience in enumerate(my_nc_benchmark.train_stream):
            unique_dataset_classes = sorted(set(experience.dataset.targets))
            expected_dataset_classes = [0, 1]
            self.assertListEqual(expected_dataset_classes,
                                 unique_dataset_classes)
            self.assertListEqual(
                sorted(order[2 * i:2 * (i + 1)]),
                sorted(my_nc_benchmark.original_classes_in_exp[i]))
        # End regression test for issue #258

    def test_sit_single_dataset_reproducibility_data(self):
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True)
        nc_benchmark_ref = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=False, shuffle=True,
            seed=5678)

        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, -1, task_labels=False,
            reproducibility_data=nc_benchmark_ref.get_reproducibility_data())

        self.assertEqual(nc_benchmark_ref.train_exps_patterns_assignment,
                         my_nc_benchmark.train_exps_patterns_assignment)

        self.assertEqual(nc_benchmark_ref.test_exps_patterns_assignment,
                         my_nc_benchmark.test_exps_patterns_assignment)

    def test_sit_single_dataset_batch_size(self):
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True)
        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, 3, task_labels=False,
            per_exp_classes={0: 5, 2: 2})

        self.assertEqual(3, my_nc_benchmark.n_experiences)
        self.assertEqual(10, my_nc_benchmark.n_classes)

        all_classes = set()
        for batch_id in range(3):
            all_classes.update(my_nc_benchmark.classes_in_experience[batch_id])
        self.assertEqual(10, len(all_classes))

        self.assertEqual(5, len(my_nc_benchmark.classes_in_experience[0]))
        self.assertEqual(3, len(my_nc_benchmark.classes_in_experience[1]))
        self.assertEqual(2, len(my_nc_benchmark.classes_in_experience[2]))

    def test_sit_multi_dataset_one_batch_per_set(self):
        split_mapping = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6]
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True)

        train_part1 = make_nc_transformation_subset(
            mnist_train, None, None, range(3))
        train_part2 = make_nc_transformation_subset(
            mnist_train, None, None, range(3, 10))
        train_part2 = AvalancheSubset(
            train_part2, class_mapping=split_mapping)

        test_part1 = make_nc_transformation_subset(
            mnist_test, None, None, range(3))
        test_part2 = make_nc_transformation_subset(
            mnist_test, None, None, range(3, 10))
        test_part2 = AvalancheSubset(test_part2,
                                     class_mapping=split_mapping)
        my_nc_benchmark = nc_benchmark(
            [train_part1, train_part2], [test_part1, test_part2], 2,
            task_labels=False, shuffle=True, seed=1234,
            one_dataset_per_exp=True)

        self.assertEqual(2, my_nc_benchmark.n_experiences)
        self.assertEqual(10, my_nc_benchmark.n_classes)

        all_classes = set()
        for batch_id in range(2):
            all_classes.update(my_nc_benchmark.classes_in_experience[batch_id])

        self.assertEqual(10, len(all_classes))

        self.assertTrue(
            (my_nc_benchmark.classes_in_experience[0] == {0, 1, 2} and
             my_nc_benchmark.classes_in_experience[1] == set(range(3, 10))) or
            (my_nc_benchmark.classes_in_experience[0] == set(range(3, 10)) and
             my_nc_benchmark.classes_in_experience[1] == {0, 1, 2}))

    def test_sit_multi_dataset_merge(self):
        split_mapping = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True)

        train_part1 = make_nc_transformation_subset(
            mnist_train, None, None, range(5))
        train_part2 = make_nc_transformation_subset(
            mnist_train, None, None, range(5, 10))
        train_part2 = AvalancheSubset(
            train_part2, class_mapping=split_mapping)

        test_part1 = make_nc_transformation_subset(
            mnist_test, None, None, range(5))
        test_part2 = make_nc_transformation_subset(
            mnist_test, None, None, range(5, 10))
        test_part2 = AvalancheSubset(test_part2,
                                     class_mapping=split_mapping)
        my_nc_benchmark = nc_benchmark(
            [train_part1, train_part2], [test_part1, test_part2], 5,
            task_labels=False, shuffle=True, seed=1234)

        self.assertEqual(5, my_nc_benchmark.n_experiences)
        self.assertEqual(10, my_nc_benchmark.n_classes)
        for batch_id in range(5):
            self.assertEqual(
                2, len(my_nc_benchmark.classes_in_experience[batch_id]))

        all_classes = set()
        for batch_id in range(5):
            all_classes.update(my_nc_benchmark.classes_in_experience[batch_id])

        self.assertEqual(10, len(all_classes))

    def test_nc_sit_slicing(self):
        mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                            train=True, download=True)
        mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                           train=False, download=True)
        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=False, shuffle=True,
            seed=1234)

        experience: NCExperience
        for batch_id, experience in enumerate(my_nc_benchmark.train_stream):
            self.assertEqual(batch_id, experience.current_experience)
            self.assertIsInstance(experience, NCExperience)

        for batch_id, experience in enumerate(my_nc_benchmark.test_stream):
            self.assertEqual(batch_id, experience.current_experience)
            self.assertIsInstance(experience, NCExperience)

        iterable_slice = [3, 4, 1]
        sliced_stream = my_nc_benchmark.train_stream[iterable_slice]
        self.assertIsInstance(sliced_stream, GenericScenarioStream)
        self.assertEqual(len(iterable_slice), len(sliced_stream))
        self.assertEqual('train', sliced_stream.name)

        for batch_id, experience in enumerate(sliced_stream):
            self.assertEqual(iterable_slice[batch_id],
                             experience.current_experience)
            self.assertIsInstance(experience, NCExperience)

        sliced_stream = my_nc_benchmark.test_stream[iterable_slice]
        self.assertIsInstance(sliced_stream, GenericScenarioStream)
        self.assertEqual(len(iterable_slice), len(sliced_stream))
        self.assertEqual('test', sliced_stream.name)

        for batch_id, experience in enumerate(sliced_stream):
            self.assertEqual(iterable_slice[batch_id],
                             experience.current_experience)
            self.assertIsInstance(experience, NCExperience)


if __name__ == '__main__':
    unittest.main()
