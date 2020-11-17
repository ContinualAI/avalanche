import unittest

import torch
from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios import NIStepInfo, GenericScenarioStream
from avalanche.training.utils import TransformationSubset
from avalanche.benchmarks.scenarios.new_classes.nc_utils import \
    make_nc_transformation_subset
from avalanche.benchmarks import ni_scenario


class NISITTests(unittest.TestCase):
    def test_ni_sit_single_dataset(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_ni_scenario = ni_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234,
            balance_steps=True)

        self.assertEqual(5, my_ni_scenario.n_steps)
        self.assertEqual(10, my_ni_scenario.n_classes)
        for batch_id in range(5):
            self.assertEqual(10, len(my_ni_scenario.classes_in_step[batch_id]))

        _, unique_count = torch.unique(torch.as_tensor(mnist_train.targets),
                                       return_counts=True)

        min_batch_size = torch.sum(unique_count //
                                   my_ni_scenario.n_steps).item()
        max_batch_size = min_batch_size + my_ni_scenario.n_classes

        pattern_count = 0
        batch_info: NIStepInfo
        for batch_id, batch_info in enumerate(my_ni_scenario.train_stream):
            cur_train_set = batch_info.dataset
            t = batch_info.task_label
            self.assertEqual(0, t)
            self.assertEqual(batch_id, batch_info.current_step)
            self.assertGreaterEqual(len(cur_train_set), min_batch_size)
            self.assertLessEqual(len(cur_train_set), max_batch_size)
            pattern_count += len(cur_train_set)
        self.assertEqual(len(mnist_train), pattern_count)

        self.assertEqual(1, len(my_ni_scenario.test_stream))
        pattern_count = 0
        for batch_id, batch_info in enumerate(my_ni_scenario.test_stream):
            cur_test_set = batch_info.dataset
            t = batch_info.task_label
            self.assertEqual(0, t)
            self.assertEqual(batch_id, batch_info.current_step)
            pattern_count += len(cur_test_set)
        self.assertEqual(len(mnist_test), pattern_count)

    def test_ni_sit_single_dataset_fixed_assignment(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        ni_scenario_reference = ni_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        reference_assignment = ni_scenario_reference.\
            train_steps_patterns_assignment

        my_ni_scenario = ni_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=4321,
            fixed_step_assignment=reference_assignment)

        self.assertEqual(ni_scenario_reference.n_steps, my_ni_scenario.n_steps)

        self.assertEqual(ni_scenario_reference.train_steps_patterns_assignment,
                         my_ni_scenario.train_steps_patterns_assignment)

        self.assertEqual(ni_scenario_reference.step_structure,
                         my_ni_scenario.step_structure)

    def test_ni_sit_single_dataset_reproducibility_data(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        ni_scenario_reference = ni_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        rep_data = ni_scenario_reference.get_reproducibility_data()

        my_ni_scenario = ni_scenario(
            mnist_train, mnist_test, 0, reproducibility_data=rep_data)

        self.assertEqual(ni_scenario_reference.n_steps, my_ni_scenario.n_steps)

        self.assertEqual(ni_scenario_reference.train_steps_patterns_assignment,
                         my_ni_scenario.train_steps_patterns_assignment)

        self.assertEqual(ni_scenario_reference.step_structure,
                         my_ni_scenario.step_structure)

    def test_ni_sit_multi_dataset_merge(self):
        split_mapping = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)

        train_part1 = make_nc_transformation_subset(
            mnist_train, None, None, range(5))
        train_part2 = make_nc_transformation_subset(
            mnist_train, None, None, range(5, 10))
        train_part2 = TransformationSubset(
            train_part2, None, class_mapping=split_mapping)

        test_part1 = make_nc_transformation_subset(
            mnist_test, None, None, range(5))
        test_part2 = make_nc_transformation_subset(
            mnist_test, None, None, range(5, 10))
        test_part2 = TransformationSubset(test_part2, None,
                                          class_mapping=split_mapping)
        my_ni_scenario = ni_scenario(
            [train_part1, train_part2], [test_part1, test_part2], 5,
            shuffle=True, seed=1234, balance_steps=True)

        self.assertEqual(5, my_ni_scenario.n_steps)
        self.assertEqual(10, my_ni_scenario.n_classes)
        for batch_id in range(5):
            self.assertEqual(10, len(my_ni_scenario.classes_in_step[batch_id]))

        all_classes = set()
        for batch_id in range(5):
            all_classes.update(my_ni_scenario.classes_in_step[batch_id])

        self.assertEqual(10, len(all_classes))

    def test_ni_sit_slicing(self):
        mnist_train = MNIST(
            './data/mnist', train=True, download=True)
        mnist_test = MNIST(
            './data/mnist', train=False, download=True)
        my_ni_scenario = ni_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        step_info: NIStepInfo
        for batch_id, step_info in enumerate(my_ni_scenario.train_stream):
            self.assertEqual(batch_id, step_info.current_step)
            self.assertIsInstance(step_info, NIStepInfo)

        self.assertEqual(1, len(my_ni_scenario.test_stream))
        for batch_id, step_info in enumerate(my_ni_scenario.test_stream):
            self.assertEqual(batch_id, step_info.current_step)
            self.assertIsInstance(step_info, NIStepInfo)

        iterable_slice = [3, 4, 1]
        sliced_stream = my_ni_scenario.train_stream[iterable_slice]
        self.assertIsInstance(sliced_stream, GenericScenarioStream)
        self.assertEqual(len(iterable_slice), len(sliced_stream))
        self.assertEqual('train', sliced_stream.name)

        for batch_id, step_info in enumerate(sliced_stream):
            self.assertEqual(iterable_slice[batch_id], step_info.current_step)
            self.assertIsInstance(step_info, NIStepInfo)

        with self.assertRaises(IndexError):
            # The test stream only has one element (the complete test set)
            sliced_stream = my_ni_scenario.test_stream[iterable_slice]

        iterable_slice = [0, 0, 0]
        sliced_stream = my_ni_scenario.test_stream[iterable_slice]
        self.assertIsInstance(sliced_stream, GenericScenarioStream)
        self.assertEqual(len(iterable_slice), len(sliced_stream))
        self.assertEqual('test', sliced_stream.name)

        for batch_id, step_info in enumerate(sliced_stream):
            self.assertEqual(iterable_slice[batch_id], step_info.current_step)
            self.assertIsInstance(step_info, NIStepInfo)


if __name__ == '__main__':
    unittest.main()
