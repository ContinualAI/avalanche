import unittest

import torch
from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios import NIStepInfo,  NIScenario
from avalanche.training.utils import TransformationSubset
from avalanche.benchmarks.scenarios.new_classes.nc_utils import \
    make_nc_transformation_subset
from avalanche.benchmarks import NIBenchmark


class NISITTests(unittest.TestCase):
    def test_ni_sit_single_dataset(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        ni_scenario = NIBenchmark(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234,
            balance_steps=True)

        self.assertEqual(5, ni_scenario.n_steps)
        self.assertEqual(10, ni_scenario.n_classes)
        for batch_id in range(5):
            self.assertEqual(10, len(ni_scenario.classes_in_step[batch_id]))

        _, unique_count = torch.unique(torch.as_tensor(mnist_train.targets),
                                       return_counts=True)

        min_batch_size = torch.sum(unique_count // ni_scenario.n_steps).item()
        max_batch_size = min_batch_size + ni_scenario.n_classes

        pattern_count = 0
        batch_info: NIStepInfo
        for batch_id, batch_info in enumerate(ni_scenario):
            cur_train_set, t = batch_info.current_training_set()
            self.assertEqual(0, t)
            self.assertEqual(batch_id, batch_info.current_step)
            self.assertGreaterEqual(len(cur_train_set), min_batch_size)
            self.assertLessEqual(len(cur_train_set), max_batch_size)
            pattern_count += len(cur_train_set)
        self.assertEqual(len(mnist_train), pattern_count)

    def test_ni_sit_single_dataset_fixed_assignment(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        ni_scenario_reference = NIBenchmark(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        reference_assignment = ni_scenario_reference.\
            train_steps_patterns_assignment

        ni_scenario = NIBenchmark(
            mnist_train, mnist_test, 5, shuffle=True, seed=4321,
            fixed_step_assignment=reference_assignment)

        self.assertEqual(ni_scenario_reference.n_steps, ni_scenario.n_steps)

        self.assertEqual(ni_scenario_reference.train_steps_patterns_assignment,
                         ni_scenario.train_steps_patterns_assignment)

        self.assertEqual(ni_scenario_reference.step_structure,
                         ni_scenario.step_structure)

    def test_ni_sit_single_dataset_reproducibility_data(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        ni_scenario_reference = NIBenchmark(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        rep_data = ni_scenario_reference.get_reproducibility_data()

        ni_scenario = NIBenchmark(
            mnist_train, mnist_test, 0, reproducibility_data=rep_data)

        self.assertEqual(ni_scenario_reference.n_steps, ni_scenario.n_steps)

        self.assertEqual(ni_scenario_reference.train_steps_patterns_assignment,
                         ni_scenario.train_steps_patterns_assignment)

        self.assertEqual(ni_scenario_reference.step_structure,
                         ni_scenario.step_structure)

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
        ni_scenario = NIBenchmark(
            [train_part1, train_part2], [test_part1, test_part2], 5,
            shuffle=True, seed=1234, balance_steps=True)

        self.assertEqual(5, ni_scenario.n_steps)
        self.assertEqual(10, ni_scenario.n_classes)
        for batch_id in range(5):
            self.assertEqual(10, len(ni_scenario.classes_in_step[batch_id]))

        all_classes = set()
        for batch_id in range(5):
            all_classes.update(ni_scenario.classes_in_step[batch_id])

        self.assertEqual(10, len(all_classes))

    def test_ni_sit_slicing(self):
        mnist_train = MNIST(
            './data/mnist', train=True, download=True)
        mnist_test = MNIST(
            './data/mnist', train=False, download=True)
        nc_scenario = NIBenchmark(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        step_info: NIStepInfo
        for batch_id, step_info in enumerate(nc_scenario):
            self.assertEqual(batch_id, step_info.current_step)
            self.assertIsInstance(step_info, NIStepInfo)

        iterable_slice = [3, 4, 1]
        sliced_scenario = nc_scenario[iterable_slice]
        self.assertIsInstance(sliced_scenario, NIScenario)
        self.assertEqual(len(iterable_slice), len(sliced_scenario))

        for batch_id, step_info in enumerate(sliced_scenario):
            self.assertEqual(iterable_slice[batch_id], step_info.current_step)
            self.assertIsInstance(step_info, NIStepInfo)


if __name__ == '__main__':
    unittest.main()
