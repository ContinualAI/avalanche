import unittest

from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios.new_classes import NCStepInfo, NCScenario
from avalanche.training.utils import TransformationSubset
from avalanche.benchmarks.scenarios.new_classes.nc_utils import \
    make_nc_transformation_subset
from avalanche.benchmarks import nc_scenario


class SITTests(unittest.TestCase):
    def test_sit_single_dataset(self):
        mnist_train = MNIST(
            './data/mnist', train=True, download=True)
        mnist_test = MNIST(
            './data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 5, task_labels=False, shuffle=True,
            seed=1234)

        self.assertEqual(5, my_nc_scenario.n_steps)
        self.assertEqual(10, my_nc_scenario.n_classes)
        for batch_id in range(my_nc_scenario.n_steps):
            self.assertEqual(2, len(my_nc_scenario.classes_in_step[batch_id]))

        all_classes = set()
        for batch_id in range(5):
            all_classes.update(my_nc_scenario.classes_in_step[batch_id])

        self.assertEqual(10, len(all_classes))

    def test_sit_single_dataset_fixed_order(self):
        order = [2, 3, 5, 7, 8, 9, 0, 1, 4, 6]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 5, task_labels=False,
            fixed_class_order=order)

        all_classes = []
        for batch_id in range(5):
            all_classes.extend(my_nc_scenario.classes_in_step[batch_id])

        self.assertEqual(order, all_classes)

    def test_sit_single_dataset_fixed_order_subset(self):
        order = [2, 3, 5, 8, 9, 1, 4, 6]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 4, task_labels=False,
            fixed_class_order=order)

        self.assertEqual(4, len(my_nc_scenario.classes_in_step))

        all_classes = set()
        for batch_id in range(4):
            self.assertEqual(2, len(my_nc_scenario.classes_in_step[batch_id]))
            all_classes.update(my_nc_scenario.classes_in_step[batch_id])

        self.assertEqual(set(order), all_classes)

    def test_sit_single_dataset_remap_indexes(self):
        order = [2, 3, 5, 8, 9, 1, 4, 6]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 4, task_labels=False,
            fixed_class_order=order, class_ids_from_zero_from_first_step=True)

        self.assertEqual(4, len(my_nc_scenario.classes_in_step))

        all_classes = []
        for batch_id in range(4):
            self.assertEqual(2, len(my_nc_scenario.classes_in_step[batch_id]))
            all_classes.extend(my_nc_scenario.classes_in_step[batch_id])

        self.assertEqual(list(range(8)), all_classes)

    def test_sit_single_dataset_reproducibility_data(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario_ref = nc_scenario(
            mnist_train, mnist_test, 5, task_labels=False, shuffle=True,
            seed=5678)

        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, -1, task_labels=False,
            reproducibility_data=nc_scenario_ref.get_reproducibility_data())

        self.assertEqual(nc_scenario_ref.train_steps_patterns_assignment,
                         my_nc_scenario.train_steps_patterns_assignment)

        self.assertEqual(nc_scenario_ref.test_steps_patterns_assignment,
                         my_nc_scenario.test_steps_patterns_assignment)

    def test_sit_single_dataset_batch_size(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 3, task_labels=False,
            per_step_classes={0: 5, 2: 2})

        self.assertEqual(3, my_nc_scenario.n_steps)
        self.assertEqual(10, my_nc_scenario.n_classes)

        all_classes = set()
        for batch_id in range(3):
            all_classes.update(my_nc_scenario.classes_in_step[batch_id])
        self.assertEqual(10, len(all_classes))

        self.assertEqual(5, len(my_nc_scenario.classes_in_step[0]))
        self.assertEqual(3, len(my_nc_scenario.classes_in_step[1]))
        self.assertEqual(2, len(my_nc_scenario.classes_in_step[2]))

    def test_sit_multi_dataset_one_batch_per_set(self):
        split_mapping = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)

        train_part1 = make_nc_transformation_subset(
            mnist_train, None, None, range(3))
        train_part2 = make_nc_transformation_subset(
            mnist_train, None, None, range(3, 10))
        train_part2 = TransformationSubset(
            train_part2, None, class_mapping=split_mapping)

        test_part1 = make_nc_transformation_subset(
            mnist_test, None, None, range(3))
        test_part2 = make_nc_transformation_subset(
            mnist_test, None, None, range(3, 10))
        test_part2 = TransformationSubset(test_part2, None,
                                          class_mapping=split_mapping)
        my_nc_scenario = nc_scenario(
            [train_part1, train_part2], [test_part1, test_part2], 2,
            task_labels=False, shuffle=True, seed=1234,
            one_dataset_per_step=True)

        self.assertEqual(2, my_nc_scenario.n_steps)
        self.assertEqual(10, my_nc_scenario.n_classes)

        all_classes = set()
        for batch_id in range(2):
            all_classes.update(my_nc_scenario.classes_in_step[batch_id])

        self.assertEqual(10, len(all_classes))

        self.assertTrue(
            (my_nc_scenario.classes_in_step[0] == {0, 1, 2} and
             my_nc_scenario.classes_in_step[1] == set(range(3, 10))) or
            (my_nc_scenario.classes_in_step[0] == set(range(3, 10)) and
             my_nc_scenario.classes_in_step[1] == {0, 1, 2}))

    def test_sit_multi_dataset_merge(self):
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
        my_nc_scenario = nc_scenario(
            [train_part1, train_part2], [test_part1, test_part2], 5,
            task_labels=False, shuffle=True, seed=1234)

        self.assertEqual(5, my_nc_scenario.n_steps)
        self.assertEqual(10, my_nc_scenario.n_classes)
        for batch_id in range(5):
            self.assertEqual(2, len(my_nc_scenario.classes_in_step[batch_id]))

        all_classes = set()
        for batch_id in range(5):
            all_classes.update(my_nc_scenario.classes_in_step[batch_id])

        self.assertEqual(10, len(all_classes))

    def test_nc_sit_slicing(self):
        mnist_train = MNIST(
            './data/mnist', train=True, download=True)
        mnist_test = MNIST(
            './data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 5, task_labels=False, shuffle=True,
            seed=1234)

        step_info: NCStepInfo
        for batch_id, step_info in enumerate(my_nc_scenario):
            self.assertEqual(batch_id, step_info.current_step)
            self.assertIsInstance(step_info, NCStepInfo)

        iterable_slice = [3, 4, 1]
        sliced_scenario = my_nc_scenario[iterable_slice]
        self.assertIsInstance(sliced_scenario, NCScenario)
        self.assertEqual(len(iterable_slice), len(sliced_scenario))

        for batch_id, step_info in enumerate(sliced_scenario):
            self.assertEqual(iterable_slice[batch_id], step_info.current_step)
            self.assertIsInstance(step_info, NCStepInfo)


if __name__ == '__main__':
    unittest.main()
