import unittest

from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios import \
    create_nc_single_dataset_sit_scenario, create_nc_multi_dataset_sit_scenario
from avalanche.training.utils import TransformationSubset
from avalanche.benchmarks.scenarios.new_classes.nc_utils import \
    make_nc_transformation_subset


class SITTests(unittest.TestCase):
    def test_sit_single_dataset(self):
        mnist_train = MNIST(
            './data/mnist', train=True, download=True)
        mnist_test = MNIST(
            './data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        self.assertEqual(5, nc_scenario.n_batches)
        self.assertEqual(10, nc_scenario.n_classes)
        for batch_id in range(nc_scenario.n_batches):
            self.assertEqual(2, len(nc_scenario.classes_in_batch[batch_id]))

        all_classes = set()
        for batch_id in range(5):
            all_classes.update(nc_scenario.classes_in_batch[batch_id])

        self.assertEqual(10, len(all_classes))

    def test_sit_single_dataset_fixed_order(self):
        order = [2, 3, 5, 7, 8, 9, 0, 1, 4, 6]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 5, fixed_class_order=order)

        all_classes = []
        for batch_id in range(5):
            all_classes.extend(nc_scenario.classes_in_batch[batch_id])

        self.assertEqual(order, all_classes)

    def test_sit_single_dataset_fixed_order_subset(self):
        order = [2, 3, 5, 8, 9, 1, 4, 6]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 4, fixed_class_order=order)

        self.assertEqual(4, len(nc_scenario.classes_in_batch))

        all_classes = []
        for batch_id in range(4):
            self.assertEqual(2, len(nc_scenario.classes_in_batch[batch_id]))
            all_classes.extend(nc_scenario.classes_in_batch[batch_id])

        self.assertEqual(order, all_classes)

    def test_sit_single_dataset_remap_indexes(self):
        order = [2, 3, 5, 8, 9, 1, 4, 6]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 4, fixed_class_order=order,
            remap_class_ids=True)

        self.assertEqual(4, len(nc_scenario.classes_in_batch))

        all_classes = []
        for batch_id in range(4):
            self.assertEqual(2, len(nc_scenario.classes_in_batch[batch_id]))
            all_classes.extend(nc_scenario.classes_in_batch[batch_id])

        self.assertEqual(list(range(8)), all_classes)

    def test_sit_single_dataset_reproducibility_data(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario_ref = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=5678)

        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, -1,
            reproducibility_data=nc_scenario_ref.get_reproducibility_data())

        self.assertEqual(nc_scenario_ref.train_steps_patterns_assignment,
                         nc_scenario.train_steps_patterns_assignment)

        self.assertEqual(nc_scenario_ref.test_steps_patterns_assignment,
                         nc_scenario.test_steps_patterns_assignment)

    def test_sit_single_dataset_batch_size(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 3, per_batch_classes={0: 5, 2: 2})

        self.assertEqual(3, nc_scenario.n_batches)
        self.assertEqual(10, nc_scenario.n_classes)

        all_classes = set()
        for batch_id in range(3):
            all_classes.update(nc_scenario.classes_in_batch[batch_id])
        self.assertEqual(10, len(all_classes))

        self.assertEqual(5, len(nc_scenario.classes_in_batch[0]))
        self.assertEqual(3, len(nc_scenario.classes_in_batch[1]))
        self.assertEqual(2, len(nc_scenario.classes_in_batch[2]))

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
        nc_scenario = create_nc_multi_dataset_sit_scenario(
            [train_part1, train_part2], [test_part1, test_part2], 2,
            shuffle=True, seed=1234, one_dataset_per_batch=True)

        self.assertEqual(2, nc_scenario.n_batches)
        self.assertEqual(10, nc_scenario.n_classes)

        all_classes = set()
        for batch_id in range(2):
            all_classes.update(nc_scenario.classes_in_batch[batch_id])

        self.assertEqual(10, len(all_classes))

        self.assertTrue(
            (nc_scenario.classes_in_batch[0] == [0, 1, 2] and
             nc_scenario.classes_in_batch[1] == list(range(3, 10))) or
            (nc_scenario.classes_in_batch[0] == list(range(3, 10)) and
             nc_scenario.classes_in_batch[1] == [0, 1, 2]))

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
        nc_scenario = create_nc_multi_dataset_sit_scenario(
            [train_part1, train_part2], [test_part1, test_part2], 5,
            shuffle=True, seed=1234)

        self.assertEqual(5, nc_scenario.n_batches)
        self.assertEqual(10, nc_scenario.n_classes)
        for batch_id in range(5):
            self.assertEqual(2, len(nc_scenario.classes_in_batch[batch_id]))

        all_classes = set()
        for batch_id in range(5):
            all_classes.update(nc_scenario.classes_in_batch[batch_id])

        self.assertEqual(10, len(all_classes))


if __name__ == '__main__':
    unittest.main()
