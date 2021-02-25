import unittest

from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios.new_classes import NCExperience
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.benchmarks.scenarios.new_classes.nc_utils import \
    make_nc_transformation_subset
from avalanche.benchmarks import nc_scenario, GenericScenarioStream


class MultiTaskTests(unittest.TestCase):
    def test_mt_single_dataset(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 5, task_labels=True, shuffle=True,
            seed=1234, class_ids_from_zero_in_each_exp=True)

        self.assertEqual(5, my_nc_scenario.n_experiences)
        self.assertEqual(10, my_nc_scenario.n_classes)
        for task_id in range(5):
            self.assertEqual(2, len(my_nc_scenario.classes_in_experience[task_id]))

        all_classes = set()
        all_original_classes = set()
        for task_id in range(5):
            all_classes.update(my_nc_scenario.classes_in_experience[task_id])
            all_original_classes.update(
                my_nc_scenario.original_classes_in_exp[task_id])

        self.assertEqual(2, len(all_classes))
        self.assertEqual(10, len(all_original_classes))

    def test_mt_single_dataset_without_class_id_remap(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 5, task_labels=True, shuffle=True,
            seed=1234, class_ids_from_zero_in_each_exp=False)

        self.assertEqual(5, my_nc_scenario.n_experiences)
        self.assertEqual(10, my_nc_scenario.n_classes)
        for task_id in range(5):
            self.assertEqual(2, len(my_nc_scenario.classes_in_experience[task_id]))

        all_classes = set()
        for task_id in range(my_nc_scenario.n_experiences):
            all_classes.update(my_nc_scenario.classes_in_experience[task_id])

        self.assertEqual(10, len(all_classes))

    def test_mt_single_dataset_fixed_order(self):
        order = [2, 3, 5, 7, 8, 9, 0, 1, 4, 6]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 5, task_labels=True,
            fixed_class_order=order, class_ids_from_zero_in_each_exp=False)

        all_classes = []
        for task_id in range(5):
            all_classes.extend(my_nc_scenario.classes_in_experience[task_id])

        self.assertEqual(order, all_classes)

    def test_sit_single_dataset_fixed_order_subset(self):
        order = [2, 5, 7, 8, 9, 0, 1, 4]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 4, task_labels=True,
            fixed_class_order=order, class_ids_from_zero_in_each_exp=True)

        self.assertEqual(4, len(my_nc_scenario.classes_in_experience))

        all_classes = []
        for task_id in range(4):
            self.assertEqual(2, len(my_nc_scenario.classes_in_experience[task_id]))
            self.assertEqual(set(order[task_id*2:(task_id+1)*2]),
                             my_nc_scenario.original_classes_in_exp[task_id])
            all_classes.extend(my_nc_scenario.classes_in_experience[task_id])

        self.assertEqual([0, 1] * 4, all_classes)

    def test_sit_single_dataset_fixed_subset_no_remap_idx(self):
        order = [2, 5, 7, 8, 9, 0, 1, 4]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 2, task_labels=True,
            fixed_class_order=order, class_ids_from_zero_in_each_exp=False)

        self.assertEqual(2, len(my_nc_scenario.classes_in_experience))

        all_classes = set()
        for task_id in range(2):
            self.assertEqual(4, len(my_nc_scenario.classes_in_experience[task_id]))
            self.assertEqual(set(order[task_id*4:(task_id+1)*4]),
                             my_nc_scenario.original_classes_in_exp[task_id])
            all_classes.update(my_nc_scenario.classes_in_experience[task_id])

        self.assertEqual(set(order), all_classes)

    def test_mt_single_dataset_reproducibility_data(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario_ref = nc_scenario(
            mnist_train, mnist_test, 5, task_labels=True, shuffle=True,
            seed=5678)

        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, -1, task_labels=True,
            reproducibility_data=nc_scenario_ref.get_reproducibility_data())

        self.assertEqual(nc_scenario_ref.train_exps_patterns_assignment,
                         my_nc_scenario.train_exps_patterns_assignment)

        self.assertEqual(nc_scenario_ref.test_exps_patterns_assignment,
                         my_nc_scenario.test_exps_patterns_assignment)

    def test_mt_single_dataset_task_size(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 3, task_labels=True,
            per_exp_classes={0: 5, 2: 2},
            class_ids_from_zero_in_each_exp=True)

        self.assertEqual(3, my_nc_scenario.n_experiences)
        self.assertEqual(10, my_nc_scenario.n_classes)

        all_classes = set()
        for task_id in range(3):
            all_classes.update(my_nc_scenario.classes_in_experience[task_id])
        self.assertEqual(5, len(all_classes))

        self.assertEqual(5, len(my_nc_scenario.classes_in_experience[0]))
        self.assertEqual(3, len(my_nc_scenario.classes_in_experience[1]))
        self.assertEqual(2, len(my_nc_scenario.classes_in_experience[2]))

    def test_mt_multi_dataset_one_task_per_set(self):
        split_mapping = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)

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
        my_nc_scenario = nc_scenario(
            [train_part1, train_part2], [test_part1, test_part2], 2,
            task_labels=True, seed=1234,
            class_ids_from_zero_in_each_exp=True, one_dataset_per_exp=True)

        self.assertEqual(2, my_nc_scenario.n_experiences)
        self.assertEqual(10, my_nc_scenario.n_classes)
        self.assertEqual(2, len(my_nc_scenario.train_stream))
        self.assertEqual(2, len(my_nc_scenario.test_stream))

        step_classes_train = []
        step_classes_test = []

        all_classes_train = set()
        all_classes_test = set()

        task_info: NCExperience
        for task_id, task_info in enumerate(my_nc_scenario.train_stream):
            self.assertLessEqual(task_id, 1)
            all_classes_train.update(my_nc_scenario.classes_in_experience[task_id])
            step_classes_train.append(task_info.classes_in_this_experience)
        self.assertEqual(7, len(all_classes_train))

        for task_id, task_info in enumerate(my_nc_scenario.test_stream):
            self.assertLessEqual(task_id, 1)
            all_classes_test.update(my_nc_scenario.classes_in_experience[task_id])
            step_classes_test.append(task_info.classes_in_this_experience)
        self.assertEqual(7, len(all_classes_test))

        self.assertTrue(
            (my_nc_scenario.classes_in_experience[0] == {0, 1, 2} and
             my_nc_scenario.classes_in_experience[1] == set(range(0, 7))) or
            (my_nc_scenario.classes_in_experience[0] == set(range(0, 7)) and
             my_nc_scenario.classes_in_experience[1] == {0, 1, 2}))

        step_classes_ref1 = [list(range(3)), list(range(7))]
        step_classes_ref2 = [list(range(7)), list(range(3))]

        self.assertTrue(step_classes_train == step_classes_ref1 or
                        step_classes_train == step_classes_ref2)

        if step_classes_train == step_classes_ref1:
            self.assertTrue(step_classes_test == step_classes_ref1)
        else:
            self.assertTrue(step_classes_test == step_classes_ref2)

    def test_nc_mt_slicing(self):
        mnist_train = MNIST(
            './data/mnist', train=True, download=True)
        mnist_test = MNIST(
            './data/mnist', train=False, download=True)
        my_nc_scenario = nc_scenario(
            mnist_train, mnist_test, 5, task_labels=True, shuffle=True,
            seed=1234)

        step_info: NCExperience
        for batch_id, step_info in enumerate(my_nc_scenario.train_stream):
            self.assertEqual(batch_id, step_info.current_step)
            self.assertIsInstance(step_info, NCExperience)

        for batch_id, step_info in enumerate(my_nc_scenario.test_stream):
            self.assertEqual(batch_id, step_info.current_step)
            self.assertIsInstance(step_info, NCExperience)

        iterable_slice = [3, 4, 1]
        sliced_stream = my_nc_scenario.train_stream[iterable_slice]
        self.assertIsInstance(sliced_stream, GenericScenarioStream)
        self.assertEqual(len(iterable_slice), len(sliced_stream))
        self.assertEqual('train', sliced_stream.name)

        for batch_id, step_info in enumerate(sliced_stream):
            self.assertEqual(iterable_slice[batch_id], step_info.current_step)
            self.assertIsInstance(step_info, NCExperience)

        sliced_stream = my_nc_scenario.test_stream[iterable_slice]
        self.assertIsInstance(sliced_stream, GenericScenarioStream)
        self.assertEqual(len(iterable_slice), len(sliced_stream))
        self.assertEqual('test', sliced_stream.name)

        for batch_id, step_info in enumerate(sliced_stream):
            self.assertEqual(iterable_slice[batch_id], step_info.current_step)
            self.assertIsInstance(step_info, NCExperience)


if __name__ == '__main__':
    unittest.main()
