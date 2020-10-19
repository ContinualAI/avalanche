import unittest

from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios import \
    create_nc_single_dataset_multi_task_scenario, \
    create_nc_multi_dataset_multi_task_scenario, NCTaskInfo
from avalanche.training.utils import TransformationSubset
from avalanche.benchmarks.scenarios.new_classes.nc_utils import \
    make_nc_transformation_subset
from avalanche.benchmarks.scenarios.generic_cl_scenario import ScenarioSlice


class MultiTaskTests(unittest.TestCase):
    def test_mt_single_dataset(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_multi_task_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        self.assertEqual(5, nc_scenario.n_tasks)
        self.assertEqual(10, nc_scenario.n_classes)
        for task_id in range(5):
            self.assertEqual(2, len(nc_scenario.classes_in_task[task_id]))

        all_classes = set()
        all_original_classes = set()
        for task_id in range(5):
            all_classes.update(nc_scenario.classes_in_task[task_id])
            all_original_classes.update(
                nc_scenario.original_classes_in_task[task_id])

        self.assertEqual(2, len(all_classes))
        self.assertEqual(10, len(all_original_classes))

    def test_mt_single_dataset_without_class_id_remap(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_multi_task_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234,
            classes_ids_from_zero_in_each_task=False)

        self.assertEqual(5, nc_scenario.n_tasks)
        self.assertEqual(10, nc_scenario.n_classes)
        for task_id in range(5):
            self.assertEqual(2, len(nc_scenario.classes_in_task[task_id]))

        all_classes = set()
        for task_id in range(nc_scenario.n_tasks):
            all_classes.update(nc_scenario.classes_in_task[task_id])

        self.assertEqual(10, len(all_classes))

    def test_mt_single_dataset_fixed_order(self):
        order = [2, 3, 5, 7, 8, 9, 0, 1, 4, 6]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_multi_task_scenario(
            mnist_train, mnist_test, 5,
            fixed_class_order=order, classes_ids_from_zero_in_each_task=False)

        all_classes = []
        for task_id in range(5):
            all_classes.extend(nc_scenario.classes_in_task[task_id])

        self.assertEqual(order, all_classes)

    def test_sit_single_dataset_fixed_order_subset(self):
        order = [2, 5, 7, 8, 9, 0, 1, 4]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_multi_task_scenario(
            mnist_train, mnist_test, 4, fixed_class_order=order)

        self.assertEqual(4, len(nc_scenario.classes_in_task))

        all_classes = []
        for task_id in range(4):
            self.assertEqual(2, len(nc_scenario.classes_in_task[task_id]))
            self.assertEqual(order[task_id*2:(task_id+1)*2],
                             nc_scenario.original_classes_in_task[task_id])
            all_classes.extend(nc_scenario.classes_in_task[task_id])

        self.assertEqual([0, 1] * 4, all_classes)

    def test_sit_single_dataset_fixed_subset_no_remap_idx(self):
        order = [2, 5, 7, 8, 9, 0, 1, 4]
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_multi_task_scenario(
            mnist_train, mnist_test, 2, fixed_class_order=order,
            classes_ids_from_zero_in_each_task=False)

        self.assertEqual(2, len(nc_scenario.classes_in_task))

        all_classes = []
        for task_id in range(2):
            self.assertEqual(4, len(nc_scenario.classes_in_task[task_id]))
            self.assertEqual(order[task_id*4:(task_id+1)*4],
                             nc_scenario.original_classes_in_task[task_id])
            all_classes.extend(nc_scenario.classes_in_task[task_id])

        self.assertEqual(order, all_classes)

    def test_mt_single_dataset_reproducibility_data(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario_ref = create_nc_single_dataset_multi_task_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=5678)

        nc_scenario = create_nc_single_dataset_multi_task_scenario(
            mnist_train, mnist_test, -1,
            reproducibility_data=nc_scenario_ref.get_reproducibility_data())

        self.assertEqual(nc_scenario_ref.train_steps_patterns_assignment,
                         nc_scenario.train_steps_patterns_assignment)

        self.assertEqual(nc_scenario_ref.test_steps_patterns_assignment,
                         nc_scenario.test_steps_patterns_assignment)

    def test_mt_single_dataset_task_size(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_multi_task_scenario(
            mnist_train, mnist_test, 3, per_task_classes={0: 5, 2: 2})

        self.assertEqual(3, nc_scenario.n_tasks)
        self.assertEqual(10, nc_scenario.n_classes)

        all_classes = set()
        for task_id in range(3):
            all_classes.update(nc_scenario.classes_in_task[task_id])
        self.assertEqual(5, len(all_classes))

        self.assertEqual(5, len(nc_scenario.classes_in_task[0]))
        self.assertEqual(3, len(nc_scenario.classes_in_task[1]))
        self.assertEqual(2, len(nc_scenario.classes_in_task[2]))

    def test_mt_multi_dataset_one_task_per_set(self):
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
        nc_scenario = create_nc_multi_dataset_multi_task_scenario(
            [train_part1, train_part2], [test_part1, test_part2], seed=1234)

        self.assertEqual(2, nc_scenario.n_tasks)
        self.assertEqual(10, nc_scenario.n_classes)

        step_classes = []

        all_classes = set()
        for task_id, task_info in enumerate(nc_scenario):
            self.assertLessEqual(task_id, 1)
            all_classes.update(nc_scenario.classes_in_task[task_id])
            step_classes.append(task_info.classes_in_this_task)

        self.assertEqual(7, len(all_classes))

        self.assertTrue(
            (nc_scenario.classes_in_task[0] == [0, 1, 2] and
             nc_scenario.classes_in_task[1] == list(range(0, 7))) or
            (nc_scenario.classes_in_task[0] == list(range(0, 7)) and
             nc_scenario.classes_in_task[1] == [0, 1, 2]))

        step_classes_ref1 = [list(range(3)), list(range(7))]
        step_classes_ref2 = [list(range(7)), list(range(3))]

        self.assertTrue(step_classes == step_classes_ref1 or
                        step_classes == step_classes_ref2)

    def test_nc_mt_slicing(self):
        mnist_train = MNIST(
            './data/mnist', train=True, download=True)
        mnist_test = MNIST(
            './data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_multi_task_scenario(
            mnist_train, mnist_test, 5, shuffle=True, seed=1234)

        step_info: NCTaskInfo
        for batch_id, step_info in enumerate(nc_scenario):
            self.assertEqual(batch_id, step_info.current_step)
            self.assertIsInstance(step_info, NCTaskInfo)

        iterable_slice = [3, 4, 1]
        sliced_scenario = nc_scenario[iterable_slice]
        self.assertIsInstance(sliced_scenario, ScenarioSlice)
        self.assertEqual(len(iterable_slice), len(sliced_scenario))

        for batch_id, step_info in enumerate(sliced_scenario):
            self.assertEqual(iterable_slice[batch_id], step_info.current_step)
            self.assertIsInstance(step_info, NCTaskInfo)


if __name__ == '__main__':
    unittest.main()
