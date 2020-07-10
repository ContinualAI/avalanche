import unittest

from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios import \
    create_nc_single_dataset_multi_task_scenario, \
    create_nc_single_dataset_sit_scenario, create_ni_single_dataset_sit_scenario
from avalanche.benchmarks.scenarios.generic_definitions import IStepInfo


class ScenariosTypeChecksTests(unittest.TestCase):
    def test_nc_mt_type(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_multi_task_scenario(
            mnist_train, mnist_test, 5)

        for task_info in nc_scenario:
            self.assertIsInstance(task_info, IStepInfo)

    def test_nc_sit_type(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = create_nc_single_dataset_sit_scenario(
            mnist_train, mnist_test, 5)

        for batch_info in nc_scenario:
            self.assertIsInstance(batch_info, IStepInfo)

    def test_ni_sit_type(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        ni_scenario = create_ni_single_dataset_sit_scenario(
            mnist_train, mnist_test, 5)

        for batch_info in ni_scenario:
            self.assertIsInstance(batch_info, IStepInfo)


if __name__ == '__main__':
    unittest.main()
