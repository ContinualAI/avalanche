import unittest
import torch

from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios import \
    create_nc_single_dataset_multi_task_scenario, \
    create_nc_single_dataset_sit_scenario, \
    create_ni_single_dataset_sit_scenario

from avalanche.benchmarks.generators import TensorScenario

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

    def test_TensorScenario_type(self):
        N_BATCHES = 3
        test_data_x = [ [torch.zeros(2,3)], torch.zeros(2,3) ],
        test_data_y = [ [torch.zeros(2)], torch.zeros(2) ],

        for complete_test in [True, False]:
            for tdx, tdy in zip(test_data_x, test_data_y):
                try:                
                    TensorScenario(
                        train_data_x = [torch.randn(2,3) \
                                            for i in range(N_BATCHES)],
                        train_data_y = [torch.zeros(2) \
                                            for i in range(N_BATCHES)],
                        test_data_x = tdx,
                        test_data_y = tdy,
                        task_labels = [0]*N_BATCHES,
                        complete_test_set_only = complete_test
                    )
                except ValueError:
                    if complete_test and \
                        not isinstance(tdx, torch.Tensor) and \
                        not isinstance(tdy, torch.Tensor):
                            print("Value Error raised correctly")
                        


if __name__ == '__main__':
    unittest.main()
