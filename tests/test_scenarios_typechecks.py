import unittest
import torch

from torchvision.datasets import MNIST

from avalanche.benchmarks.generators import TensorScenario, NCBenchmark, \
    NIBenchmark

from avalanche.benchmarks.scenarios.generic_definitions import IStepInfo


class ScenariosTypeChecksTests(unittest.TestCase):
    def test_nc_mt_type(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = NCBenchmark(
            mnist_train, mnist_test, 5, task_labels=True,
            class_ids_from_zero_in_each_step=True)

        for task_info in nc_scenario:
            self.assertIsInstance(task_info, IStepInfo)

    def test_nc_sit_type(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        nc_scenario = NCBenchmark(
            mnist_train, mnist_test, 5, task_labels=False)

        for batch_info in nc_scenario:
            self.assertIsInstance(batch_info, IStepInfo)

    def test_ni_sit_type(self):
        mnist_train = MNIST('./data/mnist', train=True, download=True)
        mnist_test = MNIST('./data/mnist', train=False, download=True)
        ni_scenario = NIBenchmark(
            mnist_train, mnist_test, 5)

        for batch_info in ni_scenario:
            self.assertIsInstance(batch_info, IStepInfo)

    def test_TensorScenario_type(self):
        n_steps = 3
        test_data_x = [[torch.zeros(2, 3)], torch.zeros(2, 3)]
        test_data_y = [[torch.zeros(2)], torch.zeros(2)]

        for complete_test in [True, False]:
            for tdx, tdy in zip(test_data_x, test_data_y):
                try:                
                    TensorScenario(
                        train_data_x=[torch.randn(2, 3)
                                      for _ in range(n_steps)],
                        train_data_y=[torch.zeros(2) for i in range(n_steps)],
                        test_data_x=tdx,
                        test_data_y=tdy,
                        task_labels=[0]*n_steps,
                        complete_test_set_only=complete_test)
                except ValueError:
                    if complete_test and \
                        not isinstance(tdx, torch.Tensor) and \
                            not isinstance(tdy, torch.Tensor):
                        print("Value Error raised correctly")
                        

if __name__ == '__main__':
    unittest.main()
