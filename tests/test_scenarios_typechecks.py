import unittest
import torch

from os.path import expanduser

from torchvision.datasets import MNIST

from avalanche.benchmarks import tensors_benchmark
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark

from avalanche.benchmarks.scenarios.generic_definitions import Experience


class ScenariosTypeChecksTests(unittest.TestCase):
    def test_nc_mt_type(self):
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
        )
        my_nc_benchmark = nc_benchmark(
            mnist_train,
            mnist_test,
            5,
            task_labels=True,
            class_ids_from_zero_in_each_exp=True,
        )

        for task_info in my_nc_benchmark.train_stream:
            self.assertIsInstance(task_info, Experience)

        for task_info in my_nc_benchmark.test_stream:
            self.assertIsInstance(task_info, Experience)

    def test_nc_sit_type(self):
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
        )
        my_nc_benchmark = nc_benchmark(
            mnist_train, mnist_test, 5, task_labels=False
        )

        for batch_info in my_nc_benchmark.train_stream:
            self.assertIsInstance(batch_info, Experience)

        for batch_info in my_nc_benchmark.test_stream:
            self.assertIsInstance(batch_info, Experience)

    def test_ni_sit_type(self):
        mnist_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
        )
        mnist_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
        )
        my_ni_benchmark = ni_benchmark(mnist_train, mnist_test, 5)

        for batch_info in my_ni_benchmark.train_stream:
            self.assertIsInstance(batch_info, Experience)

        for batch_info in my_ni_benchmark.test_stream:
            self.assertIsInstance(batch_info, Experience)

    def test_tensor_benchmark_type(self):
        n_experiences = 3

        tensors_benchmark(
            train_tensors=[
                (torch.randn(2, 3), torch.zeros(2))
                for _ in range(n_experiences)
            ],
            test_tensors=[
                (torch.randn(2, 3), torch.zeros(2))
                for _ in range(n_experiences)
            ],
            task_labels=[0] * n_experiences,
            complete_test_set_only=False,
        )

        tensors_benchmark(
            train_tensors=[
                (torch.randn(2, 3), torch.zeros(2))
                for _ in range(n_experiences)
            ],
            test_tensors=[(torch.randn(2, 3), torch.zeros(2))],
            task_labels=[0] * n_experiences,
            complete_test_set_only=True,
        )

        with self.assertRaises(Exception):
            tensors_benchmark(
                train_tensors=[
                    (torch.randn(2, 3), torch.zeros(2))
                    for _ in range(n_experiences)
                ],
                test_tensors=[
                    (torch.randn(2, 3), torch.zeros(2))
                    for _ in range(n_experiences)
                ],
                task_labels=[0] * n_experiences,
                complete_test_set_only=True,
            )


if __name__ == "__main__":
    unittest.main()
