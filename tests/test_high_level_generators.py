import unittest

import os

import torch
from torchvision.datasets import CIFAR10, MNIST
from torchvision.datasets.utils import download_url, extract_archive
from torchvision.transforms import ToTensor

from avalanche.benchmarks import dataset_benchmark, filelist_benchmark, \
    tensors_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from tests.unit_tests_utils import common_setups


class HighLevelGeneratorTests(unittest.TestCase):
    def setUp(self):
        common_setups()

    def test_dataset_benchmark(self):
        train_MNIST = MNIST(
            './data/mnist', train=True, download=True
        )
        test_MNIST = MNIST(
            './data/mnist', train=False, download=True
        )

        train_cifar10 = CIFAR10(
            './data/cifar10', train=True, download=True
        )
        test_cifar10 = CIFAR10(
            './data/cifar10', train=False, download=True
        )

        generic_scenario = dataset_benchmark(
            [train_MNIST, train_cifar10],
            [test_MNIST, test_cifar10])

    def test_dataset_benchmark_avalanche_dataset(self):
        train_MNIST = AvalancheDataset(MNIST(
            './data/mnist', train=True, download=True
        ), task_labels=0)

        test_MNIST = AvalancheDataset(MNIST(
            './data/mnist', train=False, download=True
        ), task_labels=0)

        train_cifar10 = AvalancheDataset(CIFAR10(
            './data/cifar10', train=True, download=True
        ), task_labels=1)

        test_cifar10 = AvalancheDataset(CIFAR10(
            './data/cifar10', train=False, download=True
        ), task_labels=1)

        generic_scenario = dataset_benchmark(
            [train_MNIST, train_cifar10],
            [test_MNIST, test_cifar10])

        self.assertEqual(0, generic_scenario.train_stream[0].task_label)
        self.assertEqual(1, generic_scenario.train_stream[1].task_label)
        self.assertEqual(0, generic_scenario.test_stream[0].task_label)
        self.assertEqual(1, generic_scenario.test_stream[1].task_label)

    def test_filelist_benchmark(self):
        download_url(
            'https://storage.googleapis.com/mledu-datasets/'
            'cats_and_dogs_filtered.zip', './data',
            'cats_and_dogs_filtered.zip')
        archive_name = os.path.join(
            './data', 'cats_and_dogs_filtered.zip')
        extract_archive(archive_name, to_path='./data/')

        dirpath = "./data/cats_and_dogs_filtered/train"

        for filelist, dir, label in zip(
                ["train_filelist_00.txt", "train_filelist_01.txt"],
                ["cats", "dogs"],
                [0, 1]):
            # First, obtain the list of files
            filenames_list = os.listdir(os.path.join(dirpath, dir))
            with open(filelist, "w") as wf:
                for name in filenames_list:
                    wf.write(
                        "{} {}\n".format(os.path.join(dir, name), label)
                    )

        generic_scenario = filelist_benchmark(
            dirpath,
            ["train_filelist_00.txt", "train_filelist_01.txt"],
            ["train_filelist_00.txt"],
            task_labels=[0, 0],
            complete_test_set_only=True,
            train_transform=ToTensor(),
            eval_transform=ToTensor()
        )

        self.assertEqual(2, len(generic_scenario.train_stream))
        self.assertEqual(1, len(generic_scenario.test_stream))

    def test_tensors_benchmark(self):
        pattern_shape = (3, 32, 32)

        # Definition of training experiences
        # Experience 1
        experience_1_x = torch.zeros(100, *pattern_shape)
        experience_1_y = torch.zeros(100, dtype=torch.long)

        # Experience 2
        experience_2_x = torch.zeros(80, *pattern_shape)
        experience_2_y = torch.ones(80, dtype=torch.long)

        # Test experience
        test_x = torch.zeros(50, *pattern_shape)
        test_y = torch.zeros(50, dtype=torch.long)

        generic_scenario = tensors_benchmark(
            train_tensors=[(experience_1_x, experience_1_y),
                           (experience_2_x, experience_2_y)],
            test_tensors=[(test_x, test_y)],
            task_labels=[0, 0],  # Task label of each train exp
            complete_test_set_only=True
        )

        self.assertEqual(2, len(generic_scenario.train_stream))
        self.assertEqual(1, len(generic_scenario.test_stream))
