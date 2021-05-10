import unittest

import os
from os.path import expanduser

import torch
from torchvision.datasets import CIFAR10, MNIST
from torchvision.datasets.utils import download_url, extract_archive
from torchvision.transforms import ToTensor

from avalanche.benchmarks import dataset_benchmark, filelist_benchmark, \
    tensors_benchmark, paths_benchmark, data_incremental_benchmark, \
    benchmark_with_validation_stream
from avalanche.benchmarks.utils import AvalancheDataset
from tests.unit_tests_utils import common_setups


class HighLevelGeneratorTests(unittest.TestCase):
    def setUp(self):
        common_setups()

    def test_dataset_benchmark(self):
        train_MNIST = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True, download=True
        )
        test_MNIST = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False, download=True
        )

        train_cifar10 = CIFAR10(
            root=expanduser("~") + "/.avalanche/data/cifar10/",
            train=True, download=True
        )
        test_cifar10 = CIFAR10(
            root=expanduser("~") + "/.avalanche/data/cifar10/",
            train=False, download=True
        )

        generic_scenario = dataset_benchmark(
            [train_MNIST, train_cifar10],
            [test_MNIST, test_cifar10])

    def test_dataset_benchmark_avalanche_dataset(self):
        train_MNIST = AvalancheDataset(MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True, download=True
        ), task_labels=0)

        test_MNIST = AvalancheDataset(MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False, download=True
        ), task_labels=0)

        train_cifar10 = AvalancheDataset(CIFAR10(
            root=expanduser("~") + "/.avalanche/data/cifar10/",
            train=True, download=True
        ), task_labels=1)

        test_cifar10 = AvalancheDataset(CIFAR10(
            root=expanduser("~") + "/.avalanche/data/cifar10/",
            train=False, download=True
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
            'cats_and_dogs_filtered.zip', expanduser("~") + "/.avalanche/data",
            'cats_and_dogs_filtered.zip')
        archive_name = os.path.join(
            expanduser("~") + "/.avalanche/data", 'cats_and_dogs_filtered.zip')
        extract_archive(archive_name,
                        to_path=expanduser("~") + "/.avalanche/data/")

        dirpath = expanduser("~") + \
            "/.avalanche/data/cats_and_dogs_filtered/train"

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

    def test_paths_benchmark(self):
        download_url(
            'https://storage.googleapis.com/mledu-datasets/'
            'cats_and_dogs_filtered.zip', expanduser("~") + "/.avalanche/data",
            'cats_and_dogs_filtered.zip')
        archive_name = os.path.join(
            expanduser("~") + "/.avalanche/data", 'cats_and_dogs_filtered.zip')
        extract_archive(archive_name,
                        to_path=expanduser("~") + "/.avalanche/data/")

        dirpath = expanduser("~") + \
            "/.avalanche/data/cats_and_dogs_filtered/train"

        train_experiences = []
        for rel_dir, label in zip(
                ["cats", "dogs"],
                [0, 1]):
            filenames_list = os.listdir(os.path.join(dirpath, rel_dir))

            experience_paths = []
            for name in filenames_list:
                instance_tuple = (os.path.join(dirpath, rel_dir, name), label)
                experience_paths.append(instance_tuple)
            train_experiences.append(experience_paths)

        generic_scenario = paths_benchmark(
            train_experiences,
            [train_experiences[0]],  # Single test set
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

    def test_data_incremental_benchmark(self):
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

        initial_benchmark_instance = tensors_benchmark(
            train_tensors=[(experience_1_x, experience_1_y),
                           (experience_2_x, experience_2_y)],
            test_tensors=[(test_x, test_y)],
            task_labels=[0, 0],  # Task label of each train exp
            complete_test_set_only=True)

        data_incremental_instance = data_incremental_benchmark(
            initial_benchmark_instance, 12, shuffle=False, drop_last=False)

        self.assertEqual(16, len(data_incremental_instance.train_stream))
        self.assertEqual(1, len(data_incremental_instance.test_stream))
        self.assertTrue(data_incremental_instance.complete_test_set_only)

        tensor_idx = 0
        ref_tensor_x = experience_1_x
        ref_tensor_y = experience_1_y
        for exp in data_incremental_instance.train_stream:
            if exp.current_experience == 8:
                # Last mini-exp from 1st exp
                self.assertEqual(4, len(exp.dataset))
            elif exp.current_experience == 15:
                # Last mini-exp from 2nd exp
                self.assertEqual(8, len(exp.dataset))
            else:
                # Other mini-exp
                self.assertEqual(12, len(exp.dataset))

            if tensor_idx >= 100:
                ref_tensor_x = experience_2_x
                ref_tensor_y = experience_2_y
                tensor_idx = 0

            for x, y, *_ in exp.dataset:
                self.assertTrue(torch.equal(ref_tensor_x[tensor_idx], x))
                self.assertTrue(torch.equal(ref_tensor_y[tensor_idx], y))
                tensor_idx += 1

        exp = data_incremental_instance.test_stream[0]
        self.assertEqual(50, len(exp.dataset))

        tensor_idx = 0
        for x, y, *_ in exp.dataset:
            self.assertTrue(torch.equal(test_x[tensor_idx], x))
            self.assertTrue(torch.equal(test_y[tensor_idx], y))
            tensor_idx += 1

    def test_benchmark_with_validation_stream_fixed_size(self):
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

        initial_benchmark_instance = tensors_benchmark(
            train_tensors=[(experience_1_x, experience_1_y),
                           (experience_2_x, experience_2_y)],
            test_tensors=[(test_x, test_y)],
            task_labels=[0, 0],  # Task label of each train exp
            complete_test_set_only=True)

        valid_benchmark = benchmark_with_validation_stream(
            initial_benchmark_instance, 20, shuffle=False)

        self.assertEqual(2, len(valid_benchmark.train_stream))
        self.assertEqual(2, len(valid_benchmark.valid_stream))
        self.assertEqual(1, len(valid_benchmark.test_stream))
        self.assertTrue(valid_benchmark.complete_test_set_only)

        self.assertEqual(80, len(valid_benchmark.train_stream[0].dataset))
        self.assertEqual(60, len(valid_benchmark.train_stream[1].dataset))
        self.assertEqual(20, len(valid_benchmark.valid_stream[0].dataset))
        self.assertEqual(20, len(valid_benchmark.valid_stream[1].dataset))

        self.assertTrue(
            torch.equal(
                experience_1_x[:80],
                valid_benchmark.train_stream[0].dataset[:][0]))

        self.assertTrue(
            torch.equal(
                experience_2_x[:60],
                valid_benchmark.train_stream[1].dataset[:][0]))

        self.assertTrue(
            torch.equal(
                experience_1_y[:80],
                valid_benchmark.train_stream[0].dataset[:][1]))

        self.assertTrue(
            torch.equal(
                experience_2_y[:60],
                valid_benchmark.train_stream[1].dataset[:][1]))

        self.assertTrue(
            torch.equal(
                experience_1_x[80:],
                valid_benchmark.valid_stream[0].dataset[:][0]))

        self.assertTrue(
            torch.equal(
                experience_2_x[60:],
                valid_benchmark.valid_stream[1].dataset[:][0]))

        self.assertTrue(
            torch.equal(
                experience_1_y[80:],
                valid_benchmark.valid_stream[0].dataset[:][1]))

        self.assertTrue(
            torch.equal(
                experience_2_y[60:],
                valid_benchmark.valid_stream[1].dataset[:][1]))

        self.assertTrue(
            torch.equal(
                test_x,
                valid_benchmark.test_stream[0].dataset[:][0]))

        self.assertTrue(
            torch.equal(
                test_y,
                valid_benchmark.test_stream[0].dataset[:][1]))

    def test_benchmark_with_validation_stream_rel_size(self):
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

        initial_benchmark_instance = tensors_benchmark(
            train_tensors=[(experience_1_x, experience_1_y),
                           (experience_2_x, experience_2_y)],
            test_tensors=[(test_x, test_y)],
            task_labels=[0, 0],  # Task label of each train exp
            complete_test_set_only=True)

        valid_benchmark = benchmark_with_validation_stream(
            initial_benchmark_instance, 0.2, shuffle=False)
        expected_rel_1_valid = int(100 * 0.2)
        expected_rel_1_train = 100 - expected_rel_1_valid
        expected_rel_2_valid = int(80 * 0.2)
        expected_rel_2_train = 80 - expected_rel_2_valid

        self.assertEqual(2, len(valid_benchmark.train_stream))
        self.assertEqual(2, len(valid_benchmark.valid_stream))
        self.assertEqual(1, len(valid_benchmark.test_stream))
        self.assertTrue(valid_benchmark.complete_test_set_only)

        self.assertEqual(
            expected_rel_1_train, len(valid_benchmark.train_stream[0].dataset))
        self.assertEqual(
            expected_rel_2_train, len(valid_benchmark.train_stream[1].dataset))
        self.assertEqual(
            expected_rel_1_valid, len(valid_benchmark.valid_stream[0].dataset))
        self.assertEqual(
            expected_rel_2_valid, len(valid_benchmark.valid_stream[1].dataset))

        self.assertTrue(
            torch.equal(
                experience_1_x[:expected_rel_1_train],
                valid_benchmark.train_stream[0].dataset[:][0]))

        self.assertTrue(
            torch.equal(
                experience_2_x[:expected_rel_2_train],
                valid_benchmark.train_stream[1].dataset[:][0]))

        self.assertTrue(
            torch.equal(
                experience_1_y[:expected_rel_1_train],
                valid_benchmark.train_stream[0].dataset[:][1]))

        self.assertTrue(
            torch.equal(
                experience_2_y[:expected_rel_2_train],
                valid_benchmark.train_stream[1].dataset[:][1]))

        self.assertTrue(
            torch.equal(
                experience_1_x[expected_rel_1_train:],
                valid_benchmark.valid_stream[0].dataset[:][0]))

        self.assertTrue(
            torch.equal(
                experience_2_x[expected_rel_2_train:],
                valid_benchmark.valid_stream[1].dataset[:][0]))

        self.assertTrue(
            torch.equal(
                experience_1_y[expected_rel_1_train:],
                valid_benchmark.valid_stream[0].dataset[:][1]))

        self.assertTrue(
            torch.equal(
                experience_2_y[expected_rel_2_train:],
                valid_benchmark.valid_stream[1].dataset[:][1]))

        self.assertTrue(
            torch.equal(
                test_x,
                valid_benchmark.test_stream[0].dataset[:][0]))

        self.assertTrue(
            torch.equal(
                test_y,
                valid_benchmark.test_stream[0].dataset[:][1]))
