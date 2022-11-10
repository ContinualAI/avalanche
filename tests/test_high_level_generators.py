import os
import tempfile
import unittest
from os.path import expanduser

import torch
from numpy.testing import assert_almost_equal
from torchvision.datasets import MNIST
from torchvision.datasets.utils import download_url, extract_archive
from torchvision.transforms import ToTensor
from tests.unit_tests_utils import DummyImageDataset


from avalanche.benchmarks import (
    dataset_benchmark,
    filelist_benchmark,
    tensors_benchmark,
    paths_benchmark,
    data_incremental_benchmark,
    benchmark_with_validation_stream,
)
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.generators.benchmark_generators import (
    class_balanced_split_strategy,
)
from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    create_lazy_generic_benchmark,
    LazyStreamDefinition,
)
from avalanche.benchmarks.utils import (
    make_classification_dataset,
    make_tensor_classification_dataset,
)
from tests.test_avalanche_classification_dataset import get_mbatch
from tests.unit_tests_utils import common_setups, get_fast_benchmark


class HighLevelGeneratorTests(unittest.TestCase):
    def setUp(self):
        common_setups()

    def test_dataset_benchmark(self):
        train_MNIST = MNIST(
            root=default_dataset_location("mnist"), train=True, download=True
        )
        test_MNIST = MNIST(
            root=default_dataset_location("mnist"), train=False, download=True
        )

        train_cifar10 = DummyImageDataset(n_classes=10)
        test_cifar10 = DummyImageDataset(n_classes=10)

        generic_benchmark = dataset_benchmark(
            [train_MNIST, train_cifar10], [test_MNIST, test_cifar10]
        )

    def test_dataset_benchmark_avalanche_dataset(self):
        train_MNIST = make_classification_dataset(
            MNIST(
                root=default_dataset_location("mnist"),
                train=True,
                download=True,
            ),
            task_labels=0,
        )

        test_MNIST = make_classification_dataset(
            MNIST(
                root=default_dataset_location("mnist"),
                train=False,
                download=True,
            ),
            task_labels=0,
        )

        train_cifar10 = make_classification_dataset(
            DummyImageDataset(n_classes=10),
            task_labels=1,
        )

        test_cifar10 = make_classification_dataset(
            DummyImageDataset(n_classes=10),
            task_labels=1,
        )

        generic_benchmark = dataset_benchmark(
            [train_MNIST, train_cifar10], [test_MNIST, test_cifar10]
        )

        self.assertEqual(0, generic_benchmark.train_stream[0].task_label)
        self.assertEqual(1, generic_benchmark.train_stream[1].task_label)
        self.assertEqual(0, generic_benchmark.test_stream[0].task_label)
        self.assertEqual(1, generic_benchmark.test_stream[1].task_label)

    def test_filelist_benchmark(self):
        download_url(
            "https://storage.googleapis.com/mledu-datasets/"
            "cats_and_dogs_filtered.zip",
            expanduser("~") + "/.avalanche/data",
            "cats_and_dogs_filtered.zip",
        )
        archive_name = os.path.join(
            expanduser("~") + "/.avalanche/data", "cats_and_dogs_filtered.zip"
        )
        extract_archive(
            archive_name, to_path=expanduser("~") + "/.avalanche/data/"
        )

        dirpath = (
            expanduser("~") + "/.avalanche/data/cats_and_dogs_filtered/train"
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            list_paths = []
            for filelist, rel_dir, label in zip(
                ["train_filelist_00.txt", "train_filelist_01.txt"],
                ["cats", "dogs"],
                [0, 1],
            ):
                # First, obtain the list of files
                filenames_list = os.listdir(os.path.join(dirpath, rel_dir))
                filelist_path = os.path.join(tmpdirname, filelist)
                list_paths.append(filelist_path)
                with open(filelist_path, "w") as wf:
                    for name in filenames_list:
                        wf.write(
                            "{} {}\n".format(os.path.join(rel_dir, name), label)
                        )

            generic_benchmark = filelist_benchmark(
                dirpath,
                list_paths,
                [list_paths[0]],
                task_labels=[0, 0],
                complete_test_set_only=True,
                train_transform=ToTensor(),
                eval_transform=ToTensor(),
            )

        self.assertEqual(2, len(generic_benchmark.train_stream))
        self.assertEqual(1, len(generic_benchmark.test_stream))

    def test_paths_benchmark(self):
        download_url(
            "https://storage.googleapis.com/mledu-datasets/"
            "cats_and_dogs_filtered.zip",
            expanduser("~") + "/.avalanche/data",
            "cats_and_dogs_filtered.zip",
        )
        archive_name = os.path.join(
            expanduser("~") + "/.avalanche/data", "cats_and_dogs_filtered.zip"
        )
        extract_archive(
            archive_name, to_path=expanduser("~") + "/.avalanche/data/"
        )

        dirpath = (
            expanduser("~") + "/.avalanche/data/cats_and_dogs_filtered/train"
        )

        train_experiences = []
        for rel_dir, label in zip(["cats", "dogs"], [0, 1]):
            filenames_list = os.listdir(os.path.join(dirpath, rel_dir))

            experience_paths = []
            for name in filenames_list:
                instance_tuple = (os.path.join(dirpath, rel_dir, name), label)
                experience_paths.append(instance_tuple)
            train_experiences.append(experience_paths)

        generic_benchmark = paths_benchmark(
            train_experiences,
            [train_experiences[0]],  # Single test set
            task_labels=[0, 0],
            complete_test_set_only=True,
            train_transform=ToTensor(),
            eval_transform=ToTensor(),
        )

        self.assertEqual(2, len(generic_benchmark.train_stream))
        self.assertEqual(1, len(generic_benchmark.test_stream))

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

        generic_benchmark = tensors_benchmark(
            train_tensors=[
                (experience_1_x, experience_1_y),
                (experience_2_x, experience_2_y),
            ],
            test_tensors=[(test_x, test_y)],
            task_labels=[0, 0],  # Task label of each train exp
            complete_test_set_only=True,
        )

        self.assertEqual(2, len(generic_benchmark.train_stream))
        self.assertEqual(1, len(generic_benchmark.test_stream))

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
            train_tensors=[
                (experience_1_x, experience_1_y),
                (experience_2_x, experience_2_y),
            ],
            test_tensors=[(test_x, test_y)],
            task_labels=[0, 0],  # Task label of each train exp
            complete_test_set_only=True,
        )

        data_incremental_instance = data_incremental_benchmark(
            initial_benchmark_instance, 12, shuffle=False, drop_last=False
        )

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
                self.assertTrue(
                    torch.equal(ref_tensor_y[tensor_idx], torch.tensor(y))
                )
                tensor_idx += 1

        exp = data_incremental_instance.test_stream[0]
        self.assertEqual(50, len(exp.dataset))

        tensor_idx = 0
        for x, y, *_ in exp.dataset:
            self.assertTrue(torch.equal(test_x[tensor_idx], x))
            self.assertTrue(torch.equal(test_y[tensor_idx], torch.tensor(y)))
            tensor_idx += 1

    def test_data_incremental_benchmark_from_lazy_benchmark(self):
        pattern_shape = (3, 32, 32)

        # Definition of training experiences
        # Experience 1
        experience_1_x = torch.zeros(100, *pattern_shape)
        experience_1_y = torch.zeros(100, dtype=torch.long)
        experience_1_dataset = make_tensor_classification_dataset(
            experience_1_x, experience_1_y
        )

        # Experience 2
        experience_2_x = torch.zeros(80, *pattern_shape)
        experience_2_y = torch.ones(80, dtype=torch.long)
        experience_2_dataset = make_tensor_classification_dataset(
            experience_2_x, experience_2_y
        )

        # Test experience
        test_x = torch.zeros(50, *pattern_shape)
        test_y = torch.zeros(50, dtype=torch.long)
        experience_test = make_tensor_classification_dataset(test_x, test_y)

        def train_gen():
            # Lazy generator of the training stream
            for dataset in [experience_1_dataset, experience_2_dataset]:
                yield dataset

        def test_gen():
            # Lazy generator of the test stream
            for dataset in [experience_test]:
                yield dataset

        initial_benchmark_instance = create_lazy_generic_benchmark(
            train_generator=LazyStreamDefinition(train_gen(), 2, [0, 0]),
            test_generator=LazyStreamDefinition(test_gen(), 1, [0]),
            complete_test_set_only=True,
        )

        data_incremental_instance = data_incremental_benchmark(
            initial_benchmark_instance, 12, shuffle=False, drop_last=False
        )

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
                self.assertTrue(
                    torch.equal(ref_tensor_y[tensor_idx], torch.tensor(y))
                )
                tensor_idx += 1

        exp = data_incremental_instance.test_stream[0]
        self.assertEqual(50, len(exp.dataset))

        tensor_idx = 0
        for x, y, *_ in exp.dataset:
            self.assertTrue(torch.equal(test_x[tensor_idx], x))
            self.assertTrue(torch.equal(test_y[tensor_idx], torch.tensor(y)))
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
            train_tensors=[
                (experience_1_x, experience_1_y),
                (experience_2_x, experience_2_y),
            ],
            test_tensors=[(test_x, test_y)],
            task_labels=[0, 0],  # Task label of each train exp
            complete_test_set_only=True,
        )

        valid_benchmark = benchmark_with_validation_stream(
            initial_benchmark_instance, 20, shuffle=False
        )

        self.assertEqual(2, len(valid_benchmark.train_stream))
        self.assertEqual(2, len(valid_benchmark.valid_stream))
        self.assertEqual(1, len(valid_benchmark.test_stream))
        self.assertTrue(valid_benchmark.complete_test_set_only)

        self.assertEqual(80, len(valid_benchmark.train_stream[0].dataset))
        self.assertEqual(60, len(valid_benchmark.train_stream[1].dataset))
        self.assertEqual(20, len(valid_benchmark.valid_stream[0].dataset))
        self.assertEqual(20, len(valid_benchmark.valid_stream[1].dataset))

        vd = valid_benchmark.train_stream[0].dataset
        mb = get_mbatch(vd, len(vd))
        self.assertTrue(torch.equal(experience_1_x[:80], mb[0]))
        self.assertTrue(torch.equal(experience_1_y[:80], mb[1]))

        vd = valid_benchmark.train_stream[1].dataset
        mb = get_mbatch(vd, len(vd))
        self.assertTrue(torch.equal(experience_2_x[:60], mb[0]))
        self.assertTrue(torch.equal(experience_2_y[:60], mb[1]))

        vd = valid_benchmark.valid_stream[0].dataset
        mb = get_mbatch(vd, len(vd))
        self.assertTrue(torch.equal(experience_1_x[80:], mb[0]))
        self.assertTrue(torch.equal(experience_1_y[80:], mb[1]))

        vd = valid_benchmark.valid_stream[1].dataset
        mb = get_mbatch(vd, len(vd))
        self.assertTrue(torch.equal(experience_2_x[60:], mb[0]))
        self.assertTrue(torch.equal(experience_2_y[60:], mb[1]))

        vd = valid_benchmark.test_stream[0].dataset
        mb = get_mbatch(vd, len(vd))
        self.assertTrue(torch.equal(test_x, mb[0]))
        self.assertTrue(torch.equal(test_y, mb[1]))

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
            train_tensors=[
                (experience_1_x, experience_1_y),
                (experience_2_x, experience_2_y),
            ],
            test_tensors=[(test_x, test_y)],
            task_labels=[0, 0],  # Task label of each train exp
            complete_test_set_only=True,
        )

        valid_benchmark = benchmark_with_validation_stream(
            initial_benchmark_instance, 0.2, shuffle=False
        )
        true_rel_1_valid = int(100 * 0.2)
        true_rel_1_train = 100 - true_rel_1_valid
        true_rel_2_valid = int(80 * 0.2)
        true_rel_2_train = 80 - true_rel_2_valid

        self.assertEqual(2, len(valid_benchmark.train_stream))
        self.assertEqual(2, len(valid_benchmark.valid_stream))
        self.assertEqual(1, len(valid_benchmark.test_stream))
        self.assertTrue(valid_benchmark.complete_test_set_only)

        ts = valid_benchmark.train_stream
        self.assertEqual(true_rel_1_train, len(ts[0].dataset))
        self.assertEqual(true_rel_2_train, len(ts[1].dataset))

        stm = valid_benchmark.valid_stream
        self.assertEqual(true_rel_1_valid, len(stm[0].dataset))
        self.assertEqual(true_rel_2_valid, len(stm[1].dataset))

        dd = valid_benchmark.train_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_1_x[:true_rel_1_train], mb[0]))

        dd = valid_benchmark.train_stream[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_2_x[:true_rel_2_train], mb[0]))
        self.assertTrue(torch.equal(experience_2_y[:true_rel_2_train], mb[1]))

        dd = valid_benchmark.train_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_1_x[:true_rel_1_train], mb[0]))
        self.assertTrue(torch.equal(experience_1_y[:true_rel_1_train], mb[1]))

        dd = valid_benchmark.valid_stream[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_2_x[true_rel_2_train:], mb[0]))
        self.assertTrue(torch.equal(experience_2_y[true_rel_2_train:], mb[1]))

        dd = valid_benchmark.valid_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_1_y[true_rel_1_train:], mb[1]))

        dd = valid_benchmark.test_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(test_x, mb[0]))
        self.assertTrue(torch.equal(test_y, mb[1]))

    def test_lazy_benchmark_with_validation_stream_fixed_size(self):
        lazy_options = [None, True, False]
        for lazy_option in lazy_options:
            with self.subTest(lazy_option=lazy_option):
                pattern_shape = (3, 32, 32)

                # Definition of training experiences
                # Experience 1
                experience_1_x = torch.zeros(100, *pattern_shape)
                experience_1_y = torch.zeros(100, dtype=torch.long)
                experience_1_dataset = make_tensor_classification_dataset(
                    experience_1_x, experience_1_y
                )

                # Experience 2
                experience_2_x = torch.zeros(80, *pattern_shape)
                experience_2_y = torch.ones(80, dtype=torch.long)
                experience_2_dataset = make_tensor_classification_dataset(
                    experience_2_x, experience_2_y
                )

                # Test experience
                test_x = torch.zeros(50, *pattern_shape)
                test_y = torch.zeros(50, dtype=torch.long)
                experience_test = make_tensor_classification_dataset(
                    test_x, test_y
                )

                def train_gen():
                    # Lazy generator of the training stream
                    for dataset in [experience_1_dataset, experience_2_dataset]:
                        yield dataset

                def test_gen():
                    # Lazy generator of the test stream
                    for dataset in [experience_test]:
                        yield dataset

                initial_benchmark_instance = create_lazy_generic_benchmark(
                    train_generator=LazyStreamDefinition(
                        train_gen(), 2, [0, 0]
                    ),
                    test_generator=LazyStreamDefinition(test_gen(), 1, [0]),
                    complete_test_set_only=True,
                )

                valid_benchmark = benchmark_with_validation_stream(
                    initial_benchmark_instance,
                    20,
                    shuffle=False,
                    lazy_splitting=lazy_option,
                )

                if lazy_option is None or lazy_option:
                    expect_laziness = True
                else:
                    expect_laziness = False

                self.assertEqual(
                    expect_laziness,
                    valid_benchmark.stream_definitions["train"].is_lazy,
                )

                self.assertEqual(2, len(valid_benchmark.train_stream))
                self.assertEqual(2, len(valid_benchmark.valid_stream))
                self.assertEqual(1, len(valid_benchmark.test_stream))
                self.assertTrue(valid_benchmark.complete_test_set_only)

                maybe_exp = valid_benchmark.stream_definitions[
                    "train"
                ].exps_data.get_experience_if_loaded(0)
                self.assertEqual(expect_laziness, maybe_exp is None)

                self.assertEqual(
                    80, len(valid_benchmark.train_stream[0].dataset)
                )

                maybe_exp = valid_benchmark.stream_definitions[
                    "train"
                ].exps_data.get_experience_if_loaded(1)
                self.assertEqual(expect_laziness, maybe_exp is None)

                self.assertEqual(
                    60, len(valid_benchmark.train_stream[1].dataset)
                )

                maybe_exp = valid_benchmark.stream_definitions[
                    "valid"
                ].exps_data.get_experience_if_loaded(0)
                self.assertEqual(expect_laziness, maybe_exp is None)

                self.assertEqual(
                    20, len(valid_benchmark.valid_stream[0].dataset)
                )

                maybe_exp = valid_benchmark.stream_definitions[
                    "valid"
                ].exps_data.get_experience_if_loaded(1)
                self.assertEqual(expect_laziness, maybe_exp is None)

                self.assertEqual(
                    20, len(valid_benchmark.valid_stream[1].dataset)
                )

                self.assertIsNotNone(
                    valid_benchmark.stream_definitions[
                        "train"
                    ].exps_data.get_experience_if_loaded(0)
                )
                self.assertIsNotNone(
                    valid_benchmark.stream_definitions[
                        "valid"
                    ].exps_data.get_experience_if_loaded(0)
                )
                self.assertIsNotNone(
                    valid_benchmark.stream_definitions[
                        "train"
                    ].exps_data.get_experience_if_loaded(1)
                )
                self.assertIsNotNone(
                    valid_benchmark.stream_definitions[
                        "valid"
                    ].exps_data.get_experience_if_loaded(1)
                )

                dd = valid_benchmark.train_stream[0].dataset
                mb = get_mbatch(dd, len(dd))
                self.assertTrue(
                    torch.equal(
                        experience_1_x[:80],
                        mb[0],
                    )
                )

                dd = valid_benchmark.train_stream[1].dataset
                mb = get_mbatch(dd, len(dd))
                self.assertTrue(
                    torch.equal(
                        experience_2_x[:60],
                        mb[0],
                    )
                )

                dd = valid_benchmark.train_stream[0].dataset
                mb = get_mbatch(dd, len(dd))
                self.assertTrue(
                    torch.equal(
                        experience_1_y[:80],
                        mb[1],
                    )
                )

                dd = valid_benchmark.train_stream[1].dataset
                mb = get_mbatch(dd, len(dd))
                self.assertTrue(
                    torch.equal(
                        experience_2_y[:60],
                        mb[1],
                    )
                )

                dd = valid_benchmark.valid_stream[0].dataset
                mb = get_mbatch(dd, len(dd))
                self.assertTrue(torch.equal(experience_1_x[80:], mb[0]))

                dd = valid_benchmark.valid_stream[1].dataset
                mb = get_mbatch(dd, len(dd))
                self.assertTrue(torch.equal(experience_2_x[60:], mb[0]))

                dd = valid_benchmark.valid_stream[0].dataset
                mb = get_mbatch(dd, len(dd))
                self.assertTrue(torch.equal(experience_1_y[80:], mb[1]))

                dd = valid_benchmark.valid_stream[1].dataset
                mb = get_mbatch(dd, len(dd))
                self.assertTrue(torch.equal(experience_2_y[60:], mb[1]))

                dd = valid_benchmark.test_stream[0].dataset
                mb = get_mbatch(dd, len(dd))
                self.assertTrue(torch.equal(test_x, mb[0]))
                self.assertTrue(torch.equal(test_y, mb[1]))


class DataSplitStrategiesTests(unittest.TestCase):
    def test_dataset_benchmark(self):
        benchmark = get_fast_benchmark(n_samples_per_class=1000)
        exp = benchmark.train_stream[0]
        num_classes = len(exp.classes_in_this_experience)

        train_d, valid_d = class_balanced_split_strategy(0.5, exp)
        assert abs(len(train_d) - len(valid_d)) <= num_classes
        for cid in exp.classes_in_this_experience:
            train_cnt = (torch.as_tensor(train_d.targets) == cid).sum()
            valid_cnt = (torch.as_tensor(valid_d.targets) == cid).sum()
            assert abs(train_cnt - valid_cnt) <= 1

        ratio = 0.123
        len_data = len(exp.dataset)
        train_d, valid_d = class_balanced_split_strategy(ratio, exp)
        assert_almost_equal(len(valid_d) / len_data, ratio, decimal=2)
        for cid in exp.classes_in_this_experience:
            data_cnt = (torch.as_tensor(exp.dataset.targets) == cid).sum()
            valid_cnt = (torch.as_tensor(valid_d.targets) == cid).sum()
            assert_almost_equal(valid_cnt / data_cnt, ratio, decimal=2)
