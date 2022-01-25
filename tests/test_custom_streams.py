import unittest

from os.path import expanduser

import torch
from torchvision.datasets import MNIST

from avalanche.benchmarks.scenarios.new_classes import NCExperience
from avalanche.benchmarks.utils import AvalancheSubset, AvalancheTensorDataset
from avalanche.benchmarks.scenarios.new_classes.nc_utils import (
    make_nc_transformation_subset,
)
from avalanche.benchmarks import (
    nc_benchmark,
    GenericScenarioStream,
    GenericCLScenario,
)


class CustomStreamsTests(unittest.TestCase):
    def test_custom_streams_name_and_length(self):

        train_exps = []
        test_exps = []
        valid_exps = []

        for _ in range(5):
            tensor_x = torch.rand(200, 3, 28, 28)
            tensor_y = torch.randint(0, 100, (200,))
            tensor_t = torch.randint(0, 5, (200,))
            train_exps.append(
                AvalancheTensorDataset(tensor_x, tensor_y, task_labels=tensor_t)
            )

        for _ in range(3):
            tensor_x = torch.rand(150, 3, 28, 28)
            tensor_y = torch.randint(0, 100, (150,))
            tensor_t = torch.randint(0, 3, (150,))
            test_exps.append(
                AvalancheTensorDataset(tensor_x, tensor_y, task_labels=tensor_t)
            )

        for _ in range(4):
            tensor_x = torch.rand(220, 3, 28, 28)
            tensor_y = torch.randint(0, 100, (220,))
            tensor_t = torch.randint(0, 5, (220,))
            valid_exps.append(
                AvalancheTensorDataset(tensor_x, tensor_y, task_labels=tensor_t)
            )

        valid_origin_dataset = AvalancheTensorDataset(
            torch.ones(10, 3, 32, 32), torch.zeros(10)
        )

        valid_t_labels = [{9}, {4, 5}, {7, 8}, {0}, {3}]

        with self.assertRaises(Exception):
            benchmark_instance = GenericCLScenario(
                stream_definitions={
                    "train": (train_exps,),
                    "test": (test_exps,),
                    "valid": (valid_exps, valid_t_labels, valid_origin_dataset),
                }
            )

        valid_t_labels = valid_t_labels[:-1]

        benchmark_instance = GenericCLScenario(
            stream_definitions={
                "train": (train_exps,),
                "test": (test_exps,),
                "valid": (valid_exps, valid_t_labels, valid_origin_dataset),
            }
        )

        self.assertEqual(5, len(benchmark_instance.train_stream))
        self.assertEqual(3, len(benchmark_instance.test_stream))
        self.assertEqual(4, len(benchmark_instance.valid_stream))

        self.assertEqual(None, benchmark_instance.original_train_dataset)
        self.assertEqual(None, benchmark_instance.original_test_dataset)
        self.assertEqual(
            valid_origin_dataset, benchmark_instance.original_valid_dataset
        )

        for i, exp in enumerate(benchmark_instance.train_stream):
            expect_x, expect_y, expect_t = train_exps[i][0]
            got_x, got_y, got_t = exp.dataset[0]

            self.assertTrue(torch.equal(expect_x, got_x))
            self.assertTrue(torch.equal(expect_y, got_y))
            self.assertEqual(int(expect_t), got_t)

            exp_t_labels = set(exp.task_labels)
            self.assertLess(max(exp_t_labels), 5)
            self.assertGreaterEqual(min(exp_t_labels), 0)

        for i, exp in enumerate(benchmark_instance.test_stream):
            expect_x, expect_y, expect_t = test_exps[i][0]
            got_x, got_y, got_t = exp.dataset[0]

            self.assertTrue(torch.equal(expect_x, got_x))
            self.assertTrue(torch.equal(expect_y, got_y))
            self.assertEqual(int(expect_t), got_t)

            exp_t_labels = set(exp.task_labels)
            self.assertLess(max(exp_t_labels), 3)
            self.assertGreaterEqual(min(exp_t_labels), 0)

        for i, exp in enumerate(benchmark_instance.valid_stream):
            expect_x, expect_y, expect_t = valid_exps[i][0]
            got_x, got_y, got_t = exp.dataset[0]

            self.assertTrue(torch.equal(expect_x, got_x))
            self.assertTrue(torch.equal(expect_y, got_y))
            self.assertEqual(int(expect_t), got_t)

            exp_t_labels = set(exp.task_labels)

            self.assertEqual(valid_t_labels[i], exp_t_labels)

    def test_complete_test_set_only(self):
        train_exps = []
        test_exps = []

        for _ in range(5):
            tensor_x = torch.rand(200, 3, 28, 28)
            tensor_y = torch.randint(0, 100, (200,))
            tensor_t = torch.randint(0, 5, (200,))
            train_exps.append(
                AvalancheTensorDataset(tensor_x, tensor_y, task_labels=tensor_t)
            )

        for _ in range(3):
            tensor_x = torch.rand(150, 3, 28, 28)
            tensor_y = torch.randint(0, 100, (150,))
            tensor_t = torch.randint(0, 5, (150,))
            test_exps.append(
                AvalancheTensorDataset(tensor_x, tensor_y, task_labels=tensor_t)
            )

        with self.assertRaises(Exception):
            benchmark_instance = GenericCLScenario(
                stream_definitions={
                    "train": (train_exps,),
                    "test": (test_exps,),
                },
                complete_test_set_only=True,
            )

        benchmark_instance = GenericCLScenario(
            stream_definitions={
                "train": (train_exps,),
                "test": (test_exps[0],),
            },
            complete_test_set_only=True,
        )

        self.assertEqual(5, len(benchmark_instance.train_stream))
        self.assertEqual(1, len(benchmark_instance.test_stream))


if __name__ == "__main__":
    unittest.main()
