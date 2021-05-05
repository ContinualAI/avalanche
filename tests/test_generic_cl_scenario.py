import unittest

import torch

from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheTensorDataset


class GenericCLScenarioTests(unittest.TestCase):
    def test_classes_in_exp(self):
        train_exps = []

        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 70, (200,))
        tensor_t = torch.randint(0, 5, (200,))
        train_exps.append(AvalancheTensorDataset(
            tensor_x, tensor_y, task_labels=tensor_t))

        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (200,))
        tensor_t = torch.randint(0, 5, (200,))
        train_exps.append(AvalancheTensorDataset(
            tensor_x, tensor_y, task_labels=tensor_t))

        test_exps = []
        test_x = torch.rand(200, 3, 28, 28)
        test_y = torch.randint(100, 200, (200,))
        test_t = torch.randint(0, 5, (200,))
        test_exps.append(AvalancheTensorDataset(
            test_x, test_y, task_labels=test_t))

        other_stream_exps = []
        other_x = torch.rand(200, 3, 28, 28)
        other_y = torch.randint(400, 600, (200,))
        other_t = torch.randint(0, 5, (200,))
        other_stream_exps.append(AvalancheTensorDataset(
            other_x, other_y, task_labels=other_t))

        benchmark_instance = dataset_benchmark(
            train_datasets=train_exps,
            test_datasets=test_exps,
            other_streams_datasets={'other': other_stream_exps})

        train_0_classes = benchmark_instance.classes_in_experience['train'][0]
        train_1_classes = benchmark_instance.classes_in_experience['train'][1]
        train_0_classes_min = min(train_0_classes)
        train_1_classes_min = min(train_1_classes)
        train_0_classes_max = max(train_0_classes)
        train_1_classes_max = max(train_1_classes)
        self.assertGreaterEqual(train_0_classes_min, 0)
        self.assertLess(train_0_classes_max, 70)
        self.assertGreaterEqual(train_1_classes_min, 0)
        self.assertLess(train_1_classes_max, 100)

        # Test deprecated behavior
        train_0_classes = benchmark_instance.classes_in_experience[0]
        train_1_classes = benchmark_instance.classes_in_experience[1]
        train_0_classes_min = min(train_0_classes)
        train_1_classes_min = min(train_1_classes)
        train_0_classes_max = max(train_0_classes)
        train_1_classes_max = max(train_1_classes)
        self.assertGreaterEqual(train_0_classes_min, 0)
        self.assertLess(train_0_classes_max, 70)
        self.assertGreaterEqual(train_1_classes_min, 0)
        self.assertLess(train_1_classes_max, 100)
        # End test deprecated behavior

        test_0_classes = benchmark_instance.classes_in_experience['test'][0]
        test_0_classes_min = min(test_0_classes)
        test_0_classes_max = max(test_0_classes)
        self.assertGreaterEqual(test_0_classes_min, 100)
        self.assertLess(test_0_classes_max, 200)

        other_0_classes = benchmark_instance.classes_in_experience['other'][0]
        other_0_classes_min = min(other_0_classes)
        other_0_classes_max = max(other_0_classes)
        self.assertGreaterEqual(other_0_classes_min, 400)
        self.assertLess(other_0_classes_max, 600)


if __name__ == '__main__':
    unittest.main()
