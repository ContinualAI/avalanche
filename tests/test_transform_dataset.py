import unittest

import torch
from PIL import ImageChops
from PIL.Image import Image
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.training.utils import TransformationDataset, \
    TransformationSubset, load_all_dataset
import random

from avalanche.benchmarks.scenarios.generic_scenario_creation import \
    create_generic_scenario_from_tensors


def pil_images_equal(img_a, img_b):
    diff = ImageChops.difference(img_a, img_b)

    return not diff.getbbox()


class TransformationDatasetTests(unittest.TestCase):
    def test_mnist_no_transforms(self):
        dataset = MNIST('./data/mnist', download=True)
        x, y = dataset[0]
        self.assertIsInstance(x, Image)
        self.assertEqual([x.width, x.height], [28, 28])
        self.assertIsInstance(y, int)

    def test_mnist_native_transforms(self):
        dataset = MNIST('./data/mnist', download=True, transform=ToTensor())
        x, y = dataset[0]
        self.assertIsInstance(x, Tensor)
        self.assertEqual(x.shape, (1, 28, 28))
        self.assertIsInstance(y, int)

    def test_transform_dataset_transform(self):
        dataset = MNIST('./data/mnist', download=True)
        x, y = dataset[0]
        dataset = TransformationDataset(dataset, transform=ToTensor())
        x2, y2 = dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, Tensor)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2.item())

    def test_transform_dataset_composition(self):
        dataset = MNIST('./data/mnist', download=True, transform=RandomCrop(16))
        x, y = dataset[0]
        self.assertIsInstance(x, Image)
        self.assertEqual([x.width, x.height], [16, 16])
        self.assertIsInstance(y, int)

        dataset = TransformationDataset(
            dataset, transform=ToTensor(),
            target_transform=lambda target: -1)

        x2, y2 = dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertEqual(x2.shape, (1, 16, 16))
        self.assertIsInstance(y2, Tensor)
        self.assertEqual(y2.item(), -1)


class TransformationSubsetTests(unittest.TestCase):
    def test_transform_subset_transform(self):
        dataset = MNIST('./data/mnist', download=True)
        x, y = dataset[0]
        dataset = TransformationSubset(dataset, None, transform=ToTensor())
        x2, y2 = dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, Tensor)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2.item())

    def test_transform_subset_composition(self):
        dataset = MNIST('./data/mnist', download=True, transform=RandomCrop(16))
        x, y = dataset[0]
        self.assertIsInstance(x, Image)
        self.assertEqual([x.width, x.height], [16, 16])
        self.assertIsInstance(y, int)

        dataset = TransformationSubset(
            dataset, None, transform=ToTensor(),
            target_transform=lambda target: -1)

        x2, y2 = dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertEqual(x2.shape, (1, 16, 16))
        self.assertIsInstance(y2, Tensor)
        self.assertEqual(y2.item(), -1)

    def test_transform_subset_indicies(self):
        dataset = MNIST('./data/mnist', download=True)
        x, y = dataset[1000]
        x2, y2 = dataset[1007]

        dataset = TransformationSubset(
            dataset, [1000, 1007])

        x3, y3 = dataset[0]
        x4, y4 = dataset[1]
        self.assertTrue(pil_images_equal(x, x3))
        self.assertEqual(y, y3)
        self.assertTrue(pil_images_equal(x2, x4))
        self.assertEqual(y2, y4)
        self.assertFalse(pil_images_equal(x, x4))
        self.assertFalse(pil_images_equal(x2, x3))

    def test_transform_subset_mapping(self):
        dataset = MNIST('./data/mnist', download=True)
        _, y = dataset[1000]

        mapping = list(range(10))
        other_classes = list(mapping)
        other_classes.remove(y)

        swap_y = random.choice(other_classes)

        mapping[y] = swap_y
        mapping[swap_y] = y

        dataset = TransformationSubset(dataset, None, class_mapping=mapping)

        _, y2 = dataset[1000]
        self.assertEqual(y2, swap_y)


class TransformationTensorDatasetTests(unittest.TestCase):
    def test_tensor_dataset_helper_tensor_y(self):
        dataset_train_x = [torch.rand(50, 32, 32) for _ in range(5)]
        dataset_train_y = [torch.randint(0, 100, (50,)) for _ in range(5)]

        dataset_test_x = [torch.rand(23, 32, 32) for _ in range(5)]
        dataset_test_y = [torch.randint(0, 100, (23,)) for _ in range(5)]

        cl_scenario = create_generic_scenario_from_tensors(
            dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,
            [0] * 5)

        self.assertEqual(5, len(cl_scenario))

        for step_id, step in enumerate(cl_scenario):
            scenario_train_x, scenario_train_y = \
                load_all_dataset(step.current_training_set()[0])
            scenario_test_x, scenario_test_y = \
                load_all_dataset(step.current_test_set()[0])

            self.assertTrue(torch.all(torch.eq(
                dataset_train_x[step_id],
                scenario_train_x)))
            self.assertTrue(torch.all(torch.eq(
                dataset_train_y[step_id],
                scenario_train_y)))
            self.assertSequenceEqual(
                dataset_train_y[step_id].tolist(),
                step.current_training_set()[0].targets)
            self.assertEqual(0, step.current_training_set()[1])  # Task label

            self.assertTrue(torch.all(torch.eq(
                dataset_test_x[step_id],
                scenario_test_x)))
            self.assertTrue(torch.all(torch.eq(
                dataset_test_y[step_id],
                scenario_test_y)))
            self.assertSequenceEqual(
                dataset_test_y[step_id].tolist(),
                step.current_test_set()[0].targets)
            self.assertEqual(0, step.current_test_set()[1])  # Task label

    def test_tensor_dataset_helper_list_y(self):
        dataset_train_x = [torch.rand(50, 32, 32) for _ in range(5)]
        dataset_train_y = [torch.randint(0, 100, (50,)).tolist()
                           for _ in range(5)]

        dataset_test_x = [torch.rand(23, 32, 32) for _ in range(5)]
        dataset_test_y = [torch.randint(0, 100, (23,)).tolist()
                          for _ in range(5)]

        cl_scenario = create_generic_scenario_from_tensors(
            dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y,
            [0] * 5)

        self.assertEqual(5, len(cl_scenario))

        for step_id, step in enumerate(cl_scenario):
            scenario_train_x, scenario_train_y = \
                load_all_dataset(step.current_training_set()[0])
            scenario_test_x, scenario_test_y = \
                load_all_dataset(step.current_test_set()[0])

            self.assertTrue(torch.all(torch.eq(
                dataset_train_x[step_id],
                scenario_train_x)))
            self.assertSequenceEqual(
                dataset_train_y[step_id],
                scenario_train_y.tolist())
            self.assertSequenceEqual(
                dataset_train_y[step_id],
                step.current_training_set()[0].targets)
            self.assertEqual(0, step.current_training_set()[1])  # Task label

            self.assertTrue(torch.all(torch.eq(
                dataset_test_x[step_id],
                scenario_test_x)))
            self.assertSequenceEqual(
                dataset_test_y[step_id],
                scenario_test_y.tolist())
            self.assertSequenceEqual(
                dataset_test_y[step_id],
                step.current_test_set()[0].targets)
            self.assertEqual(0, step.current_test_set()[1])  # Task label


if __name__ == '__main__':
    unittest.main()
