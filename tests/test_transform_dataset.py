import unittest

import torch
from PIL import ImageChops
from PIL.Image import Image
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop, ToPILImage

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
        self.assertIsInstance(y2, int)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2)

    def test_transform_dataset_slice(self):
        dataset = MNIST('./data/mnist', download=True)
        x0, y0 = dataset[0]
        x1, y1 = dataset[1]
        dataset = TransformationDataset(dataset, transform=ToTensor())
        x2, y2 = dataset[:2]
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, Tensor)
        self.assertTrue(torch.equal(ToTensor()(x0), x2[0]))
        self.assertTrue(torch.equal(ToTensor()(x1), x2[1]))
        self.assertEqual(y0, y2[0].item())
        self.assertEqual(y1, y2[1].item())

    def test_transform_dataset_indexing(self):
        dataset = MNIST('./data/mnist', download=True)
        x0, y0 = dataset[0]
        x1, y1 = dataset[5]
        dataset = TransformationDataset(dataset, transform=ToTensor())
        x2, y2 = dataset[0, 5]
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, Tensor)
        self.assertTrue(torch.equal(ToTensor()(x0), x2[0]))
        self.assertTrue(torch.equal(ToTensor()(x1), x2[1]))
        self.assertEqual(y0, y2[0].item())
        self.assertEqual(y1, y2[1].item())

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
        self.assertIsInstance(y2, int)
        self.assertEqual(y2, -1)


class TransformationSubsetTests(unittest.TestCase):
    def test_transform_subset_transform(self):
        dataset = MNIST('./data/mnist', download=True)
        x, y = dataset[0]
        dataset = TransformationSubset(dataset, transform=ToTensor())
        x2, y2 = dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, int)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2)

    def test_transform_subset_composition(self):
        dataset = MNIST('./data/mnist', download=True, transform=RandomCrop(16))
        x, y = dataset[0]
        self.assertIsInstance(x, Image)
        self.assertEqual([x.width, x.height], [16, 16])
        self.assertIsInstance(y, int)

        dataset = TransformationSubset(
            dataset, transform=ToTensor(),
            target_transform=lambda target: -1)

        x2, y2 = dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertEqual(x2.shape, (1, 16, 16))
        self.assertIsInstance(y2, int)
        self.assertEqual(y2, -1)

    def test_transform_subset_indicies(self):
        dataset = MNIST('./data/mnist', download=True)
        x, y = dataset[1000]
        x2, y2 = dataset[1007]

        dataset = TransformationSubset(
            dataset, indices=[1000, 1007])

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

        dataset = TransformationSubset(dataset, class_mapping=mapping)

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

        self.assertEqual(5, len(cl_scenario.train_stream))
        self.assertEqual(5, len(cl_scenario.test_stream))
        self.assertEqual(5, cl_scenario.n_steps)

        for step_id in range(cl_scenario.n_steps):
            scenario_train_x, scenario_train_y = \
                load_all_dataset(cl_scenario.train_stream[step_id].dataset)
            scenario_test_x, scenario_test_y = \
                load_all_dataset(cl_scenario.test_stream[step_id].dataset)

            self.assertTrue(torch.all(torch.eq(
                dataset_train_x[step_id],
                scenario_train_x)))
            self.assertTrue(torch.all(torch.eq(
                dataset_train_y[step_id],
                scenario_train_y)))
            self.assertSequenceEqual(
                dataset_train_y[step_id].tolist(),
                cl_scenario.train_stream[step_id].dataset.targets)
            self.assertEqual(0, cl_scenario.train_stream[step_id].task_label)

            self.assertTrue(torch.all(torch.eq(
                dataset_test_x[step_id],
                scenario_test_x)))
            self.assertTrue(torch.all(torch.eq(
                dataset_test_y[step_id],
                scenario_test_y)))
            self.assertSequenceEqual(
                dataset_test_y[step_id].tolist(),
                cl_scenario.test_stream[step_id].dataset.targets)
            self.assertEqual(0, cl_scenario.test_stream[step_id].task_label)

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

        self.assertEqual(5, len(cl_scenario.train_stream))
        self.assertEqual(5, len(cl_scenario.test_stream))
        self.assertEqual(5, cl_scenario.n_steps)

        for step_id in range(cl_scenario.n_steps):
            scenario_train_x, scenario_train_y = \
                load_all_dataset(cl_scenario.train_stream[step_id].dataset)
            scenario_test_x, scenario_test_y = \
                load_all_dataset(cl_scenario.test_stream[step_id].dataset)

            self.assertTrue(torch.all(torch.eq(
                dataset_train_x[step_id],
                scenario_train_x)))
            self.assertSequenceEqual(
                dataset_train_y[step_id],
                scenario_train_y.tolist())
            self.assertSequenceEqual(
                dataset_train_y[step_id],
                cl_scenario.train_stream[step_id].dataset.targets)
            self.assertEqual(0, cl_scenario.train_stream[step_id].task_label)

            self.assertTrue(torch.all(torch.eq(
                dataset_test_x[step_id],
                scenario_test_x)))
            self.assertSequenceEqual(
                dataset_test_y[step_id],
                scenario_test_y.tolist())
            self.assertSequenceEqual(
                dataset_test_y[step_id],
                cl_scenario.test_stream[step_id].dataset.targets)
            self.assertEqual(0, cl_scenario.test_stream[step_id].task_label)


class TransformationDatasetTransformationsChainTests(unittest.TestCase):
    def test_freeze_transforms(self):
        original_dataset = MNIST('./data/mnist', download=True)
        x, y = original_dataset[0]
        dataset = TransformationDataset(original_dataset, transform=ToTensor())
        dataset_frozen = dataset.freeze_transforms()
        dataset_frozen.transform = None

        x2, y2 = dataset_frozen[0]
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, int)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2)

        dataset.transform = None
        x2, y2 = dataset[0]
        self.assertIsInstance(x2, Image)

        x2, y2 = dataset_frozen[0]
        self.assertIsInstance(x2, Tensor)

    def test_freeze_transforms_chain(self):
        dataset = MNIST('./data/mnist', download=True, transform=ToTensor())
        x, _ = dataset[0]
        self.assertIsInstance(x, Tensor)

        dataset_transform = TransformationDataset(dataset,
                                                  transform=ToPILImage())
        x, _ = dataset_transform[0]
        self.assertIsInstance(x, Image)

        dataset_frozen = dataset_transform.freeze_transforms()

        x2, _ = dataset_frozen[0]
        self.assertIsInstance(x2, Image)

        dataset_transform.transform = None

        x2, _ = dataset_transform[0]
        self.assertIsInstance(x2, Tensor)

        dataset_frozen.transform = ToTensor()

        x2, _ = dataset_frozen[0]
        self.assertIsInstance(x2, Tensor)

        dataset_frozen2 = dataset_frozen.freeze_transforms()

        x2, _ = dataset_frozen2[0]
        self.assertIsInstance(x2, Tensor)

        dataset_frozen.transform = None

        x2, _ = dataset_frozen2[0]
        self.assertIsInstance(x2, Tensor)
        x2, _ = dataset_frozen[0]
        self.assertIsInstance(x2, Image)


if __name__ == '__main__':
    unittest.main()
