import unittest

from PIL import ImageChops
from PIL.Image import Image
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.training.utils import TransformationDataset, TransformationSubset
import random

def pil_images_equal(a, b):
    diff = ImageChops.difference(a, b)

    if diff.getbbox():
        return False
    else:
        return True


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

        x2, y2 = dataset[1000]
        self.assertEqual(y2, swap_y)


if __name__ == '__main__':
    unittest.main()
