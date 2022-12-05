import unittest

from avalanche.benchmarks.datasets import default_dataset_location
import PIL
import torch
from PIL import ImageChops
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import ConcatDataset, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import (
    ToTensor,
    RandomCrop,
    CenterCrop,
)
from avalanche.benchmarks.utils import (
    make_classification_dataset,
    make_avalanche_dataset,
)
from avalanche.benchmarks.utils.flat_data import ConstantSequence
import random

import numpy as np

from avalanche.benchmarks.utils import DefaultTransformGroups
from avalanche.benchmarks.utils.data_attribute import TaskLabels, DataAttribute
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.flat_data import (
    _flatdata_depth,
    _flatdata_print,
)
from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
)
from tests.unit_tests_utils import (
    load_image_benchmark,
    load_tensor_benchmark,
    load_image_data,
)


def pil_images_equal(img_a, img_b):
    diff = ImageChops.difference(img_a, img_b)
    return not diff.getbbox()


def zero_if_label_2(img_tensor: Tensor, class_label):
    if int(class_label) == 2:
        torch.full(img_tensor.shape, 0.0, out=img_tensor)

    return img_tensor, class_label


class FrozenTransformGroupsCenterCrop:
    pass


class AvalancheDatasetTests(unittest.TestCase):

    def test_attribute_cat_sub(self):
        # Create a dataset of 100 data points described by 22
        # features + 1 class label
        x_data = torch.rand(100, 22)
        y_data = torch.randint(0, 5, (100,))
        torch_data = TensorDataset(x_data, y_data)

        tls = [0 for _ in range(100)]  # one task label for each sample
        sup_data = make_classification_dataset(torch_data, task_labels=tls)
        print(sup_data.targets.name, len(sup_data.targets._data))
        print(sup_data.targets_task_labels.name,
              len(sup_data.targets_task_labels._data))
        assert len(sup_data) == 100

        # after subsampling
        sub_data = sup_data.subset(range(10))
        print(sub_data.targets.name, len(sub_data.targets._data))
        print(sub_data.targets_task_labels.name,
              len(sub_data.targets_task_labels._data))
        assert len(sub_data) == 10

        # after concat
        cat_data = sup_data.concat(sup_data)
        print(cat_data.targets.name, len(cat_data.targets._data))
        print(cat_data.targets_task_labels.name,
              len(cat_data.targets_task_labels._data))
        assert len(cat_data) == 200

    def test_avldata_subset_size(self):
        data = [1, 2, 3, 4]
        attr = DataAttribute(data, "a")
        # avl data subset expects len(attribute) == len(dataset)
        AvalancheDataset([data], data_attributes=[attr], indices=[0, 1])

        # avl data should warn if len(attribute) != len(dataset)
        with self.assertRaises(ValueError):
            attr = DataAttribute(data[:2], "a")
            AvalancheDataset([data], data_attributes=[attr], indices=[0, 1])

    def test_avalanche_dataset_creation_without_list(self):
        dataset_mnist = load_image_benchmark()
        dataset = AvalancheDataset(dataset_mnist)
        self.assertIsInstance(dataset, AvalancheDataset)
        self.assertEqual(len(dataset_mnist), len(dataset))

        dataset = AvalancheDataset(dataset)
        self.assertIsInstance(dataset, AvalancheDataset)
        self.assertEqual(len(dataset_mnist), len(dataset))

    def test_disallowed_attribute_name(self):
        d_sz = 3
        xdata = torch.rand(d_sz, 2)
        dadata = torch.randint(0, 10, (d_sz,))
        da = DataAttribute(torch.zeros(d_sz), "collate_fn")
        with self.assertRaises(ValueError):
            d = make_avalanche_dataset(
                TensorDataset(xdata), data_attributes=[da]
            )

    def test_subset_subset_merge(self):
        d_sz, num_permutations = 3, 4

        # prepare dataset
        xdata = torch.rand(d_sz, 2)
        dadata = torch.randint(0, 10, (d_sz,))
        curr_dataset = make_avalanche_dataset(
            TensorDataset(xdata), data_attributes=[TaskLabels(dadata)]
        )

        # apply permutations iteratively
        ps = []
        true_indices = range(d_sz)
        for idx in range(num_permutations):
            print(idx, "depth: ", _flatdata_depth(curr_dataset))
            _flatdata_print(curr_dataset)

            idx_permuted = list(range(d_sz))
            random.shuffle(idx_permuted)
            true_indices = [true_indices[x] for x in idx_permuted]
            ps.append(list(idx_permuted))

            curr_dataset = curr_dataset.subset(indices=idx_permuted)
            self.assertEqual(len(curr_dataset), d_sz)

            print("Check data")
            x_curr = torch.stack(
                [curr_dataset[idx][0] for idx in range(d_sz)], dim=0
            )
            x_true = torch.stack([xdata[idx] for idx in true_indices], dim=0)
            self.assertTrue(torch.equal(x_curr, x_true))

            t_curr = torch.tensor(
                [curr_dataset.task_labels[idx] for idx in range(d_sz)]
            )
            t_true = torch.stack([dadata[idx] for idx in true_indices], dim=0)
            self.assertTrue(torch.equal(t_curr, t_true))

    def test_mnist_no_transforms(self):
        """check properties we need from the data for testing."""
        dataset = load_image_benchmark()
        x, y = dataset[0]
        self.assertIsInstance(x, Image)
        self.assertEqual([x.width, x.height], [28, 28])
        self.assertIsInstance(y, int)

    def test_avalanche_dataset_transform(self):
        dataset = load_image_benchmark()
        x, y = dataset[0]

        dataset = make_avalanche_dataset(
            dataset, transform_groups=DefaultTransformGroups((ToTensor(), None))
        )
        x2, y2 = dataset[0][0], dataset[0][1]
        # TODO: check __getitem__ task label

        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, int)
        # TODO: self.assertIsInstance(t2, int)
        # TODO: self.assertEqual(0, t2)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2)

    def test_avalanche_dataset_composition(self):
        dataset_mnist = load_image_benchmark()
        tgs = DefaultTransformGroups((RandomCrop(16), None))
        dataset = make_avalanche_dataset(dataset_mnist, transform_groups=tgs)

        x, y = dataset[0]
        self.assertIsInstance(x, Image)
        self.assertEqual([x.width, x.height], [16, 16])
        self.assertIsInstance(y, int)

        tgs = DefaultTransformGroups((ToTensor(), lambda target: -1))
        dataset = make_avalanche_dataset(dataset, transform_groups=tgs)

        x2, y2 = dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertEqual(x2.shape, (1, 16, 16))
        self.assertIsInstance(y2, int)
        self.assertEqual(y2, -1)

    def test_avalanche_dataset_add(self):
        dataset_mnist = load_image_benchmark()
        tgs = DefaultTransformGroups((CenterCrop(16), None))
        dataset_mnist = make_avalanche_dataset(
            dataset_mnist, transform_groups=tgs
        )

        taskl = DataAttribute(
            ConstantSequence(0, len(dataset_mnist)), "task_labels"
        )
        tgs = DefaultTransformGroups((ToTensor(), lambda target: -1))
        dataset1 = make_avalanche_dataset(
            dataset_mnist, data_attributes=[taskl], transform_groups=tgs
        )

        taskl = DataAttribute(
            ConstantSequence(2, len(dataset_mnist)), "task_labels"
        )
        tgs = DefaultTransformGroups((None, lambda target: -2))
        dataset2 = make_avalanche_dataset(
            dataset_mnist, data_attributes=[taskl], transform_groups=tgs
        )

        dataset3 = dataset1 + dataset2
        self.assertEqual(len(dataset_mnist) * 2, len(dataset3))

        x1, y1 = dataset1[0]
        x2, y2 = dataset2[0]

        x3, y3 = dataset3[0]
        x3_2, y3_2 = dataset3[len(dataset_mnist)]

        self.assertIsInstance(x1, Tensor)
        self.assertEqual(x1.shape, (1, 16, 16))
        self.assertEqual(-1, y1)

        self.assertIsInstance(x2, PIL.Image.Image)
        self.assertEqual(x2.size, (16, 16))
        self.assertEqual(-2, y2)

        self.assertEqual((y1), (y3))
        self.assertEqual(16 * 16, torch.sum(torch.eq(x1, x3)).item())

        self.assertEqual((y2), (y3_2))
        self.assertTrue(pil_images_equal(x2, x3_2))

    def test_avalanche_dataset_radd(self):
        dataset_mnist = load_image_benchmark()
        tgs = DefaultTransformGroups((CenterCrop(16), None))
        dataset_mnist = make_avalanche_dataset(
            dataset_mnist, transform_groups=tgs
        )

        tgs = DefaultTransformGroups((ToTensor(), lambda target: -1))
        dataset1 = make_avalanche_dataset(dataset_mnist, transform_groups=tgs)

        dataset2 = dataset_mnist + dataset1
        self.assertIsInstance(dataset2, AvalancheDataset)
        self.assertEqual(len(dataset_mnist) * 2, len(dataset2))

        dataset3 = dataset_mnist + dataset1 + dataset_mnist
        self.assertIsInstance(dataset3, AvalancheDataset)
        self.assertEqual(len(dataset_mnist) * 3, len(dataset3))

        dataset4 = dataset_mnist + dataset_mnist + dataset1
        self.assertIsInstance(dataset4, AvalancheDataset)
        self.assertEqual(len(dataset_mnist) * 3, len(dataset4))

    def test_dataset_add_monkey_patch_vanilla_behaviour(self):
        dataset_mnist = load_image_benchmark()
        dataset = dataset_mnist + dataset_mnist

        self.assertIsInstance(dataset, ConcatDataset)
        self.assertEqual(len(dataset_mnist) * 2, len(dataset))

    def test_avalanche_dataset_uniform_task_labels(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[0]

        dataset = make_classification_dataset(
            dataset_mnist, transform=ToTensor()
        )
        x2, y2, t2 = dataset[0]

        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, int)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2)

        # self.assertEqual(len(taskl.uniques), 1)
        # subset_task1 = taskl.val_to_idx[1]

        # TODO
        # self.assertIsInstance(subset_task1, AvalancheDataset)
        # self.assertEqual(len(dataset), len(subset_task1))

        # with self.assertRaises(KeyError):
        #     subset_task0 = dataset.task_set[0]

    def test_avalanche_dataset_tensor_task_labels(self):
        data = load_tensor_benchmark()
        taskl = torch.ones(32).int()  # Single task
        dataset = make_classification_dataset(data, task_labels=taskl)

        x2, y2, t2 = dataset[0]

        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, Tensor)
        self.assertIsInstance(t2, int)
        self.assertTrue(torch.equal(data[0][0], x2))
        self.assertTrue(torch.equal(data[0][1], y2))
        self.assertTrue(taskl[0] == t2)

        self.assertListEqual([1] * 32, list(dataset.targets_task_labels))

        # Regression test for #654
        self.assertEqual(1, len(dataset.task_set))

        subset_task1 = dataset.task_set[1]
        self.assertIsInstance(subset_task1, ClassificationDataset)
        self.assertEqual(len(dataset), len(subset_task1))

        with self.assertRaises(KeyError):
            subset_task0 = dataset.task_set[0]

        with self.assertRaises(KeyError):
            subset_task0 = dataset.task_set[2]

        # Check single instance types
        x2, y2, t2 = dataset[0]

        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, Tensor)
        self.assertIsInstance(t2, int)

    @unittest.skipIf(True, "Test needs refactoring")
    def test_avalanche_dataset_uniform_task_labels_simple_def(self):
        dataset_mnist = load_image_data()
        dataset = make_classification_dataset(
            dataset_mnist, transform=ToTensor(), task_labels=1
        )
        _, _, t2 = dataset[0]

        self.assertIsInstance(t2, int)
        self.assertEqual(1, t2)

        self.assertListEqual(
            [1] * len(dataset_mnist), list(dataset.targets_task_labels)
        )

        subset_task1 = dataset.task_set[1]
        self.assertIsInstance(subset_task1, make_classification_dataset)
        self.assertEqual(len(dataset), len(subset_task1))

        with self.assertRaises(KeyError):
            subset_task0 = dataset.task_set[0]

    def test_avalanche_dataset_mixed_task_labels(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[0]

        random_task_labels = [
            random.randint(0, 10) for _ in range(len(dataset_mnist))
        ]
        dataset = make_classification_dataset(
            dataset_mnist, transform=ToTensor(), task_labels=random_task_labels
        )
        x2, y2, t2 = dataset[0]

        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, int)
        self.assertIsInstance(t2, int)
        self.assertEqual(random_task_labels[0], t2)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2)

        self.assertListEqual(
            random_task_labels, list(dataset.targets_task_labels)
        )

        u_labels, counts = np.unique(random_task_labels, return_counts=True)
        for i, task_label in enumerate(u_labels.tolist()):
            subset_task = dataset.task_set[task_label]
            self.assertIsInstance(subset_task, ClassificationDataset)
            self.assertEqual(int(counts[i]), len(subset_task))

            unique_task_labels = list(subset_task.targets_task_labels)
            self.assertListEqual(
                [task_label] * int(counts[i]), unique_task_labels
            )

        with self.assertRaises(KeyError):
            subset_task11 = dataset.task_set[11]


if __name__ == "__main__":
    unittest.main()
