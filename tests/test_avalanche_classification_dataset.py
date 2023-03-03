import unittest

from os.path import expanduser

import avalanche
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.models import SimpleMLP
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.training.supervised import Naive
from avalanche.benchmarks.generators import dataset_benchmark
import PIL
import torch
from PIL import ImageChops
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import TensorDataset, Subset, ConcatDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import (
    ToTensor,
    RandomCrop,
    ToPILImage,
    Compose,
    Lambda,
)
from typing import List

from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    create_generic_benchmark_from_tensor_lists,
)
from avalanche.benchmarks.utils import (
    make_classification_dataset,
    classification_subset,
    concat_classification_datasets,
    make_tensor_classification_dataset,
)
from avalanche.benchmarks.utils.utils import (
    concat_datasets_sequentially,
    concat_datasets,
)
from avalanche.training.utils import load_all_dataset
import random

import numpy as np

from avalanche.benchmarks.utils.flat_data import (
    _flatdata_depth,
    _flatdata_print,
)
from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
)
from tests.unit_tests_utils import load_image_data


def pil_images_equal(img_a, img_b):
    diff = ImageChops.difference(img_a, img_b)

    return not diff.getbbox()


def zero_if_label_2(img_tensor: Tensor, class_label):
    if int(class_label) == 2:
        torch.full(img_tensor.shape, 0.0, out=img_tensor)

    return img_tensor, class_label


def get_mbatch(data, batch_size=5):
    dl = DataLoader(
        data, shuffle=False, batch_size=batch_size, collate_fn=data.collate_fn
    )
    return next(iter(dl))


class AvalancheDatasetTests(unittest.TestCase):
    def test_avalanche_dataset_multi_param_transform(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )

        ref_instance2_idx = None
        for instance_idx, (_, instance_y) in enumerate(dataset_mnist):
            if instance_y == 2:
                ref_instance2_idx = instance_idx
                break
        self.assertIsNotNone(ref_instance2_idx)

        ref_instance_idx = None
        for instance_idx, (_, instance_y) in enumerate(dataset_mnist):
            if instance_y != 2:
                ref_instance_idx = instance_idx
                break
        self.assertIsNotNone(ref_instance_idx)

        with self.assertWarns(
            avalanche.benchmarks.utils.ComposeMaxParamsWarning
        ):
            dataset_transform = avalanche.benchmarks.utils.MultiParamCompose(
                [ToTensor(), zero_if_label_2]
            )

        self.assertEqual(1, dataset_transform.min_params)
        self.assertEqual(2, dataset_transform.max_params)

        tgs = {"train": dataset_transform, "eval": dataset_transform}
        x, y = dataset_mnist[ref_instance_idx]
        dataset = make_classification_dataset(
            dataset_mnist, transform_groups=tgs
        )
        x2, y2, t2 = dataset[ref_instance_idx]

        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, int)
        self.assertIsInstance(t2, int)
        self.assertEqual(0, t2)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2)

        # Check that the multi-param transform was correctly called
        x3, y3, _ = dataset[ref_instance2_idx]

        self.assertEqual(2, y3)
        self.assertIsInstance(x3, Tensor)
        self.assertEqual(0.0, torch.min(x3))
        self.assertEqual(0.0, torch.max(x3))

    def test_avalanche_dataset_tensor_task_labels(self):
        x = torch.rand(32, 10)
        y = torch.rand(32, 10)
        t = torch.ones(32)  # Single task
        dataset = make_classification_dataset(
            TensorDataset(x, y), targets=1, task_labels=t
        )

        x2, y2, t2 = get_mbatch(dataset, batch_size=32)

        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, Tensor)
        self.assertIsInstance(t2, Tensor)
        self.assertTrue(torch.equal(x, x2))
        self.assertTrue(torch.equal(y, y2))
        self.assertTrue(torch.equal(t.to(int), t2))

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

    def test_avalanche_dataset_uniform_task_labels_simple_def(self):
        dataset_mnist = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/", download=True
        )
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
        self.assertIsInstance(subset_task1, ClassificationDataset)
        self.assertEqual(len(dataset), len(subset_task1))

        with self.assertRaises(KeyError):
            subset_task0 = dataset.task_set[0]

    def test_avalanche_dataset_mixed_task_labels(self):
        dataset_mnist = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/", download=True
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

    def test_avalanche_tensor_dataset_task_labels_train(self):
        tr_ds = [
            make_tensor_classification_dataset(
                torch.randn(10, 4),
                torch.randint(0, 3, (10,)),
                task_labels=torch.randint(0, 5, (10,)).tolist(),
            )
            for i in range(3)
        ]
        ts_ds = [
            make_tensor_classification_dataset(
                torch.randn(10, 4),
                torch.randint(0, 3, (10,)),
                task_labels=torch.randint(0, 5, (10,)).tolist(),
            )
            for i in range(3)
        ]
        benchmark = dataset_benchmark(train_datasets=tr_ds, test_datasets=ts_ds)
        model = SimpleMLP(input_size=4, num_classes=3)
        cl_strategy = Naive(
            model,
            SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(),
            train_mb_size=5,
            train_epochs=1,
            eval_mb_size=5,
            device="cpu",
            evaluator=None,
        )
        exp = []
        for i, experience in enumerate(benchmark.train_stream):
            exp.append(i)
            cl_strategy.train(experience)
        self.assertEqual(len(exp), 3)

    def test_avalanche_dataset_task_labels_inheritance(self):
        dataset_mnist = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/", download=True
        )
        random_task_labels = [
            random.randint(0, 10) for _ in range(len(dataset_mnist))
        ]
        dataset_orig = make_classification_dataset(
            dataset_mnist, transform=ToTensor(), task_labels=random_task_labels
        )

        dataset_child = make_classification_dataset(dataset_orig)
        x2, y2, t2 = dataset_orig[0]
        x3, y3, t3 = dataset_child[0]

        self.assertIsInstance(t2, int)
        self.assertEqual(random_task_labels[0], t2)

        self.assertIsInstance(t3, int)
        self.assertEqual(random_task_labels[0], t3)

        self.assertListEqual(
            random_task_labels, list(dataset_orig.targets_task_labels)
        )

        self.assertListEqual(
            random_task_labels, list(dataset_child.targets_task_labels)
        )

    def test_avalanche_dataset_tensor_dataset_input(self):
        train_x = torch.rand(500, 3, 28, 28)
        train_y = torch.zeros(500)
        test_x = torch.rand(200, 3, 28, 28)
        test_y = torch.ones(200)

        train = TensorDataset(train_x, train_y)
        test = TensorDataset(test_x, test_y)
        train_dataset = make_classification_dataset(train)
        test_dataset = make_classification_dataset(test)

        self.assertEqual(500, len(train_dataset))
        self.assertEqual(200, len(test_dataset))

        x, y, t = train_dataset[0]
        self.assertIsInstance(x, Tensor)
        self.assertEqual(0, y)
        self.assertEqual(0, t)

        x2, y2, t2 = test_dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertEqual(1, y2)
        self.assertEqual(0, t2)

    def test_avalanche_dataset_multiple_outputs_and_float_y(self):
        train_x = torch.rand(500, 3, 28, 28)
        train_y = torch.zeros(500)
        train_z = torch.ones(500)
        test_x = torch.rand(200, 3, 28, 28)
        test_y = torch.ones(200)
        test_z = torch.full((200,), 5)

        train = TensorDataset(train_x, train_y, train_z)
        test = TensorDataset(test_x, test_y, test_z)
        train_dataset = make_classification_dataset(train)
        test_dataset = make_classification_dataset(test)

        self.assertEqual(500, len(train_dataset))
        self.assertEqual(200, len(test_dataset))

        x, y, z, t = train_dataset[0]
        self.assertIsInstance(x, Tensor)
        self.assertEqual(0, y)
        self.assertEqual(1, z)
        self.assertEqual(0, t)

        x2, y2, z2, t2 = test_dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertEqual(1, y2)
        self.assertEqual(5, z2)
        self.assertEqual(0, t2)

    def test_avalanche_dataset_from_pytorch_subset(self):
        tensor_x = torch.rand(500, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (500,))

        whole_dataset = TensorDataset(tensor_x, tensor_y)

        train = Subset(whole_dataset, indices=list(range(400)))
        test = Subset(whole_dataset, indices=list(range(400, 500)))

        train_dataset = make_classification_dataset(train)
        test_dataset = make_classification_dataset(test)

        self.assertEqual(400, len(train_dataset))
        self.assertEqual(100, len(test_dataset))

        x, y, t = train_dataset[0]
        self.assertIsInstance(x, Tensor)
        self.assertTrue(torch.equal(tensor_x[0], x))
        self.assertTrue(torch.equal(tensor_y[0], y))
        self.assertEqual(0, t)

        self.assertTrue(
            torch.equal(torch.as_tensor(train_dataset.targets), tensor_y[:400])
        )

        x2, y2, t2 = test_dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertTrue(torch.equal(tensor_x[400], x2))
        self.assertTrue(torch.equal(tensor_y[400], y2))
        self.assertEqual(0, t2)

        self.assertTrue(
            torch.equal(torch.as_tensor(test_dataset.targets), tensor_y[400:])
        )

    def test_avalanche_dataset_from_pytorch_concat_dataset(self):
        tensor_x = torch.rand(500, 3, 28, 28)
        tensor_x2 = torch.rand(300, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (500,))
        tensor_y2 = torch.randint(0, 100, (300,))

        dataset1 = TensorDataset(tensor_x, tensor_y)
        dataset2 = TensorDataset(tensor_x2, tensor_y2)

        concat_dataset = ConcatDataset((dataset1, dataset2))

        av_dataset = make_classification_dataset(concat_dataset)

        self.assertEqual(500, len(dataset1))
        self.assertEqual(300, len(dataset2))

        x, y, t = av_dataset[0]
        x2, y2, t2 = av_dataset[500]
        self.assertIsInstance(x, Tensor)
        self.assertTrue(torch.equal(tensor_x[0], x))
        self.assertTrue(torch.equal(tensor_y[0], y))
        self.assertEqual(0, t)

        self.assertIsInstance(x2, Tensor)
        self.assertTrue(torch.equal(tensor_x2[0], x2))
        self.assertTrue(torch.equal(tensor_y2[0], y2))
        self.assertEqual(0, t2)

        self.assertTrue(
            torch.equal(
                torch.as_tensor(av_dataset.targets),
                torch.cat((tensor_y, tensor_y2)),
            )
        )

    def test_avalanche_dataset_from_chained_pytorch_concat_dataset(self):
        tensor_x = torch.rand(500, 3, 28, 28)
        tensor_x2 = torch.rand(300, 3, 28, 28)
        tensor_x3 = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (500,))
        tensor_y2 = torch.randint(0, 100, (300,))
        tensor_y3 = torch.randint(0, 100, (200,))

        dataset1 = TensorDataset(tensor_x, tensor_y)
        dataset2 = TensorDataset(tensor_x2, tensor_y2)
        dataset3 = TensorDataset(tensor_x3, tensor_y3)

        concat_dataset = ConcatDataset((dataset1, dataset2))
        concat_dataset2 = ConcatDataset((concat_dataset, dataset3))

        av_dataset = make_classification_dataset(concat_dataset2)

        self.assertEqual(500, len(dataset1))
        self.assertEqual(300, len(dataset2))

        x, y, t = av_dataset[0]
        x2, y2, t2 = av_dataset[500]
        x3, y3, t3 = av_dataset[800]
        self.assertIsInstance(x, Tensor)
        self.assertTrue(torch.equal(tensor_x[0], x))
        self.assertTrue(torch.equal(tensor_y[0], y))
        self.assertEqual(0, t)

        self.assertIsInstance(x2, Tensor)
        self.assertTrue(torch.equal(tensor_x2[0], x2))
        self.assertTrue(torch.equal(tensor_y2[0], y2))
        self.assertEqual(0, t2)

        self.assertIsInstance(x3, Tensor)
        self.assertTrue(torch.equal(tensor_x3[0], x3))
        self.assertTrue(torch.equal(tensor_y3[0], y3))
        self.assertEqual(0, t3)

        self.assertTrue(
            torch.equal(
                torch.as_tensor(av_dataset.targets),
                torch.cat((tensor_y, tensor_y2, tensor_y3)),
            )
        )

    def test_avalanche_dataset_from_chained_pytorch_subsets(self):
        tensor_x = torch.rand(500, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (500,))

        whole_dataset = TensorDataset(tensor_x, tensor_y)

        subset1 = Subset(whole_dataset, indices=list(range(400, 500)))
        subset2 = Subset(subset1, indices=[5, 7, 0])

        dataset = make_classification_dataset(subset2)

        self.assertEqual(3, len(dataset))

        x, y, t = dataset[0]
        self.assertIsInstance(x, Tensor)
        self.assertTrue(torch.equal(tensor_x[405], x))
        self.assertTrue(torch.equal(tensor_y[405], y))
        self.assertEqual(0, t)

        self.assertTrue(
            torch.equal(
                torch.as_tensor(dataset.targets),
                torch.as_tensor([tensor_y[405], tensor_y[407], tensor_y[400]]),
            )
        )

    def test_avalanche_dataset_from_chained_pytorch_concat_subset_dataset(self):
        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_x2 = torch.rand(100, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (200,))
        tensor_y2 = torch.randint(0, 100, (100,))

        dataset1 = TensorDataset(tensor_x, tensor_y)
        dataset2 = TensorDataset(tensor_x2, tensor_y2)

        indices = [random.randint(0, 299) for _ in range(1000)]

        concat_dataset = ConcatDataset((dataset1, dataset2))
        subset = Subset(concat_dataset, indices)

        av_dataset = make_classification_dataset(subset)

        self.assertEqual(200, len(dataset1))
        self.assertEqual(100, len(dataset2))
        self.assertEqual(1000, len(av_dataset))

        for idx in range(1000):
            orig_idx = indices[idx]
            if orig_idx < 200:
                expected_x, expected_y = dataset1[orig_idx]
            else:
                expected_x, expected_y = dataset2[orig_idx - 200]

            x, y, t = av_dataset[idx]
            self.assertIsInstance(x, Tensor)
            self.assertTrue(torch.equal(expected_x, x))
            self.assertTrue(torch.equal(expected_y, y))
            self.assertEqual(0, t)
            self.assertEqual(int(expected_y), int(av_dataset.targets[idx]))

    def test_avalanche_dataset_collate_fn(self):
        tensor_x = torch.rand(500, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (500,))
        tensor_z = torch.randint(0, 100, (500,))

        def my_collate_fn(patterns):
            x_values = torch.stack([pat[0] for pat in patterns], 0)
            y_values = torch.tensor([pat[1] for pat in patterns]) + 1
            z_values = torch.tensor([-1 for _ in patterns])
            t_values = torch.tensor([pat[3] for pat in patterns])
            return x_values, y_values, z_values, t_values

        whole_dataset = TensorDataset(tensor_x, tensor_y, tensor_z)
        dataset = make_classification_dataset(
            whole_dataset, collate_fn=my_collate_fn
        )

        x, y, z, t = dataset[0]
        self.assertIsInstance(x, Tensor)
        self.assertTrue(torch.equal(tensor_x[0], x))
        self.assertTrue(torch.equal(tensor_y[0], y))
        self.assertEqual(0, t)

        x2, y2, z2, t2 = get_mbatch(dataset)
        self.assertIsInstance(x2, Tensor)
        self.assertTrue(torch.equal(tensor_x[0:5], x2))
        self.assertTrue(torch.equal(tensor_y[0:5] + 1, y2))
        self.assertTrue(torch.equal(torch.full((5,), -1, dtype=torch.long), z2))
        self.assertTrue(torch.equal(torch.zeros(5, dtype=torch.long), t2))

        inherited = make_classification_dataset(dataset)

        x3, y3, z3, t3 = get_mbatch(inherited)
        self.assertIsInstance(x3, Tensor)
        self.assertTrue(torch.equal(tensor_x[0:5], x3))
        self.assertTrue(torch.equal(tensor_y[0:5] + 1, y3))
        self.assertTrue(torch.equal(torch.full((5,), -1, dtype=torch.long), z3))
        self.assertTrue(torch.equal(torch.zeros(5, dtype=torch.long), t3))

    def test_avalanche_dataset_collate_fn_inheritance(self):
        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (200,))
        tensor_z = torch.randint(0, 100, (200,))

        def my_collate_fn(patterns):
            x_values = torch.stack([pat[0] for pat in patterns], 0)
            y_values = torch.tensor([pat[1] for pat in patterns]) + 1
            z_values = torch.tensor([-1 for _ in patterns])
            t_values = torch.tensor([pat[3] for pat in patterns])
            return x_values, y_values, z_values, t_values

        def my_collate_fn2(patterns):
            x_values = torch.stack([pat[0] for pat in patterns], 0)
            y_values = torch.tensor([pat[1] for pat in patterns]) + 2
            z_values = torch.tensor([-2 for _ in patterns])
            t_values = torch.tensor([pat[3] for pat in patterns])
            return x_values, y_values, z_values, t_values

        whole_dataset = TensorDataset(tensor_x, tensor_y, tensor_z)
        dataset = make_classification_dataset(
            whole_dataset, collate_fn=my_collate_fn
        )
        inherited = make_classification_dataset(
            dataset, collate_fn=my_collate_fn2
        )  # Ok

        x, y, z, t = get_mbatch(inherited)
        self.assertIsInstance(x, Tensor)
        self.assertTrue(torch.equal(tensor_x[0:5], x))
        self.assertTrue(torch.equal(tensor_y[0:5] + 2, y))
        self.assertTrue(torch.equal(torch.full((5,), -2, dtype=torch.long), z))
        self.assertTrue(torch.equal(torch.zeros(5, dtype=torch.long), t))

        classification_dataset = make_classification_dataset(whole_dataset)

        ok_inherited_classification = make_classification_dataset(
            classification_dataset
        )

    def test_avalanche_concat_dataset_collate_fn_inheritance(self):
        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (200,))
        tensor_z = torch.randint(0, 100, (200,))

        tensor_x2 = torch.rand(200, 3, 28, 28)
        tensor_y2 = torch.randint(0, 100, (200,))
        tensor_z2 = torch.randint(0, 100, (200,))

        def my_collate_fn(patterns):
            x_values = torch.stack([pat[0] for pat in patterns], 0)
            y_values = torch.tensor([pat[1] for pat in patterns]) + 1
            z_values = torch.tensor([-1 for _ in patterns])
            t_values = torch.tensor([pat[3] for pat in patterns])
            return x_values, y_values, z_values, t_values

        def my_collate_fn2(patterns):
            x_values = torch.stack([pat[0] for pat in patterns], 0)
            y_values = torch.tensor([pat[1] for pat in patterns]) + 2
            z_values = torch.tensor([-2 for _ in patterns])
            t_values = torch.tensor([pat[3] for pat in patterns])
            return x_values, y_values, z_values, t_values

        dataset1 = TensorDataset(tensor_x, tensor_y, tensor_z)
        dataset2 = make_tensor_classification_dataset(
            tensor_x2, tensor_y2, tensor_z2, collate_fn=my_collate_fn
        )
        concat = concat_classification_datasets(
            [dataset1, dataset2], collate_fn=my_collate_fn2
        )  # Ok

        x, y, z, t = get_mbatch(dataset2)
        self.assertIsInstance(x, Tensor)
        self.assertTrue(torch.equal(tensor_x2[0:5], x))
        self.assertTrue(torch.equal(tensor_y2[0:5] + 1, y))
        self.assertTrue(torch.equal(torch.full((5,), -1, dtype=torch.long), z))
        self.assertTrue(torch.equal(torch.zeros(5, dtype=torch.long), t))

        x2, y2, z2, t2 = get_mbatch(concat)
        self.assertIsInstance(x2, Tensor)
        self.assertTrue(torch.equal(tensor_x[0:5], x2))
        self.assertTrue(torch.equal(tensor_y[0:5] + 2, y2))
        self.assertTrue(torch.equal(torch.full((5,), -2, dtype=torch.long), z2))
        self.assertTrue(torch.equal(torch.zeros(5, dtype=torch.long), t2))

        dataset1_classification = make_tensor_classification_dataset(
            tensor_x, tensor_y, tensor_z
        )

        dataset2_segmentation = make_classification_dataset(dataset2)

        # with self.assertRaises(ValueError):
        #     bad_concat_types = dataset1_classification + dataset2_segmentation
        #
        # with self.assertRaises(ValueError):
        #     bad_concat_collate = AvalancheConcatDataset(
        #         [dataset1, dataset2_segmentation], collate_fn=my_collate_fn
        #     )

        ok_concat_classification = dataset1_classification + dataset2

        ok_concat_classification2 = dataset2 + dataset1_classification

    def test_avalanche_concat_dataset_recursion(self):
        def gen_random_tensors(n):
            return (
                torch.rand(n, 3, 28, 28),
                torch.randint(0, 100, (n,)),
                torch.randint(0, 100, (n,)),
            )

        tensor_x, tensor_y, tensor_z = gen_random_tensors(200)
        tensor_x2, tensor_y2, tensor_z2 = gen_random_tensors(200)
        tensor_x3, tensor_y3, tensor_z3 = gen_random_tensors(200)
        tensor_x4, tensor_y4, tensor_z4 = gen_random_tensors(200)
        tensor_x5, tensor_y5, tensor_z5 = gen_random_tensors(200)
        tensor_x6, tensor_y6, tensor_z6 = gen_random_tensors(200)
        tensor_x7, tensor_y7, tensor_z7 = gen_random_tensors(200)

        dataset1 = TensorDataset(tensor_x, tensor_y, tensor_z)
        dataset2 = make_classification_dataset(
            TensorDataset(tensor_x2, tensor_y2, tensor_z2),
            targets=tensor_y2,
            task_labels=1,
        )
        dataset3 = make_classification_dataset(
            TensorDataset(tensor_x3, tensor_y3, tensor_z3),
            targets=tensor_y3,
            task_labels=2,
        )

        dataset4 = make_classification_dataset(
            TensorDataset(tensor_x4, tensor_y4, tensor_z4),
            targets=tensor_y4,
            task_labels=3,
        )
        dataset5 = make_classification_dataset(
            TensorDataset(tensor_x5, tensor_y5, tensor_z5),
            targets=tensor_y5,
            task_labels=4,
        )
        dataset6 = make_classification_dataset(
            TensorDataset(tensor_x6, tensor_y6, tensor_z6), targets=tensor_y6
        )
        dataset7 = make_classification_dataset(
            TensorDataset(tensor_x7, tensor_y7, tensor_z7), targets=tensor_y7
        )

        # This will test recursion on both PyTorch ConcatDataset and
        # AvalancheConcatDataset
        concat = concat_classification_datasets([dataset1, dataset2])

        # Beware of the explicit task_labels=5 that *must* override the
        # task labels set in dataset4 and dataset5

        def transform_target_to_constant(ignored_target_value):
            return 101

        def transform_target_to_constant2(ignored_target_value):
            return 102

        concat2 = concat_classification_datasets(
            [dataset4, dataset5],
            task_labels=5,
            target_transform=transform_target_to_constant,
        )

        concat3 = concat_classification_datasets(
            [dataset6, dataset7], target_transform=transform_target_to_constant2
        ).freeze_transforms()
        concat_uut = concat_datasets([concat, dataset3, concat2, concat3])

        self.assertEqual(400, len(concat))
        self.assertEqual(400, len(concat2))
        self.assertEqual(400, len(concat3))
        self.assertEqual(1400, len(concat_uut))

        x, y, z, t = concat_uut[0]
        x2, y2, z2, t2 = concat_uut[200]
        x3, y3, z3, t3 = concat_uut[400]
        x4, y4, z4, t4 = concat_uut[600]
        x5, y5, z5, t5 = concat_uut[800]
        x6, y6, z6, t6 = concat_uut[1000]
        x7, y7, z7, t7 = concat_uut[1200]

        self.assertTrue(torch.equal(x, tensor_x[0]))
        self.assertTrue(torch.equal(y, tensor_y[0]))
        self.assertTrue(torch.equal(z, tensor_z[0]))
        self.assertEqual(0, t)

        self.assertTrue(torch.equal(x2, tensor_x2[0]))
        self.assertTrue(torch.equal(y2, tensor_y2[0]))
        self.assertTrue(torch.equal(z2, tensor_z2[0]))
        self.assertEqual(1, t2)

        self.assertTrue(torch.equal(x3, tensor_x3[0]))
        self.assertTrue(torch.equal(y3, tensor_y3[0]))
        self.assertTrue(torch.equal(z3, tensor_z3[0]))
        self.assertEqual(2, t3)

        self.assertTrue(torch.equal(x4, tensor_x4[0]))
        self.assertEqual(101, y4)
        self.assertTrue(torch.equal(z4, tensor_z4[0]))
        self.assertEqual(5, t4)

        self.assertTrue(torch.equal(x5, tensor_x5[0]))
        self.assertEqual(101, y5)
        self.assertTrue(torch.equal(z5, tensor_z5[0]))
        self.assertEqual(5, t5)

        self.assertTrue(torch.equal(x6, tensor_x6[0]))
        self.assertEqual(102, y6)
        self.assertTrue(torch.equal(z6, tensor_z6[0]))
        self.assertEqual(0, t6)

        self.assertTrue(torch.equal(x7, tensor_x7[0]))
        self.assertEqual(102, y7)
        self.assertTrue(torch.equal(z7, tensor_z7[0]))
        self.assertEqual(0, t7)

    def test_avalanche_pytorch_subset_recursion(self):
        dataset_mnist = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/", download=True
        )
        x, y = dataset_mnist[3000]
        x2, y2 = dataset_mnist[1010]

        subset = Subset(dataset_mnist, indices=[3000, 8, 4, 1010, 12])

        dataset = classification_subset(subset, indices=[0, 3])

        self.assertEqual(5, len(subset))
        self.assertEqual(2, len(dataset))

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]
        self.assertTrue(pil_images_equal(x, x3))
        self.assertEqual(y, y3)
        self.assertEqual(0, t3)
        self.assertTrue(pil_images_equal(x2, x4))
        self.assertEqual(y2, y4)
        self.assertEqual(0, t4)
        self.assertFalse(pil_images_equal(x, x4))
        self.assertFalse(pil_images_equal(x2, x3))

        def transform_target_to_constant(ignored_target_value):
            return 101

        subset = Subset(dataset_mnist, indices=[3000, 8, 4, 1010, 12])

        dataset = classification_subset(
            subset,
            indices=[0, 3],
            target_transform=transform_target_to_constant,
            task_labels=5,
        )

        self.assertEqual(5, len(subset))
        self.assertEqual(2, len(dataset))

        x5, y5, t5 = dataset[0]
        x6, y6, t6 = dataset[1]
        self.assertTrue(pil_images_equal(x, x5))
        self.assertEqual(101, y5)
        self.assertEqual(5, t5)
        self.assertTrue(pil_images_equal(x2, x6))
        self.assertEqual(101, y6)
        self.assertEqual(5, t6)
        self.assertFalse(pil_images_equal(x, x6))
        self.assertFalse(pil_images_equal(x2, x5))

    def test_avalanche_pytorch_subset_recursion_no_indices(self):
        dataset_mnist = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/", download=True
        )
        x, y = dataset_mnist[3000]
        x2, y2 = dataset_mnist[8]

        subset = Subset(dataset_mnist, indices=[3000, 8, 4, 1010, 12])

        dataset = classification_subset(subset)

        self.assertEqual(5, len(subset))
        self.assertEqual(5, len(dataset))

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]
        self.assertTrue(pil_images_equal(x, x3))
        self.assertEqual(y, y3)
        self.assertTrue(pil_images_equal(x2, x4))
        self.assertEqual(y2, y4)
        self.assertFalse(pil_images_equal(x, x4))
        self.assertFalse(pil_images_equal(x2, x3))

    def test_avalanche_avalanche_subset_recursion_no_indices_transform(self):
        dataset_mnist = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/", download=True
        )
        x, y = dataset_mnist[3000]
        x2, y2 = dataset_mnist[8]

        def transform_target_to_constant(ignored_target_value):
            return 101

        def transform_target_plus_one(target_value):
            return target_value + 1

        subset = classification_subset(
            dataset_mnist,
            indices=[3000, 8, 4, 1010, 12],
            transform=ToTensor(),
            target_transform=transform_target_to_constant,
        )

        dataset = classification_subset(
            subset, target_transform=transform_target_plus_one
        )

        self.assertEqual(5, len(subset))
        self.assertEqual(5, len(dataset))

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]
        self.assertIsInstance(x3, Tensor)
        self.assertIsInstance(x4, Tensor)
        self.assertTrue(torch.equal(ToTensor()(x), x3))
        self.assertEqual(102, y3)
        self.assertTrue(torch.equal(ToTensor()(x2), x4))
        self.assertEqual(102, y4)
        self.assertFalse(torch.equal(ToTensor()(x), x4))
        self.assertFalse(torch.equal(ToTensor()(x2), x3))

    def test_avalanche_avalanche_subset_recursion_transform(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[3000]
        x2, y2 = dataset_mnist[1010]

        def transform_target_to_constant(ignored_target_value):
            return 101

        def transform_target_plus_one(target_value):
            return target_value + 2

        subset = classification_subset(
            dataset_mnist,
            indices=[3000, 8, 4, 1010, 12],
            target_transform=transform_target_to_constant,
        )

        dataset = classification_subset(
            subset,
            indices=[0, 3, 1],
            target_transform=transform_target_plus_one,
        )

        self.assertEqual(5, len(subset))
        self.assertEqual(3, len(dataset))

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]

        self.assertTrue(pil_images_equal(x, x3))
        self.assertEqual(103, y3)
        self.assertTrue(pil_images_equal(x2, x4))
        self.assertEqual(103, y4)
        self.assertFalse(pil_images_equal(x, x4))
        self.assertFalse(pil_images_equal(x2, x3))

    def test_avalanche_avalanche_subset_recursion_frozen_transform(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[3000]
        x2, y2 = dataset_mnist[1010]

        def transform_target_to_constant(ignored_target_value):
            return 101

        def transform_target_plus_two(target_value):
            return target_value + 2

        subset = classification_subset(
            dataset_mnist,
            indices=[3000, 8, 4, 1010, 12],
            target_transform=transform_target_to_constant,
        )
        subset = subset.freeze_transforms()

        dataset = classification_subset(
            subset,
            indices=[0, 3, 1],
            target_transform=transform_target_plus_two,
        )

        self.assertEqual(5, len(subset))
        self.assertEqual(3, len(dataset))

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]

        self.assertTrue(pil_images_equal(x, x3))
        self.assertEqual(103, y3)
        self.assertTrue(pil_images_equal(x2, x4))
        self.assertEqual(103, y4)
        self.assertFalse(pil_images_equal(x, x4))
        self.assertFalse(pil_images_equal(x2, x3))

        dataset = classification_subset(
            subset,
            indices=[0, 3, 1],
            target_transform=transform_target_plus_two,
        )
        dataset = dataset.replace_current_transform_group(None)

        x5, y5, t5 = dataset[0]
        x6, y6, t6 = dataset[1]

        self.assertTrue(pil_images_equal(x, x5))
        self.assertEqual(101, y5)
        self.assertTrue(pil_images_equal(x2, x6))
        self.assertEqual(101, y6)
        self.assertFalse(pil_images_equal(x, x6))
        self.assertFalse(pil_images_equal(x2, x5))

    def test_avalanche_avalanche_subset_recursion_sub_class_mapping(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[3000]
        x2, y2 = dataset_mnist[1010]

        class_mapping = list(range(10))
        random.shuffle(class_mapping)

        subset = classification_subset(
            dataset_mnist,
            indices=[3000, 8, 4, 1010, 12],
            class_mapping=class_mapping,
        )

        dataset = classification_subset(subset, indices=[0, 3, 1])

        self.assertEqual(5, len(subset))
        self.assertEqual(3, len(dataset))

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]

        self.assertTrue(pil_images_equal(x, x3))
        expected_y3 = class_mapping[y]
        self.assertEqual(expected_y3, y3)
        self.assertTrue(pil_images_equal(x2, x4))
        expected_y4 = class_mapping[y2]
        self.assertEqual(expected_y4, y4)
        self.assertFalse(pil_images_equal(x, x4))
        self.assertFalse(pil_images_equal(x2, x3))

    def test_avalanche_avalanche_subset_recursion_up_class_mapping(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[3000]
        x2, y2 = dataset_mnist[1010]

        class_mapping = list(range(10))
        random.shuffle(class_mapping)

        subset = classification_subset(
            dataset_mnist, indices=[3000, 8, 4, 1010, 12]
        )

        dataset = classification_subset(
            subset, indices=[0, 3, 1], class_mapping=class_mapping
        )

        self.assertEqual(5, len(subset))
        self.assertEqual(3, len(dataset))

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]

        self.assertTrue(pil_images_equal(x, x3))
        expected_y3 = class_mapping[y]
        self.assertEqual(expected_y3, y3)
        self.assertTrue(pil_images_equal(x2, x4))
        expected_y4 = class_mapping[y2]
        self.assertEqual(expected_y4, y4)
        self.assertFalse(pil_images_equal(x, x4))
        self.assertFalse(pil_images_equal(x2, x3))

    def test_avalanche_avalanche_subset_recursion_mix_class_mapping(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[3000]
        x2, y2 = dataset_mnist[1010]

        class_mapping = list(range(10))
        class_mapping2 = list(range(10))
        random.shuffle(class_mapping)
        random.shuffle(class_mapping2)

        subset = classification_subset(
            dataset_mnist,
            indices=[3000, 8, 4, 1010, 12],
            class_mapping=class_mapping,
        )

        dataset = classification_subset(
            subset, indices=[0, 3, 1], class_mapping=class_mapping2
        )

        self.assertEqual(5, len(subset))
        self.assertEqual(3, len(dataset))

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]

        self.assertTrue(pil_images_equal(x, x3))
        expected_y3 = class_mapping2[class_mapping[y]]
        self.assertEqual(expected_y3, y3)
        self.assertTrue(pil_images_equal(x2, x4))
        expected_y4 = class_mapping2[class_mapping[y2]]
        self.assertEqual(expected_y4, y4)
        self.assertFalse(pil_images_equal(x, x4))
        self.assertFalse(pil_images_equal(x2, x3))

    def test_avalanche_avalanche_subset_concat_stack_overflow(self):
        d_sz = 4
        tensor_x = torch.rand(d_sz, 2)
        tensor_y = torch.randint(0, 7, (d_sz,))
        tensor_t = torch.randint(0, 7, (d_sz,))
        dataset = make_classification_dataset(
            TensorDataset(tensor_x, tensor_y),
            targets=tensor_y,
            task_labels=tensor_t,
        )
        dataset_hierarchy_depth = 500

        # prepare random permutations for each step
        random_permutations: List[List[int]] = []
        for _ in range(dataset_hierarchy_depth):
            idx_permuted = list(range(d_sz))
            random.shuffle(idx_permuted)
            random_permutations.append(idx_permuted)

        # compute expected indices after all permutations
        current_indices = range(d_sz)
        true_indices: List[List[int]] = []
        true_indices.append(list(current_indices))
        for idx in range(dataset_hierarchy_depth):
            current_indices = [
                current_indices[x] for x in random_permutations[idx]
            ]
            true_indices.append(current_indices)
        true_indices = list(reversed(true_indices))

        # apply permutations and concatenations iteratively
        curr_dataset = dataset
        for idx in range(dataset_hierarchy_depth):
            # print(idx)
            # print(idx, "depth: ", _flatdata_depth(curr_dataset))
            # _flatdata_print(curr_dataset)
            intermediate_idx_test = (dataset_hierarchy_depth - 1) - idx
            subset = classification_subset(
                curr_dataset, indices=random_permutations[idx]
            )
            curr_dataset = subset.concat(curr_dataset)

            # Regression test for #616 (second bug)
            # https://github.com/ContinualAI/avalanche/issues/616#issuecomment-848852287
            all_targets = torch.tensor(curr_dataset.targets)
            self.assertTrue(torch.equal(tensor_y, all_targets[-d_sz:]))

            curr_targets = torch.tensor(list(curr_dataset.targets))
            for idx_internal in range(idx + 1):
                # curr_dataset is the concat of idx+1 datasets.
                # Check all of them are permuted correctly
                leaf_range = range(
                    idx_internal * d_sz, (idx_internal + 1) * d_sz
                )
                permuted = true_indices[idx_internal + intermediate_idx_test]
                self.assertTrue(
                    torch.equal(tensor_y[permuted], curr_targets[leaf_range])
                )

            self.assertTrue(torch.equal(tensor_y, curr_targets[-d_sz:]))

        self.assertEqual(
            d_sz * dataset_hierarchy_depth + d_sz, len(curr_dataset)
        )

        def collect_permuted_data(dataset, indices):
            x, y, t = [], [], []
            for idx in indices:
                x_, y_, t_ = dataset[idx]
                x.append(x_)
                y.append(y_)
                t.append(t_)
            return torch.stack(x, dim=0), torch.stack(y, dim=0), torch.tensor(t)

        for idx in range(dataset_hierarchy_depth):
            leaf_range = range(idx * d_sz, (idx + 1) * d_sz)
            permuted = true_indices[idx]

            x_leaf, y_leaf, t_leaf = collect_permuted_data(
                curr_dataset, leaf_range
            )
            self.assertTrue(torch.equal(tensor_x[permuted], x_leaf))
            self.assertTrue(torch.equal(tensor_y[permuted], y_leaf))
            self.assertTrue(torch.equal(tensor_t[permuted], t_leaf))

            trg_leaf = torch.tensor(curr_dataset.targets)[leaf_range]
            self.assertTrue(torch.equal(tensor_y[permuted], trg_leaf))

        slice_idxs = list(
            range(d_sz * dataset_hierarchy_depth, len(curr_dataset))
        )
        x_slice, y_slice, t_slice = collect_permuted_data(
            curr_dataset, slice_idxs
        )
        self.assertTrue(torch.equal(tensor_x, x_slice))
        self.assertTrue(torch.equal(tensor_y, y_slice))
        self.assertTrue(torch.equal(tensor_t, t_slice))

        trg_slice = torch.tensor(curr_dataset.targets)[
            d_sz * dataset_hierarchy_depth :
        ]
        self.assertTrue(torch.equal(tensor_y, trg_slice))

        # If you broke this test it means that dataset merging is not working
        # anymore. you are probably doing something that disable merging
        # (passing custom transforms?)
        # Good luck...
        assert _flatdata_depth(curr_dataset) == 3

    def test_avalanche_concat_datasets_sequentially(self):
        # create list of training datasets
        train = [
            make_classification_dataset(
                TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
            ),
            make_classification_dataset(
                TensorDataset(torch.randn(20, 10), torch.randint(2, 4, (20,)))
            ),
            make_classification_dataset(
                TensorDataset(torch.randn(20, 10), torch.randint(4, 6, (20,)))
            ),
            make_classification_dataset(
                TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
            ),
        ]

        # create list of test datasets
        test = [
            make_classification_dataset(
                TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
            ),
            make_classification_dataset(
                TensorDataset(torch.randn(20, 10), torch.randint(2, 4, (20,)))
            ),
            make_classification_dataset(
                TensorDataset(torch.randn(20, 10), torch.randint(4, 6, (20,)))
            ),
            make_classification_dataset(
                TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
            ),
        ]

        # concatenate datasets
        final_train, _, classes = concat_datasets_sequentially(train, test)

        # merge all classes into a single list
        classes_all = []
        for class_list in classes:
            classes_all.extend(class_list)

        # get the target set of classes
        target_classes = list(set(map(int, final_train.targets)))

        # test for correctness
        self.assertEqual(classes_all, target_classes)


class TransformationSubsetTests(unittest.TestCase):
    def test_avalanche_subset_transform(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[0]
        dataset = classification_subset(dataset_mnist, transform=ToTensor())
        x2, y2, t2 = dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, int)
        self.assertIsInstance(t2, int)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2)
        self.assertEqual(0, t2)

    def test_avalanche_subset_composition(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"),
            download=True,
            transform=RandomCrop(16),
        )
        x, y = dataset_mnist[0]
        self.assertIsInstance(x, Image)
        self.assertEqual([x.width, x.height], [16, 16])
        self.assertIsInstance(y, int)

        dataset = classification_subset(
            dataset_mnist,
            transform=ToTensor(),
            target_transform=lambda target: -1,
        )

        x2, y2, t2 = dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertEqual(x2.shape, (1, 16, 16))
        self.assertIsInstance(y2, int)
        self.assertIsInstance(t2, int)
        self.assertEqual(y2, -1)
        self.assertEqual(0, t2)

    def test_avalanche_subset_indices(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[1000]
        x2, y2 = dataset_mnist[1007]

        dataset = classification_subset(dataset_mnist, indices=[1000, 1007])

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]
        self.assertTrue(pil_images_equal(x, x3))
        self.assertEqual(y, y3)
        self.assertTrue(pil_images_equal(x2, x4))
        self.assertEqual(y2, y4)
        self.assertFalse(pil_images_equal(x, x4))
        self.assertFalse(pil_images_equal(x2, x3))

    def test_avalanche_subset_mapping(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        _, y = dataset_mnist[1000]

        mapping = list(range(10))
        other_classes = list(mapping)
        other_classes.remove(y)

        swap_y = random.choice(other_classes)

        mapping[y] = swap_y
        mapping[swap_y] = y

        dataset = classification_subset(dataset_mnist, class_mapping=mapping)

        _, y2, _ = dataset[1000]
        self.assertEqual(y2, swap_y)

    def test_avalanche_subset_uniform_task_labels(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[1000]
        x2, y2 = dataset_mnist[1007]

        # First, test by passing len(task_labels) == len(dataset_mnist)
        dataset = classification_subset(
            dataset_mnist,
            indices=[1000, 1007],
            task_labels=[1] * len(dataset_mnist),
        )

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]
        self.assertEqual(y, y3)
        self.assertEqual(1, t3)
        self.assertEqual(y2, y4)
        self.assertEqual(1, t4)

        # Secondly, test by passing len(task_labels) == len(indices)
        dataset = classification_subset(
            dataset_mnist, indices=[1000, 1007], task_labels=[1, 1]
        )

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]
        self.assertEqual(y, y3)
        self.assertEqual(1, t3)
        self.assertEqual(y2, y4)
        self.assertEqual(1, t4)

    def test_avalanche_subset_mixed_task_labels(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = dataset_mnist[1000]
        x2, y2 = dataset_mnist[1007]

        full_task_labels = [1] * len(dataset_mnist)
        full_task_labels[1000] = 2
        # First, test by passing len(task_labels) == len(dataset_mnist)
        dataset = classification_subset(
            dataset_mnist, indices=[1000, 1007],
            task_labels=full_task_labels
        )

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]
        self.assertEqual(y, y3)
        self.assertEqual(2, t3)
        self.assertEqual(y2, y4)
        self.assertEqual(1, t4)

        # Secondly, test by passing len(task_labels) == len(indices)
        dataset = classification_subset(
            dataset_mnist, indices=[1000, 1007],
            task_labels=[3, 5]
        )

        x3, y3, t3 = dataset[0]
        x4, y4, t4 = dataset[1]
        self.assertEqual(y, y3)
        self.assertEqual(3, t3)
        self.assertEqual(y2, y4)
        self.assertEqual(5, t4)

    def test_avalanche_subset_task_labels_inheritance(self):
        dataset_mnist = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        random_task_labels = [
            random.randint(0, 10) for _ in range(len(dataset_mnist))
        ]
        dataset_orig = make_classification_dataset(
            dataset_mnist, transform=ToTensor(), task_labels=random_task_labels
        )

        dataset_child = classification_subset(
            dataset_orig, indices=[1000, 1007]
        )
        _, _, t2 = dataset_orig[1000]
        _, _, t5 = dataset_orig[1007]
        _, _, t3 = dataset_child[0]
        _, _, t6 = dataset_child[1]

        self.assertEqual(random_task_labels[1000], t2)
        self.assertEqual(random_task_labels[1007], t5)
        self.assertEqual(random_task_labels[1000], t3)
        self.assertEqual(random_task_labels[1007], t6)

        self.assertListEqual(
            random_task_labels, list(dataset_orig.targets_task_labels)
        )

        self.assertListEqual(
            [random_task_labels[1000], random_task_labels[1007]],
            list(dataset_child.targets_task_labels),
        )

    def test_avalanche_subset_collate_fn_inheritance(self):
        tensor_x = torch.rand(200, 3, 28, 28)
        tensor_y = torch.randint(0, 100, (200,))
        tensor_z = torch.randint(0, 100, (200,))

        def my_collate_fn(patterns):
            x_values = torch.stack([pat[0] for pat in patterns], 0)
            y_values = torch.tensor([pat[1] for pat in patterns]) + 1
            z_values = torch.tensor([-1 for _ in patterns])
            t_values = torch.tensor([pat[3] for pat in patterns])
            return x_values, y_values, z_values, t_values

        def my_collate_fn2(patterns):
            x_values = torch.stack([pat[0] for pat in patterns], 0)
            y_values = torch.tensor([pat[1] for pat in patterns]) + 2
            z_values = torch.tensor([-2 for _ in patterns])
            t_values = torch.tensor([pat[3] for pat in patterns])
            return x_values, y_values, z_values, t_values

        whole_dataset = TensorDataset(tensor_x, tensor_y, tensor_z)
        dataset = make_classification_dataset(
            whole_dataset, collate_fn=my_collate_fn
        )
        inherited = classification_subset(
            dataset, indices=list(range(5, 150)), collate_fn=my_collate_fn2
        )  # Ok

        x, y, z, t = get_mbatch(inherited, batch_size=5)
        self.assertIsInstance(x, Tensor)
        self.assertTrue(torch.equal(tensor_x[5:10], x))
        self.assertTrue(torch.equal(tensor_y[5:10] + 2, y))
        self.assertTrue(torch.equal(torch.full((5,), -2, dtype=torch.long), z))
        self.assertTrue(torch.equal(torch.zeros(5, dtype=torch.long), t))

        classification_dataset = make_classification_dataset(whole_dataset)

        ok_inherited_classification = classification_subset(
            classification_dataset, indices=list(range(5, 150))
        )


class TransformationTensorDatasetTests(unittest.TestCase):
    def test_tensor_dataset_helper_tensor_y(self):

        train_exps = [
            [torch.rand(50, 32, 32), torch.randint(0, 100, (50,))]
            for _ in range(5)
        ]
        test_exps = [
            [torch.rand(23, 32, 32), torch.randint(0, 100, (23,))]
            for _ in range(5)
        ]

        cl_benchmark = create_generic_benchmark_from_tensor_lists(
            train_tensors=train_exps,
            test_tensors=test_exps,
            task_labels=[0] * 5,
        )

        self.assertEqual(5, len(cl_benchmark.train_stream))
        self.assertEqual(5, len(cl_benchmark.test_stream))
        self.assertEqual(5, cl_benchmark.n_experiences)

        for exp_id in range(cl_benchmark.n_experiences):
            benchmark_train_x, benchmark_train_y, _ = load_all_dataset(
                cl_benchmark.train_stream[exp_id].dataset
            )
            benchmark_test_x, benchmark_test_y, _ = load_all_dataset(
                cl_benchmark.test_stream[exp_id].dataset
            )

            self.assertTrue(
                torch.all(torch.eq(train_exps[exp_id][0], benchmark_train_x))
            )
            self.assertTrue(
                torch.all(torch.eq(train_exps[exp_id][1], benchmark_train_y))
            )
            self.assertSequenceEqual(
                train_exps[exp_id][1].tolist(),
                cl_benchmark.train_stream[exp_id].dataset.targets,
            )
            self.assertEqual(0, cl_benchmark.train_stream[exp_id].task_label)

            self.assertTrue(
                torch.all(torch.eq(test_exps[exp_id][0], benchmark_test_x))
            )
            self.assertTrue(
                torch.all(torch.eq(test_exps[exp_id][1], benchmark_test_y))
            )
            self.assertSequenceEqual(
                test_exps[exp_id][1].tolist(),
                cl_benchmark.test_stream[exp_id].dataset.targets,
            )
            self.assertEqual(0, cl_benchmark.test_stream[exp_id].task_label)

    def test_tensor_dataset_helper_list_y(self):
        train_exps = [
            (torch.rand(50, 32, 32), torch.randint(0, 100, (50,)))
            for _ in range(5)
        ]
        test_exps = [
            (torch.rand(23, 32, 32), torch.randint(0, 100, (23,)))
            for _ in range(5)
        ]

        cl_benchmark = create_generic_benchmark_from_tensor_lists(
            train_tensors=train_exps,
            test_tensors=test_exps,
            task_labels=[0] * 5,
        )

        self.assertEqual(5, len(cl_benchmark.train_stream))
        self.assertEqual(5, len(cl_benchmark.test_stream))
        self.assertEqual(5, cl_benchmark.n_experiences)

        for exp_id in range(cl_benchmark.n_experiences):
            benchmark_train_x, benchmark_train_y, _ = load_all_dataset(
                cl_benchmark.train_stream[exp_id].dataset
            )
            benchmark_test_x, benchmark_test_y, _ = load_all_dataset(
                cl_benchmark.test_stream[exp_id].dataset
            )

            self.assertTrue(
                torch.all(torch.eq(train_exps[exp_id][0], benchmark_train_x))
            )
            self.assertSequenceEqual(
                train_exps[exp_id][1], benchmark_train_y.tolist()
            )
            self.assertSequenceEqual(
                train_exps[exp_id][1],
                cl_benchmark.train_stream[exp_id].dataset.targets,
            )
            self.assertEqual(0, cl_benchmark.train_stream[exp_id].task_label)

            self.assertTrue(
                torch.all(torch.eq(test_exps[exp_id][0], benchmark_test_x))
            )
            self.assertSequenceEqual(
                test_exps[exp_id][1], benchmark_test_y.tolist()
            )
            self.assertSequenceEqual(
                test_exps[exp_id][1],
                cl_benchmark.test_stream[exp_id].dataset.targets,
            )
            self.assertEqual(0, cl_benchmark.test_stream[exp_id].task_label)


class AvalancheDatasetTransformOpsTests(unittest.TestCase):
    def test_avalanche_inherit_groups(self):
        original_dataset = MNIST(
            root=default_dataset_location("mnist"), download=True
        )

        def plus_one_target(target):
            return target + 1

        transform_groups = dict(
            train=(ToTensor(), None), eval=(None, plus_one_target)
        )
        x, y = original_dataset[0]
        dataset = make_classification_dataset(
            original_dataset, transform_groups=transform_groups
        )

        x2, y2, _ = dataset[0]
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, int)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2)

        dataset_eval = dataset.eval()
        x3, y3, _ = dataset_eval[0]
        self.assertIsInstance(x3, PIL.Image.Image)
        self.assertIsInstance(y3, int)
        self.assertEqual(y + 1, y3)

        # Regression test for #565
        dataset_inherit = make_classification_dataset(dataset_eval)
        x4, y4, _ = dataset_inherit[0]
        self.assertIsInstance(x4, PIL.Image.Image)
        self.assertIsInstance(y4, int)
        self.assertEqual(y + 1, y4)

        # Regression test for #566
        dataset_sub_train = classification_subset(dataset)
        dataset_sub_eval = dataset_sub_train.eval()
        dataset_sub = classification_subset(dataset_sub_eval, indices=[0])

        x5, y5, _ = dataset_sub[0]
        self.assertIsInstance(x5, PIL.Image.Image)
        self.assertIsInstance(y5, int)
        self.assertEqual(y + 1, y5)
        # End regression tests

        concat_dataset = dataset_sub_eval.concat(dataset_sub)
        x6, y6, _ = concat_dataset[0]
        self.assertIsInstance(x6, PIL.Image.Image)
        self.assertIsInstance(y6, int)
        self.assertEqual(y + 1, y6)

        # DEPRECATED BEHAVIOR
        # concat_dataset_no_inherit_initial =
        #   AvalancheConcatClassificationDataset(
        #     [dataset_sub_eval, dataset]
        # )
        # x7, y7, _ = concat_dataset_no_inherit_initial[0]
        # self.assertIsInstance(x7, Tensor)
        # self.assertIsInstance(y7, int)
        # self.assertEqual(y, y7)

    def test_freeze_transforms(self):
        original_dataset = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = original_dataset[0]
        dataset = make_classification_dataset(
            original_dataset, transform=ToTensor()
        )
        dataset_frozen = dataset.freeze_transforms()

        x2, y2, _ = dataset_frozen[0]
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(y2, int)
        self.assertTrue(torch.equal(ToTensor()(x), x2))
        self.assertEqual(y, y2)

    def test_freeze_transforms_chain(self):
        original_dataset = MNIST(
            root=default_dataset_location("mnist"),
            download=True,
            transform=ToTensor(),
        )
        x, *_ = original_dataset[0]
        self.assertIsInstance(x, Tensor)

        dataset_transform = make_classification_dataset(
            original_dataset, transform=ToPILImage()
        )  # TRANSFORMS: ToTensor -> ToPILImage
        self.assertIsInstance(dataset_transform[0][0], Image)

        dataset_frozen = dataset_transform.freeze_transforms()
        self.assertIsInstance(dataset_frozen[0][0], Image)

        dataset_transform = dataset_transform.replace_current_transform_group(
            None
        )
        self.assertIsInstance(dataset_transform[0][0], Tensor)
        self.assertIsInstance(dataset_frozen[0][0], Image)

        dataset_frozen = dataset_frozen.replace_current_transform_group(
            ToTensor()
        )
        self.assertIsInstance(dataset_frozen[0][0], Tensor)

        dataset_frozen2 = dataset_frozen.freeze_transforms()
        x2, *_ = dataset_frozen2[0]
        self.assertIsInstance(x2, Tensor)

        dataset_frozen = dataset_frozen.replace_current_transform_group(None)
        x2, *_ = dataset_frozen2[0]
        self.assertIsInstance(x2, Tensor)
        x2, *_ = dataset_frozen[0]
        self.assertIsInstance(x2, Image)

    def test_replace_transforms(self):
        original_dataset = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, y = original_dataset[0]
        dataset = make_classification_dataset(
            original_dataset, transform=ToTensor()
        )
        x2, *_ = dataset[0]
        dataset_reset = dataset.replace_current_transform_group(None)
        x3, *_ = dataset_reset[0]

        self.assertIsInstance(x, Image)
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(x3, Image)

        dataset_reset = dataset_reset.replace_current_transform_group(
            ToTensor()
        )
        x4, *_ = dataset_reset[0]
        self.assertIsInstance(x4, Tensor)

        dataset_reset.replace_current_transform_group(None)

        x5, *_ = dataset_reset[0]
        self.assertIsInstance(x5, Tensor)

        dataset_other = make_classification_dataset(dataset_reset)
        dataset_other = dataset_other.replace_current_transform_group(
            (None, lambda lll: lll + 1)
        )

        _, y6, _ = dataset_other[0]
        self.assertEqual(y + 1, y6)

    def test_transforms_replace_freeze_mix(self):
        original_dataset = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        x, _ = original_dataset[0]
        dataset = make_classification_dataset(
            original_dataset, transform=ToTensor()
        )
        x2, *_ = dataset[0]
        dataset_reset = dataset.replace_current_transform_group((None, None))
        x3, *_ = dataset_reset[0]

        self.assertIsInstance(x, Image)
        self.assertIsInstance(x2, Tensor)
        self.assertIsInstance(x3, Image)

        dataset_frozen = dataset.freeze_transforms()

        x4, *_ = dataset_frozen[0]
        self.assertIsInstance(x4, Tensor)

        dataset_frozen_reset = dataset_frozen.replace_current_transform_group(
            (None, None)
        )

        x5, *_ = dataset_frozen_reset[0]
        self.assertIsInstance(x5, Tensor)

    def test_transforms_groups_base_usage(self):
        original_dataset = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        dataset = make_classification_dataset(
            original_dataset,
            transform_groups=dict(
                train=(ToTensor(), None),
                eval=(None, Lambda(lambda t: float(t))),
            ),
        )

        x, y, _ = dataset[0]
        self.assertIsInstance(x, Tensor)
        self.assertIsInstance(y, int)

        dataset_test = dataset.eval()

        x2, y2, _ = dataset_test[0]
        x3, y3, _ = dataset[0]
        self.assertIsInstance(x2, Image)
        self.assertIsInstance(y2, float)
        self.assertIsInstance(x3, Tensor)
        self.assertIsInstance(y3, int)

        dataset_train = dataset.train()
        dataset = dataset.replace_current_transform_group(None)

        x4, y4, _ = dataset_train[0]
        x5, y5, _ = dataset[0]
        self.assertIsInstance(x4, Tensor)
        self.assertIsInstance(y4, int)
        self.assertIsInstance(x5, Image)
        self.assertIsInstance(y5, int)

    def test_transforms_groups_constructor_error(self):
        original_dataset = load_image_data()

        with self.assertRaises(Exception):
            # Test is not a tuple has only one element
            dataset = make_classification_dataset(
                original_dataset,
                transform_groups=dict(
                    train=(ToTensor(), None),
                    eval=[None, Lambda(lambda t: float(t))],
                ),
            )

        with self.assertRaises(Exception):
            # Train is None
            dataset = make_classification_dataset(
                original_dataset,
                transform_groups=dict(
                    train=None, eval=(None, Lambda(lambda t: float(t)))
                ),
            )

        with self.assertRaises(Exception):
            # transform_groups is not a dictionary
            dataset = make_classification_dataset(
                original_dataset, transform_groups="Hello world!"
            )

    def test_transforms_groups_alternative_default_group(self):
        original_dataset = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        dataset = make_classification_dataset(
            original_dataset,
            transform_groups=dict(train=(ToTensor(), None), eval=(None, None)),
            initial_transform_group="eval",
        )

        x, *_ = dataset[0]
        self.assertIsInstance(x, Image)

        dataset_test = dataset.eval()

        x2, *_ = dataset_test[0]
        x3, *_ = dataset[0]
        self.assertIsInstance(x2, Image)
        self.assertIsInstance(x3, Image)

    def test_transforms_groups_partial_constructor(self):
        original_dataset = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        dataset = make_classification_dataset(
            original_dataset, transform_groups=dict(train=(ToTensor(), None))
        )

        x, *_ = dataset[0]
        self.assertIsInstance(x, Tensor)

        dataset = dataset.eval()
        x2, *_ = dataset[0]
        self.assertIsInstance(x2, Tensor)

    def test_transforms_groups_multiple_groups(self):
        original_dataset = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        dataset = make_classification_dataset(
            original_dataset,
            transform_groups=dict(
                train=(ToTensor(), None),
                eval=(None, None),
                other=(
                    Compose(
                        [ToTensor(), Lambda(lambda tensor: tensor.numpy())]
                    ),
                    None,
                ),
            ),
        )

        x, *_ = dataset[0]
        self.assertIsInstance(x, Tensor)

        dataset = dataset.eval()
        x2, *_ = dataset[0]
        self.assertIsInstance(x2, Image)

        dataset = dataset.with_transforms("other")
        x3, *_ = dataset[0]
        self.assertIsInstance(x3, np.ndarray)

    def test_transformation_concat_dataset(self):
        original_dataset = MNIST(
            root=default_dataset_location("mnist"), download=True
        )
        original_dataset2 = MNIST(
            root=default_dataset_location("mnist"), download=True
        )

        dataset = concat_datasets([original_dataset, original_dataset2])
        self.assertEqual(
            len(original_dataset) + len(original_dataset2), len(dataset)
        )

    def test_transformation_concat_dataset_groups(self):
        original_dataset = make_classification_dataset(
            MNIST(root=default_dataset_location("mnist"), download=True),
            transform_groups=dict(eval=(None, None), train=(ToTensor(), None)),
        )
        original_dataset2 = make_classification_dataset(
            MNIST(root=default_dataset_location("mnist"), download=True),
            transform_groups=dict(train=(None, None), eval=(ToTensor(), None)),
        )

        dataset = original_dataset.concat(original_dataset2)

        self.assertEqual(
            len(original_dataset) + len(original_dataset2), len(dataset)
        )

        x, *_ = dataset[0]
        x2, *_ = dataset[len(original_dataset)]
        self.assertIsInstance(x, Tensor)
        self.assertIsInstance(x2, Image)

        dataset = dataset.eval()

        x3, *_ = dataset[0]
        x4, *_ = dataset[len(original_dataset)]
        self.assertIsInstance(x3, Image)
        self.assertIsInstance(x4, Tensor)


if __name__ == "__main__":
    unittest.main()
