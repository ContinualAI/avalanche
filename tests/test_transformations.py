import copy
import unittest
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.dataset_traversal_utils import single_flat_dataset
from avalanche.benchmarks.utils.detection_dataset import DetectionDataset
from avalanche.benchmarks.classic.cmnist import SplitMNIST
from avalanche.benchmarks.utils.transform_groups import TransformGroups

from avalanche.benchmarks.utils.transforms import (
    MultiParamCompose,
    MultiParamTransformCallable,
    TupleTransform,
    flat_transforms_recursive,
)

import torch
from PIL import ImageChops
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import MNIST
from torchvision.transforms import (
    ToTensor,
    Compose,
    CenterCrop,
    Normalize,
    Lambda,
    RandomHorizontalFlip,
)
from torchvision.transforms.functional import to_tensor
from PIL.Image import Image

from tests.unit_tests_utils import get_fast_detection_datasets


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


class TransformsTest(unittest.TestCase):
    def test_multi_param_transform_callable(self):
        dataset: DetectionDataset
        dataset, _ = get_fast_detection_datasets()

        boxes = []
        i = 0
        while len(boxes) == 0:
            x_orig, y_orig, t_orig = dataset[i]
            boxes = y_orig["boxes"]
            i += 1
        i -= 1

        x_expect = to_tensor(copy.deepcopy(x_orig))
        x_expect[0][0] += 1

        y_expect = copy.deepcopy(y_orig)
        y_expect["boxes"][0][0] += 1

        def do_something_xy(img, target):
            img = to_tensor(img)
            img[0][0] += 1
            target["boxes"][0][0] += 1
            return img, target

        uut = MultiParamTransformCallable(do_something_xy)

        # Test __eq__
        uut_eq = MultiParamTransformCallable(do_something_xy)
        self.assertTrue(uut == uut_eq)
        self.assertTrue(uut_eq == uut)

        x, y, t = uut(*dataset[i])

        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, dict)
        self.assertIsInstance(t, int)

        self.assertTrue(torch.equal(x_expect, x))
        keys = set(y_expect.keys())
        self.assertSetEqual(keys, set(y.keys()))

        for k in keys:
            self.assertTrue(torch.equal(y_expect[k], y[k]), msg=f"Wrong {k}")

    def test_multi_param_compose(self):
        dataset: DetectionDataset
        dataset, _ = get_fast_detection_datasets()

        assert_called = 0

        def do_something_xy(img: Tensor, target):
            nonlocal assert_called
            assert_called += 1
            img = img.clone()
            img[0][0] += 1
            target["boxes"][0][0] += 1
            return img, target

        t_x = lambda x, y: (to_tensor(x), y)
        t_xy = do_something_xy
        t_x_1_element = ToTensor()

        boxes = []
        i = 0
        while len(boxes) == 0:
            x_orig, y_orig, t_orig = dataset[i]
            boxes = y_orig["boxes"]
            i += 1
        i -= 1

        x_expect = to_tensor(copy.deepcopy(x_orig))
        x_expect[0][0] += 1

        y_expect = copy.deepcopy(y_orig)
        y_expect["boxes"][0][0] += 1

        uut_2 = MultiParamCompose([t_x, t_xy])

        # Test __eq__
        uut_2_eq = MultiParamCompose([t_x, t_xy])
        self.assertTrue(uut_2 == uut_2_eq)
        self.assertTrue(uut_2_eq == uut_2)

        with self.assertWarns(Warning):
            # Assert that the following warn is raised:
            # "Transformations define a different number of parameters. ..."
            uut_1 = MultiParamCompose([t_x_1_element, t_xy])

        for uut, uut_type in zip((uut_1, uut_2), ("uut_1", "uut_2")):
            with self.subTest(uut_type=uut_type):
                initial_assert_called = assert_called

                x, y, t = uut(*dataset[i])

                self.assertEqual(initial_assert_called + 1, assert_called)

                self.assertIsInstance(x, torch.Tensor)
                self.assertIsInstance(y, dict)
                self.assertIsInstance(t, int)

                self.assertTrue(torch.equal(x_expect, x))
                keys = set(y_expect.keys())
                self.assertSetEqual(keys, set(y.keys()))

                for k in keys:
                    self.assertTrue(torch.equal(y_expect[k], y[k]), msg=f"Wrong {k}")

    def test_tuple_transform(self):
        dataset = MNIST(root=default_dataset_location("mnist"), download=True)

        t_x = ToTensor()
        t_y = lambda element: element + 1
        t_bad = lambda element: element - 1

        uut = TupleTransform([t_x, t_y])

        uut_eq = TupleTransform(
            (t_x, t_y)  # Also test with a tuple instead of a list here
        )

        uut_not_x = TupleTransform([None, t_y])

        uut_bad = TupleTransform((t_x, t_y, t_bad))

        x_orig, y_orig = dataset[0]

        # Test with x transform
        x, y = uut(*dataset[0])

        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, int)

        self.assertTrue(torch.equal(to_tensor(x_orig), x))
        self.assertEqual(y_orig + 1, y)

        # Test without x transform
        x, y = uut_not_x(*dataset[0])

        self.assertIsInstance(x, Image)
        self.assertIsInstance(y, int)

        self.assertEqual(x_orig, x)
        self.assertEqual(y_orig + 1, y)

        # Check __eq__ works
        self.assertTrue(uut == uut_eq)
        self.assertTrue(uut_eq == uut)

        self.assertFalse(uut == uut_not_x)
        self.assertFalse(uut_not_x == uut)

        with self.assertRaises(Exception):
            # uut_bad has 3 transforms, which is incorrect
            uut_bad(*dataset[0])

    def test_flat_transforms_recursive_only_torchvision(self):
        x_transform = ToTensor()
        x_transform_list = [CenterCrop(24), Normalize(0.5, 0.1)]
        x_transform_composed = Compose(x_transform_list)

        expected_x = [x_transform] + x_transform_list

        # Single transforms checks
        self.assertSequenceEqual(
            [x_transform], flat_transforms_recursive([x_transform], 0)
        )

        self.assertSequenceEqual(
            [x_transform], flat_transforms_recursive(x_transform, 0)
        )

        self.assertSequenceEqual(
            x_transform_list, flat_transforms_recursive(x_transform_list, 0)
        )

        self.assertSequenceEqual(
            x_transform_list, flat_transforms_recursive(x_transform_composed, 0)
        )

        # Hybrid list checks
        self.assertSequenceEqual(
            expected_x,
            flat_transforms_recursive([x_transform, x_transform_composed], 0),
        )

    def test_flat_transforms_recursive_from_dataset(self):
        x_transform = ToTensor()
        x_transform_list = [CenterCrop(24), Normalize(0.5, 0.1)]
        x_transform_additional = RandomHorizontalFlip(p=0.2)
        x_transform_composed = Compose(x_transform_list)

        expected_x = [x_transform] + x_transform_list + [x_transform_additional]

        y_transform = Lambda(lambda x: max(0, x - 1))

        dataset = MNIST(
            root=default_dataset_location("mnist"), download=True, transform=x_transform
        )

        transform_group = TransformGroups(
            transform_groups={
                "train": TupleTransform([x_transform_composed, y_transform])
            }
        )

        transform_group_additional_1a = TransformGroups(
            transform_groups={"train": TupleTransform([x_transform_additional, None])}
        )
        transform_group_additional_1b = TransformGroups(
            transform_groups={"train": TupleTransform([x_transform_additional, None])}
        )

        avl_dataset = AvalancheDataset([dataset], transform_groups=transform_group)

        avl_subset_1 = avl_dataset.subset([1, 2, 3])
        avl_subset_2 = avl_dataset.subset([5, 6, 7])

        avl_subset_1 = AvalancheDataset(
            [avl_subset_1], transform_groups=transform_group_additional_1a
        )
        avl_subset_2 = AvalancheDataset(
            [avl_subset_2], transform_groups=transform_group_additional_1b
        )

        for concat_type, avl_concat in zip(
            ["avalanche", "pytorch"],
            [
                avl_subset_1.concat(avl_subset_2),
                ConcatDataset([avl_subset_1, avl_subset_2]),
            ],
        ):
            with self.subTest("Concatenation type", concat_type=concat_type):
                _, _, transforms = single_flat_dataset(avl_concat)
                x_flattened = flat_transforms_recursive(transforms, 0)
                y_flattened = flat_transforms_recursive(transforms, 1)

                self.assertSequenceEqual(expected_x, x_flattened)
                self.assertSequenceEqual([y_transform], y_flattened)


if __name__ == "__main__":
    unittest.main()
