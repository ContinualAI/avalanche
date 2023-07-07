import copy
import unittest
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.benchmarks.utils.detection_dataset import DetectionDataset


from avalanche.benchmarks.utils.transforms import (
    MultiParamCompose,
    MultiParamTransformCallable,
    TupleTransform,
)

import torch
from PIL import ImageChops
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
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
            boxes = y_orig['boxes']
            i += 1
        i -= 1

        x_expect = to_tensor(copy.deepcopy(x_orig))
        x_expect[0][0] += 1

        y_expect = copy.deepcopy(y_orig)
        y_expect['boxes'][0][0] += 1

        def do_something_xy(img, target):
            img = to_tensor(img)
            img[0][0] += 1
            target['boxes'][0][0] += 1
            return img, target
        
        uut = MultiParamTransformCallable(
            do_something_xy
        )

        # Test __eq__
        uut_eq = MultiParamTransformCallable(
            do_something_xy
        )
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
            self.assertTrue(
                torch.equal(y_expect[k], y[k]),
                msg=f'Wrong {k}'
            )

    def test_multi_param_compose(self):
        dataset: DetectionDataset
        dataset, _ = get_fast_detection_datasets()

        assert_called = 0
        def do_something_xy(img: Tensor, target):
            nonlocal assert_called
            assert_called += 1
            img = img.clone()
            img[0][0] += 1
            target['boxes'][0][0] += 1
            return img, target

        t_x = lambda x, y: (to_tensor(x), y)
        t_xy = do_something_xy
        t_x_1_element = ToTensor()

        boxes = []
        i = 0
        while len(boxes) == 0:
            x_orig, y_orig, t_orig = dataset[i]
            boxes = y_orig['boxes']
            i += 1
        i -= 1

        x_expect = to_tensor(copy.deepcopy(x_orig))
        x_expect[0][0] += 1

        y_expect = copy.deepcopy(y_orig)
        y_expect['boxes'][0][0] += 1

        uut_2 = MultiParamCompose(
            [t_x, t_xy]
        )

        # Test __eq__
        uut_2_eq = MultiParamCompose(
            [t_x, t_xy]
        )
        self.assertTrue(uut_2 == uut_2_eq)
        self.assertTrue(uut_2_eq == uut_2)

        with self.assertWarns(Warning):
            # Assert that the following warn is raised:
            # "Transformations define a different number of parameters. ..."
            uut_1 = MultiParamCompose(
                [t_x_1_element, t_xy]
            )

        for uut, uut_type in zip((uut_1, uut_2), ('uut_1', 'uut_2')):
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
                    self.assertTrue(
                        torch.equal(y_expect[k], y[k]),
                        msg=f'Wrong {k}'
                    )

    def test_tuple_transform(self):
        dataset = MNIST(
            root=default_dataset_location("mnist"),
            download=True
        )

        t_x = ToTensor()
        t_y = lambda element: element+1
        t_bad = lambda element: element-1
        
        uut = TupleTransform(
            [t_x, t_y]
        )

        uut_eq = TupleTransform(
            (t_x, t_y)  # Also test with a tuple instead of a list here
        )

        uut_not_x = TupleTransform(
            [None, t_y]
        )

        uut_bad = TupleTransform(
            (t_x, t_y, t_bad)
        )

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


if __name__ == "__main__":
    unittest.main()
