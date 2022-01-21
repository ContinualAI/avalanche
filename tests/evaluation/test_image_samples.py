import unittest
from unittest.mock import MagicMock

from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.utils import save_image

from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks.utils import AvalancheTensorDataset
from avalanche.evaluation.metrics import ImagesSamplePlugin


class ImageSamplesTests(unittest.TestCase):
    def test_image_samples(args):
        p_metric = ImagesSamplePlugin(
            n_cols=5, n_rows=5, group=True, mode="train"
        )

        scenario = SplitMNIST(5)
        curr_exp = scenario.train_stream[0]
        curr_dataset = curr_exp.dataset
        strategy_mock = MagicMock(
            eval_mb_size=32, experience=curr_exp, adapted_dataset=curr_dataset
        )

        mval = p_metric.after_train_dataset_adaptation(strategy_mock)
        img_grid = mval[0].value.image

        # save_image(img_grid, './logs/test_image_grid.png')

    def test_tensor_samples(args):
        p_metric = ImagesSamplePlugin(
            n_cols=5, n_rows=5, group=True, mode="train"
        )

        scenario = SplitMNIST(5)
        curr_exp = scenario.train_stream[0]
        for mb in DataLoader(curr_exp.dataset, batch_size=32):
            break
        curr_dataset = AvalancheTensorDataset(*mb[:2], targets=mb[1])

        strategy_mock = MagicMock(
            eval_mb_size=32, experience=curr_exp, adapted_dataset=curr_dataset
        )

        mval = p_metric.after_train_dataset_adaptation(strategy_mock)
        img_grid = mval[0].value.image

        # save_image(img_grid, './logs/test_tensor_grid.png')

    def test_samples_augmentations(args):
        scenario = SplitMNIST(5)
        curr_exp = scenario.train_stream[0]

        # we use a ReSize transform because it's easy to detect if it's been
        # applied without looking at the image.
        curr_dataset = curr_exp.dataset.replace_transforms(
            transform=Compose([Resize(8), ToTensor()]), target_transform=None
        )

        ##########################################
        # WITH AUGMENTATIONS
        ##########################################
        p_metric = ImagesSamplePlugin(
            n_cols=5,
            n_rows=5,
            group=True,
            mode="train",
            disable_augmentations=False,
        )

        strategy_mock = MagicMock(
            eval_mb_size=32, experience=curr_exp, adapted_dataset=curr_dataset
        )

        mval = p_metric.after_train_dataset_adaptation(strategy_mock)
        img_grid = mval[0].value.image
        assert img_grid.shape == (3, 52, 52)
        # save_image(img_grid, './logs/test_image_with_aug.png')

        ##########################################
        # WITHOUT AUGMENTATIONS
        ##########################################
        p_metric = ImagesSamplePlugin(
            n_cols=5,
            n_rows=5,
            group=True,
            mode="train",
            disable_augmentations=True,
        )

        strategy_mock = MagicMock(
            eval_mb_size=32, experience=curr_exp, adapted_dataset=curr_dataset
        )

        mval = p_metric.after_train_dataset_adaptation(strategy_mock)
        img_grid = mval[0].value.image
        assert img_grid.shape == (3, 152, 152)
        # save_image(img_grid, './logs/test_image_with_aug.png')
