from typing import List, TYPE_CHECKING, Tuple

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from avalanche.evaluation.metric_definitions import PluginMetric

from avalanche.evaluation.metric_results import (
    MetricResult,
    TensorImage,
    MetricValue,
)
from avalanche.evaluation.metric_utils import get_metric_name

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class ImagesSamplePlugin(PluginMetric):
    """
    A metric used to sample images at random.
    No data augmentation is shown.
    Only images in strategy.adapted dataset are used. Images added in the
    dataloader (like the replay plugins do) are missed.

    :param n_rows: The numbers of raws to use in the grid of images.
    :param n_cols: The numbers of columns to use in the grid of images.
    :param group: If True, images will be grouped by (task, label)
    :param mode: The plugin can be used at train or eval time.
    :return: The corresponding plugins.
    """

    def __init__(
        self,
        *,
        mode: Literal["train", "eval"],
        n_cols: int,
        n_rows: int,
        group: bool = True,
    ):
        super().__init__()
        self.group = group
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.mode = mode

        self.images: List[Tensor] = []
        self.n_wanted_images = self.n_cols * self.n_rows

    def after_train_dataset_adaptation(
        self, strategy: "BaseStrategy"
    ) -> "MetricResult":
        if self.mode == "train":
            return self.make_grid_sample(strategy)

    def after_eval_dataset_adaptation(
        self, strategy: "BaseStrategy"
    ) -> "MetricResult":
        if self.mode == "eval":
            return self.make_grid_sample(strategy)

    def make_grid_sample(self, strategy: "BaseStrategy") -> "MetricResult":
        self.load_sorted_images(strategy)

        return [
            MetricValue(
                self,
                name=get_metric_name(
                    self,
                    strategy,
                    add_experience=self.mode == "eval",
                    add_task=True,
                ),
                value=TensorImage(
                    make_grid(
                        list(self.images), normalize=False, nrow=self.n_cols
                    )
                ),
                x_plot=strategy.clock.train_iterations,
            )
        ]

    def load_sorted_images(self, strategy: "BaseStrategy"):
        self.reset()
        self.images, labels, tasks = self.load_data(strategy)
        if self.group:
            self.sort_images(labels, tasks)

    def load_data(
        self, strategy: "BaseStrategy"
    ) -> Tuple[List[Tensor], List[int], List[int]]:
        dataloader = self.make_dataloader(strategy)

        images, labels, tasks = [], [], []

        for batch_images, batch_labels, batch_tasks in dataloader:
            n_missing_images = self.n_wanted_images - len(images)
            labels.extend(batch_labels[:n_missing_images].tolist())
            tasks.extend(batch_tasks[:n_missing_images].tolist())
            images.extend(batch_images[:n_missing_images])
            if len(images) == self.n_wanted_images:
                return images, labels, tasks

    def sort_images(self, labels: List[int], tasks: List[int]):
        self.images = [
            image
            for task, label, image in sorted(
                zip(tasks, labels, self.images), key=lambda t: (t[0], t[1]),
            )
        ]

    def make_dataloader(self, strategy: "BaseStrategy") -> DataLoader:
        return DataLoader(
            dataset=strategy.adapted_dataset.replace_transforms(
                transform=ToTensor(), target_transform=None,
            ),
            batch_size=min(strategy.eval_mb_size, self.n_wanted_images),
            shuffle=True,
        )

    def reset(self) -> None:
        self.images = []

    def result(self) -> List[Tensor]:
        return self.images

    def __str__(self):
        return "images"


def images_samples_metrics(
    *,
    n_rows: int = 8,
    n_cols: int = 8,
    group: bool = True,
    on_train: bool = True,
    on_eval: bool = False,
) -> List[PluginMetric]:
    """
    Create the plugins to log some images samples in grids.
    No data augmentation is shown.
    Only images in strategy.adapted dataset are used. Images added in the
    dataloader (like the replay plugins do) are missed.

    :param n_rows: The numbers of raws to use in the grid of images.
    :param n_cols: The numbers of columns to use in the grid of images.
    :param group: If True, images will be grouped by (task, label)
    :param on_train: If True, will emit some images samples during training.
    :param on_eval: If True, will emit some images samples during evaluation.
    :return: The corresponding plugins.
    """
    plugins = []
    if on_eval:
        plugins.append(
            ImagesSamplePlugin(
                mode="eval", n_rows=n_rows, n_cols=n_cols, group=group
            )
        )
    if on_train:
        plugins.append(
            ImagesSamplePlugin(
                mode="train", n_rows=n_rows, n_cols=n_cols, group=group
            )
        )
    return plugins


__all__ = [
    'images_samples_metrics',
    'ImagesSamplePlugin'
]
