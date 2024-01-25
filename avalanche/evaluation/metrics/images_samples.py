from typing import List, TYPE_CHECKING, Tuple, Literal

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from avalanche.benchmarks.utils.data import AvalancheDataset

from avalanche.evaluation.metric_definitions import PluginMetric

from avalanche.evaluation.metric_results import (
    MetricResult,
    TensorImage,
    MetricValue,
)
from avalanche.evaluation.metric_utils import get_metric_name


if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class ImagesSamplePlugin(PluginMetric):
    """Metric used to sample random images.

    Only images in strategy.adapted dataset are used. Images added in the
    dataloader (like the replay plugins do) are missed.
    By default data augmentation are removed.

    :param n_rows: The numbers of raws to use in the grid of images.
    :param n_cols: The numbers of columns to use in the grid of images.
    :param group: If True, images will be grouped by (task, label)
    :param mode: The plugin can be used at train or eval time.
    :param disable_augmentations: determines whether to show the augmented
        images or the raw images (default: True).
    :return: The corresponding plugins.
    """

    def __init__(
        self,
        *,
        mode: Literal["train", "eval", "both"],
        n_cols: int,
        n_rows: int,
        group: bool = True,
        disable_augmentations: bool = True,
    ):
        super().__init__()
        self.group = group
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.mode = mode
        self.disable_augmentations = disable_augmentations

        self.images: List[Tensor] = []
        self.n_wanted_images = self.n_cols * self.n_rows

    def after_train_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        if self.mode == "train" or self.mode == "both":
            return self._make_grid_sample(strategy)
        return None

    def after_eval_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        if self.mode == "eval" or self.mode == "both":
            return self._make_grid_sample(strategy)
        return None

    def reset(self) -> None:
        self.images = []

    def result(self) -> List[Tensor]:
        return self.images

    def __str__(self):
        return "images"

    def _make_grid_sample(self, strategy: "SupervisedTemplate") -> "MetricResult":
        self._load_sorted_images(strategy)

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
                    make_grid(list(self.images), normalize=False, nrow=self.n_cols)
                ),
                x_plot=strategy.clock.train_iterations,
            )
        ]

    def _load_sorted_images(self, strategy: "SupervisedTemplate"):
        self.reset()
        self.images, labels, tasks = self._load_data(strategy)
        if self.group:
            self._sort_images(labels, tasks)

    def _load_data(
        self, strategy: "SupervisedTemplate"
    ) -> Tuple[List[Tensor], List[int], List[int]]:
        assert strategy.adapted_dataset is not None
        dataloader = self._make_dataloader(
            strategy.adapted_dataset, strategy.eval_mb_size
        )

        images: List[Tensor] = []
        labels: List[Tensor] = []
        tasks: List[Tensor] = []

        for batch_images, batch_labels, batch_tasks in dataloader:
            n_missing_images = self.n_wanted_images - len(images)
            labels.extend(batch_labels[:n_missing_images].tolist())
            tasks.extend(batch_tasks[:n_missing_images].tolist())
            images.extend(batch_images[:n_missing_images])
            if len(images) == self.n_wanted_images:
                return images, labels, tasks
        return images, labels, tasks

    def _sort_images(self, labels: List[int], tasks: List[int]):
        self.images = [
            image
            for task, label, image in sorted(
                zip(tasks, labels, self.images),
                key=lambda t: (t[0], t[1]),
            )
        ]

    def _make_dataloader(self, data: AvalancheDataset, mb_size: int) -> DataLoader:
        if self.disable_augmentations:
            data = data.replace_current_transform_group(_MaybeToTensor())
        collate_fn = data.collate_fn if hasattr(data, "collate_fn") else None
        return DataLoader(
            dataset=data,
            batch_size=min(mb_size, self.n_wanted_images),
            shuffle=True,
            collate_fn=collate_fn,
        )


class _MaybeToTensor(ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. Pytorch tensors
    are left as is.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, Tensor):
            return pic
        return super().__call__(pic)


def images_samples_metrics(
    *,
    n_rows: int = 8,
    n_cols: int = 8,
    group: bool = True,
    on_train: bool = True,
    on_eval: bool = False,
) -> List[ImagesSamplePlugin]:
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
    plugins: List[ImagesSamplePlugin] = []
    if on_eval:
        plugins.append(
            ImagesSamplePlugin(mode="eval", n_rows=n_rows, n_cols=n_cols, group=group)
        )
    if on_train:
        plugins.append(
            ImagesSamplePlugin(mode="train", n_rows=n_rows, n_cols=n_cols, group=group)
        )
    return plugins


__all__ = ["images_samples_metrics", "ImagesSamplePlugin"]
