import random
from collections import defaultdict
from itertools import chain
from typing import Dict, Iterable, List, TYPE_CHECKING, Optional, Tuple

from torch import Tensor
from torchvision.transforms import Normalize
from torchvision.utils import make_grid

from avalanche.evaluation import Metric
from avalanche.evaluation.metric_definitions import (
    GenericPluginMetric,
    PluginMetric,
)

from avalanche.evaluation.metric_results import (
    MetricResult,
    TensorImage,
    MetricValue,
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


GroupByMode = Literal["label", "task", None]


class ImagesSamplesMetric(Metric):
    """
    A metric used to sample images at random.

    :param means: The means that were used during a normalization step, to
        reverse it before displaying
    :param std: The std that were used during a normalization step, to
        reverse it before displaying
    :param n_images: The number of images to sample.
    """

    def __init__(
        self, *, n_images: int, means: Iterable[float], std: Iterable[float]
    ):
        self.n_images = n_images
        self.un_normalize = make_un_normalize(means, std)
        self.images: List[Tensor] = []
        self.n_seen_images = 0

    def reset(self):
        """
        Drop the current sampling.
        """
        self.images = []
        self.n_seen_images = 0

    def update(self, images: Tensor):
        """
        Will update the selected samples to randomly select some of the new
            images. New images have the same probability of being selected than
            old images.

        :param images: New images from which to sample.
        """
        free_memory = self.n_images - len(self.images)

        images_to_add = images[:free_memory]
        self.images.extend(self.un_normalize(im) for im in images_to_add)
        self.n_seen_images = len(images_to_add)

        for image in images[free_memory:]:
            self.n_seen_images += 1
            if random.random() < 1 / self.n_seen_images:
                i = random.randint(0, self.n_images - 1)
                self.images[i] = self.un_normalize(image)

    def result(self) -> List[Tensor]:
        """
        :return: The image that were sampled during the successive update calls
        """
        return self.images


class BalancedImagesSamplesMetric(Metric):
    """
    A metric used to sample images balanced by label.

    :param means: The means that were used during a normalization step, to
        reverse it before displaying
    :param std: The std that were used during a normalization step, to
        reverse it before displaying
    :param n_images_per_label: The number of images to sample per label.
    :param group_by: The images can be grouped by labels (1 list is emitted per
        tuple task,label), by tasks (1 list per task, the images of same labels
        will follow each other). If None, only 1 list is emitted, images from
        the same labels will follow each other.
    """

    def __init__(
        self,
        *,
        means: Iterable[float],
        std: Iterable[float],
        n_images_per_label: Optional[int],
        group_by: GroupByMode,
    ):
        self.group_by = group_by
        self.n_images_per_label = n_images_per_label
        self.task2label2images: Dict[int, Dict[int, List[Tensor]]] = {}
        self.un_normalize = make_un_normalize(means, std)
        self.reset()

    def reset(self) -> None:
        """
        Drop the current sampling.
        """
        self.task2label2images = defaultdict(lambda: defaultdict(list))

    def update(self, images: Tensor, labels: List[int], tasks: List[int]):
        """
        Update the sampling with new images.

        :param images: New images to sample.
        :param labels: Corresponding labels.
        :param tasks: Corresponding tasks labels.
        """
        for image, label, task in zip(images, labels, tasks):
            label_images = self.task2label2images[task][label]
            if len(label_images) < self.n_images_per_label:
                label_images.append(self.un_normalize(image))

    def results(self) -> Iterable[Tuple[Iterable[Tensor], str]]:
        """
        :return: An iterable of tuples (images, name). The number of elements
            is controlled by the group_by attribute of the object (only 1 if no
            grouping, 1 per task if grouping by task, or one per tuple
            task,label if grouping per label).
        """
        if self.group_by is None:
            return [
                (
                    chain.from_iterable(
                        chain.from_iterable(label2images.values())
                        for task, label2images in self.task2label2images.items()
                    ),
                    "",
                )
            ]
        elif self.group_by == "task":
            return [
                (chain.from_iterable(label2images.values()), f"/task_{task}",)
                for task, label2images in self.task2label2images.items()
            ]
        elif self.group_by == "label":
            return [
                (images, f"/task_{task}/class_{label}")
                for task, label2images in self.task2label2images.items()
                for label, images in label2images.items()
            ]


class RandomImagesSamplesPlugin(GenericPluginMetric[TensorImage]):
    """
    A plugin to log some images samples taken at random in a grid.

    :param means: The means that were used during a normalization step, to
        reverse it before displaying
    :param std: The std that were used during a normalization step, to
        reverse it before displaying
    :param n_rows: The numbers of raws to use in the grid of images.
    :param n_cols: The numbers of columns to use in the grid of images.
    :param mode: This plugin can work during training or evaluation.
    """

    def __init__(
        self,
        means: Iterable[float],
        std: Iterable[float],
        n_rows: int = 3,
        n_cols: int = 5,
        mode: Literal["train", "eval"] = "train",
    ):
        super().__init__(
            ImagesSamplesMetric(n_images=n_rows * n_rows, means=means, std=std),
            mode=mode,
            emit_at="stream",
            reset_at="stream",
        )
        self.n_cols = n_cols

    def before_training_iteration(
        self, strategy: "BaseStrategy"
    ) -> "MetricResult":
        if self._mode == "train":
            return self._update(strategy)

    def before_eval_iteration(self, strategy: "BaseStrategy") -> "MetricResult":
        if self._mode == "eval":
            return self._update(strategy)

    def _update(self, strategy: "BaseStrategy"):
        self._metric.update(strategy.mb_x)

    def _package_result(self, strategy: "BaseStrategy") -> "MetricResult":
        return [
            MetricValue(
                self,
                name=f"random_images/{self._mode}_phase",
                value=TensorImage(
                    make_grid(
                        list(self._metric.result()),
                        normalize=False,
                        nrow=self.n_cols,
                    )
                ),
                x_plot=self.get_global_counter(),
            )
        ]


class BalancedImagesSamplesPlugin(GenericPluginMetric[TensorImage]):
    """
    A plugin to log some images samples in grids, with each label being
        represented evenly.

    :param means: The means that were used during a normalization step, to
        reverse it before displaying
    :param std: The std that were used during a normalization step, to
        reverse it before displaying
    :param n_rows: The numbers of raws to use in the grid of images. Only used
        when grouping by labels, otherwise the number of labels is used.
    :param n_cols: The numbers of columns to use in the grid of images.
    :param mode: This plugin can work during training or evaluation.
    :param group_by: The images can be grouped by labels (1 grid is emitted per
        tuple task,label), by tasks (1 grid per task, the images of same labels
        will be on the same raw). If None, only 1 grid is emitted, with on a
        given raw images from the same tuple task,label.
    """

    def __init__(
        self,
        *,
        means: Iterable[float],
        std: Iterable[float],
        n_rows: int = 3,
        n_cols: int = 5,
        mode: Literal["train", "eval"] = "train",
        group_by: GroupByMode = None,
    ):
        n_images_per_label = n_rows * n_cols if group_by == "label" else n_cols
        super().__init__(
            BalancedImagesSamplesMetric(
                n_images_per_label=n_images_per_label,
                means=means,
                std=std,
                group_by=group_by,
            ),
            reset_at="stream",
            emit_at="stream",
            mode=mode,
        )
        self.n_cols = n_cols
        self.group_by = group_by

    def update(self, strategy: "BaseStrategy"):
        self._metric.update(
            strategy.mb_x, strategy.mb_y.tolist(), strategy.mb_task_id.tolist()
        )

    def _package_result(self, strategy: "BaseStrategy") -> "MetricResult":
        metric: BalancedImagesSamplesMetric = self._metric
        return [
            MetricValue(
                self,
                name=f"balanced_images_by_{self.group_by}"
                f"/{self._mode}_phase"
                f"{name}",
                value=TensorImage(
                    make_grid(list(images), normalize=False, nrow=self.n_cols)
                ),
                x_plot=self.get_global_counter(),
            )
            for images, name in metric.results()
        ]


def images_samples_metrics(
    *,
    means: Iterable[float],
    std: Iterable[float],
    n_rows: int = 3,
    n_cols: int = 5,
    on_train: bool = True,
    on_eval: bool = False,
    group_by: GroupByMode = "task",
    do_balanced_sampling: bool = True,
    do_random_sampling: bool = True,
) -> List[PluginMetric]:
    """
    Create the plugins to log some images samples in grids.

    :param means: The means that were used during a normalization step, to
        reverse it before displaying
    :param std: The std that were used during a normalization step, to
        reverse it before displaying
    :param n_rows: The numbers of raws to use in the grid of images. Ignored
        during balanced sampling when the number of raws matches the number of
        labels.
    :param n_cols: The numbers of columns to use in the grid of images.
    :param on_train: If True, will emit some images samples during training.
    :param on_eval: If True, will emit some images samples during evaluation.
    :param group_by: (only in case of balanced sampling, ignored otherwise)
        The images can be grouped by labels (1 grid is emitted per tuple
        task,label), by tasks (1 grid per task, the images of same labels will
        be on the same raw). If None, only 1 grid is emitted, with on a given
        raw images from the same tuple task,label.
    :param do_balanced_sampling: If True, will emit grids with each label being
        represented evenly.
    :param do_random_sampling: If True, will emit grids with images taken at
        random.
    :return: The corresponding plugins.
    """
    modes = []
    if on_eval:
        modes.append("eval")
    if on_train:
        modes.append("train")
    if do_balanced_sampling:
        plugins: List[PluginMetric] = [
            BalancedImagesSamplesPlugin(
                means=means,
                std=std,
                n_rows=n_rows,
                n_cols=n_cols,
                mode=mode,
                group_by=group_by,
            )
            for mode in modes
        ]
    if do_random_sampling:
        plugins.extend(
            [
                RandomImagesSamplesPlugin(
                    means=means,
                    std=std,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    mode=mode,
                )
                for mode in modes
            ]
        )

    return plugins


def make_un_normalize(
    means: Iterable[float], std: Iterable[float]
) -> Normalize:
    """
    Reverse the normalization step.

    :param means: The mean used in the normalization step.
    :param std: The std used in the normalization step.
    """
    return Normalize(
        mean=[-mean / s for mean, s in zip(means, std)],
        std=[1 / s for s in std],
    )


__all__ = [
    images_samples_metrics,
    BalancedImagesSamplesPlugin,
    BalancedImagesSamplesMetric,
    RandomImagesSamplesPlugin,
    ImagesSamplesMetric,
]
