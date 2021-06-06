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
    def __init__(
        self, *, n_images: int, means: Iterable[float], std: Iterable[float]
    ):
        self.n_images = n_images
        self.un_normalize = make_un_normalize(means, std)
        self.images: List[Tensor] = []
        self.n_seen_images = 0

    def reset(self):
        self.images = []
        self.n_seen_images = 0

    def update(self, images: Tensor):
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
        return self.images


class BalancedImagesSamplesMetric(Metric):
    def __init__(
        self,
        *,
        n_images_per_label: Optional[int],
        means: Iterable[float],
        std: Iterable[float],
        group_by: GroupByMode,
    ):
        self.group_by = group_by
        self.n_images_per_label = n_images_per_label
        self.task2label2images: Dict[int, Dict[int, List[Tensor]]] = {}
        self.un_normalize = make_un_normalize(means, std)

    def reset(self) -> None:
        self.task2label2images = defaultdict(lambda: defaultdict(list))

    def update(self, images: Tensor, labels: List[int], tasks: List[int]):
        for image, label, task in zip(images, labels, tasks):
            label_images = self.task2label2images[task][label]
            if len(label_images) < self.n_images_per_label:
                label_images.append(self.un_normalize(image))

    def results(self) -> Iterable[Tuple[Iterable[Tensor], str]]:
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
    def __init__(
        self,
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

    def before_training_iteration(
        self, strategy: "BaseStrategy"
    ) -> "MetricResult":
        if self._mode == "train":
            return self._update(strategy)

    def before_eval_iteration(self, strategy: "BaseStrategy") -> "MetricResult":
        if self._mode == "eval":
            return self._update(strategy)

    def _update(self, strategy: "BaseStrategy"):
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
    group_by: GroupByMode = "train",
    do_balanced_sampling: bool = True,
    do_random_sampling: bool = True,
) -> List[PluginMetric]:
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
    return Normalize(
        mean=[-mean / s for mean, s in zip(means, std)],
        std=[1 / s for s in std],
    )
