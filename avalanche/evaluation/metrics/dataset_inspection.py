from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Sequence,
    TYPE_CHECKING,
    Union,
    Optional,
    List,
    Counter,
)

from matplotlib.figure import Figure

from avalanche.evaluation import GenericPluginMetric, Metric
from avalanche.evaluation.metric_results import MetricValue, AlternativeValues
from avalanche.evaluation.metric_utils import (
    stream_type,
    default_history_repartition_image_creator,
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy
    from avalanche.evaluation.metric_results import MetricResult


class LabelsRepartition(Metric):
    def __init__(self):
        super().__init__()
        self.task2label2count: Dict[int, Dict[int, int]] = {}
        self.class_order = None
        self.reset()

    def reset(self) -> None:
        self.task2label2count = defaultdict(Counter)

    def update(
        self,
        tasks: Sequence[int],
        labels: Sequence[Union[str, int]],
        class_order: Optional[List[int]],
    ):
        self.class_order = class_order
        for task, label in zip(tasks, labels):
            self.task2label2count[task][label] += 1

    def update_order(self, class_order: Optional[List[int]]):
        self.class_order = class_order

    def result(self) -> Dict[int, Dict[int, int]]:
        if self.class_order is None:
            return self.task2label2count
        return {
            task: {
                label: label2count[label]
                for label in self.class_order
                if label in label2count
            }
            for task, label2count in self.task2label2count.items()
        }


class LabelsRepartitionPlugin(GenericPluginMetric[Figure]):
    def __init__(
        self,
        image_creator: Callable[
            [Dict[int, List[int]], List[int]], Figure
        ] = default_history_repartition_image_creator,
        mode: Literal["train", "eval"] = "train",
    ):
        super().__init__(
            metric=LabelsRepartition(),
            emit_at="stream",
            reset_at="stream",
            mode=mode,
        )
        self.image_creator = image_creator
        self.steps = [0]
        self.task2labels2counts: Dict[int, Dict[int, List[int]]] = defaultdict(
            dict
        )

    def reset(self) -> None:
        self.steps.append(self.global_it_counter)
        return super().reset()

    def update(self, strategy: "BaseStrategy"):
        self._metric.update(
            strategy.mb_task_id.tolist(),
            strategy.mb_y.tolist(),
            class_order=getattr(
                strategy.experience.scenario, "classes_order", None
            ),
        )

    def _package_result(self, strategy: "BaseStrategy") -> "MetricResult":
        self.steps.append(self.global_it_counter)
        task2label2count: Dict[int, Dict[int, int]] = self._metric.result()
        for task, label2count in task2label2count.items():
            for label, count in label2count.items():
                self.task2labels2counts[task].setdefault(
                    label, [0] * (len(self.steps) - 2)
                ).extend((count, count))
        return [
            MetricValue(
                self,
                name=f"Repartition"
                f"/{self._mode}_phase"
                f"/{stream_type(strategy.experience)}_stream"
                f"/Task_{task:03}",
                value=AlternativeValues(
                    self.image_creator(label2counts, self.steps), label2counts,
                ),
                x_plot=self.get_global_counter(),
            )
            for task, label2counts in self.task2labels2counts.items()
        ]

    def __str__(self):
        return "Repartition"


def labels_repartition_metrics(
    *,
    on_train: bool = True,
    on_eval: bool = False,
    image_creator=default_history_repartition_image_creator,
):
    modes = []
    if on_eval:
        modes.append("eval")
    if on_train:
        modes.append("train")

    return [
        LabelsRepartitionPlugin(image_creator=image_creator, mode=mode)
        for mode in modes
    ]
