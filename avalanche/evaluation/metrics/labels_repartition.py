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

from avalanche.evaluation import GenericPluginMetric, Metric, PluginMetric
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
    from avalanche.training.templates.supervised import SupervisedTemplate
    from avalanche.evaluation.metric_results import MetricResult


class LabelsRepartition(Metric):
    """
    Metric used to monitor the labels repartition.
    """

    def __init__(self):
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


LabelsRepartitionImageCreator = Callable[
    [Dict[int, List[int]], List[int]], Figure
]


class LabelsRepartitionPlugin(GenericPluginMetric[Figure]):
    """
    A plugin to monitor the labels repartition.

    :param image_creator: The function to use to create an image from the
        history of the labels repartition. It will receive a dictionary of the
        form {label_id: [count_at_step_0, count_at_step_1, ...], ...}
        and the list of the corresponding steps [step_0, step_1, ...].
        If set to None, only the raw data is emitted.
    :param mode: Indicates if this plugin should run on train or eval.
    :param emit_reset_at: The refreshment rate of the plugin.
    :return: The list of corresponding plugins.
    """

    def __init__(
        self,
        *,
        image_creator: Optional[
            LabelsRepartitionImageCreator
        ] = default_history_repartition_image_creator,
        mode: Literal["train", "eval"] = "train",
        emit_reset_at: Literal["stream", "experience", "epoch"] = "epoch",
    ):
        self.labels_repartition = LabelsRepartition()
        super().__init__(
            metric=self.labels_repartition,
            emit_at=emit_reset_at,
            reset_at=emit_reset_at,
            mode=mode,
        )
        self.emit_reset_at = emit_reset_at
        self.mode = mode
        self.image_creator = image_creator
        self.steps = [0]
        self.task2label2counts: Dict[int, Dict[int, List[int]]] = defaultdict(
            dict
        )

    def reset(self) -> None:
        self.steps.append(self.global_it_counter)
        return super().reset()

    def update(self, strategy: "SupervisedTemplate"):
        if strategy.clock.train_exp_epochs and self.emit_reset_at != "epoch":
            return
        self.labels_repartition.update(
            strategy.mb_task_id.tolist(),
            strategy.mb_y.tolist(),
            class_order=getattr(
                strategy.experience.benchmark, "classes_order", None
            ),
        )

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        self.steps.append(self.global_it_counter)
        task2label2count = self.labels_repartition.result()
        for task, label2count in task2label2count.items():
            for label, count in label2count.items():
                self.task2label2counts[task].setdefault(
                    label, [0] * (len(self.steps) - 2)
                ).extend((count, count))
        for task, label2counts in self.task2label2counts.items():
            for label, counts in label2counts.items():
                counts.extend([0] * (len(self.steps) - len(counts)))
        return [
            MetricValue(
                self,
                name=f"Repartition"
                f"/{self._mode}_phase"
                f"/{stream_type(strategy.experience)}_stream"
                f"/Task_{task:03}",
                value=AlternativeValues(
                    self.image_creator(label2counts, self.steps),
                    label2counts,
                )
                if self.image_creator is not None
                else label2counts,
                x_plot=strategy.clock.train_iterations,
            )
            for task, label2counts in self.task2label2counts.items()
        ]

    def __str__(self):
        return "Repartition"


def labels_repartition_metrics(
    *,
    on_train: bool = True,
    emit_train_at: Literal["stream", "experience", "epoch"] = "epoch",
    on_eval: bool = False,
    emit_eval_at: Literal["stream", "experience"] = "stream",
    image_creator: Optional[
        LabelsRepartitionImageCreator
    ] = default_history_repartition_image_creator,
) -> List[PluginMetric]:
    """
    Create plugins to monitor the labels repartition.

    :param on_train: If True, emit the metrics during training.
    :param emit_train_at: (only if on_train is True) when to emit the training
        metrics.
    :param on_eval:  If True, emit the metrics during evaluation.
    :param emit_eval_at: (only if on_eval is True) when to emit the evaluation
        metrics.
    :param image_creator: The function to use to create an image from the
        history of the labels repartition. It will receive a dictionary of the
        form {label_id: [count_at_step_0, count_at_step_1, ...], ...}
        and the list of the corresponding steps [step_0, step_1, ...].
        If set to None, only the raw data is emitted.
    :return: The list of corresponding plugins.
    """
    plugins = []
    if on_eval:
        plugins.append(
            LabelsRepartitionPlugin(
                image_creator=image_creator,
                mode="eval",
                emit_reset_at=emit_eval_at,
            )
        )
    if on_train:
        plugins.append(
            LabelsRepartitionPlugin(
                image_creator=image_creator,
                mode="train",
                emit_reset_at=emit_train_at,
            )
        )

    return plugins


__all__ = [
    "LabelsRepartitionPlugin",
    "LabelsRepartition",
    "labels_repartition_metrics",
]
