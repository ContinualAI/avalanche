"""
This metric was described in the IL2M paper:

E. Belouadah and A. Popescu,
"IL2M: Class Incremental Learning With Dual Memory,"
2019 IEEE/CVF International Conference on Computer Vision (ICCV),
2019, pp. 583-592, doi: 10.1109/ICCV.2019.00067.

It selects the scores of the true class and then average them for past and new
classes.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, Set, TYPE_CHECKING, List, Optional, TypeVar, Literal

import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from torch import Tensor, arange

from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_utils import get_metric_name

from avalanche.evaluation.metrics import Mean
from avalanche.evaluation.metric_results import MetricValue, AlternativeValues


if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
    from avalanche.evaluation.metric_results import MetricResult


TAggregation = TypeVar("TAggregation")
LabelCat = Literal["new", "old"]


class MeanScores(Metric[Dict[TAggregation, float]], ABC):
    """
    Average the scores of the true class by label
    """

    def __init__(self):
        self.label2mean: Dict[int, Mean] = defaultdict(Mean)
        self.reset()

    def reset(self) -> None:
        self.label2mean = defaultdict(Mean)

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor):
        assert (
            len(predicted_y.size()) == 2
        ), "Predictions need to be logits or scores, not labels"

        if len(true_y.size()) == 2:
            true_y = true_y.argmax(axis=1)

        scores = predicted_y[arange(len(true_y)), true_y]

        for score, label in zip(scores.tolist(), true_y.tolist()):
            self.label2mean[label].update(score)

    @abstractmethod
    def result(self) -> Dict[TAggregation, float]:
        pass


class PerClassMeanScores(MeanScores[int]):
    def result(self) -> Dict[int, float]:
        return {label: m.result() for label, m in self.label2mean.items()}


class MeanNewOldScores(MeanScores[LabelCat]):
    """
    Average the scores of the true class by old and new classes
    """

    def __init__(self):
        super().__init__()
        self.new_classes: Set[int] = set()

    def reset(self) -> None:
        super().reset()
        self.new_classes = set()

    def update_new_classes(self, new_classes: Set[int]):
        self.new_classes.update(new_classes)

    @property
    def old_classes(self) -> Set[int]:
        return set(self.label2mean) - self.new_classes

    def result(self) -> Dict[LabelCat, float]:
        # print(self.new_classes, self.label2mean)
        rv: Dict[LabelCat, float] = {
            "new": sum(
                (self.label2mean[label] for label in self.new_classes),
                start=Mean(),
            ).result()
        }
        if not self.old_classes:
            return rv

        rv["old"] = sum(
            (self.label2mean[label] for label in self.old_classes),
            start=Mean(),
        ).result()

        return rv


def default_mean_scores_image_creator(
    label2step2mean_scores: Dict[LabelCat, Dict[int, float]]
) -> Figure:
    """
    Default function to create an image of the evolution of the scores of the
        true class, averaged by new and old classes.

    :param label2step2mean_scores: A dictionary that, for each label category
        ("old" and "new") contains a dictionary of mean scores indexed by the
        step of the observation.
    :return: The figure containing the graphs.
    """
    ax: Axes
    fig, ax = subplots()

    markers = "*o"

    for marker, (label, step2mean_scores) in zip(
        markers, label2step2mean_scores.items()
    ):
        ax.plot(
            step2mean_scores.keys(),
            step2mean_scores.values(),
            marker,
            label=label,
        )

    ax.legend(loc="lower left")
    ax.set_xlabel("step")
    ax.set_ylabel("mean score")

    fig.tight_layout()
    return fig


MeanScoresImageCreator = Callable[[Dict[LabelCat, Dict[int, float]]], Figure]


class MeanScoresPluginMetricABC(PluginMetric, ABC):
    """
    Base class for the plugins that show the scores of the true class, averaged
        by new and old classes.

    :param image_creator: The function to use to create an image of the history
        of the mean scores grouped by old and new classes
    """

    def __init__(
        self,
        image_creator: Optional[
            MeanScoresImageCreator
        ] = default_mean_scores_image_creator,
    ):
        super().__init__()
        self.mean_scores = MeanNewOldScores()
        self.image_creator = image_creator
        self.label_cat2step2mean: Dict[LabelCat, Dict[int, float]] = defaultdict(dict)

    def reset(self) -> None:
        self.mean_scores.reset()

    def update_new_classes(self, strategy: "SupervisedTemplate"):
        assert strategy.experience is not None
        self.mean_scores.update_new_classes(
            strategy.experience.classes_in_this_experience
        )

    def update(self, strategy: "SupervisedTemplate"):
        self.mean_scores.update(predicted_y=strategy.mb_output, true_y=strategy.mb_y)

    def result(self) -> Dict[LabelCat, float]:
        return self.mean_scores.result()

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        label_cat2mean_score: Dict[LabelCat, float] = self.result()
        num_it = strategy.clock.train_iterations

        for label_cat, m in label_cat2mean_score.items():
            self.label_cat2step2mean[label_cat][num_it] = m

        base_metric_name = get_metric_name(
            self, strategy, add_experience=False, add_task=False
        )

        rv = [
            MetricValue(
                self,
                name=base_metric_name + f"/{label_cat}_classes",
                value=m,
                x_plot=num_it,
            )
            for label_cat, m in label_cat2mean_score.items()
        ]
        if "old" in label_cat2mean_score and "new" in label_cat2mean_score:
            rv.append(
                MetricValue(
                    self,
                    name=base_metric_name + f"/new_old_diff",
                    value=label_cat2mean_score["new"] - label_cat2mean_score["old"],
                    x_plot=num_it,
                )
            )
        if self.image_creator is not None:
            rv.append(
                MetricValue(
                    self,
                    name=base_metric_name,
                    value=AlternativeValues(
                        self.image_creator(self.label_cat2step2mean),
                        self.label_cat2step2mean,
                    ),
                    x_plot=num_it,
                )
            )

        return rv

    def __str__(self):
        return "MeanScores"


class MeanScoresTrainPluginMetric(MeanScoresPluginMetricABC):
    """
    Plugin to show the scores of the true class during the lasts training
        epochs of each experience, averaged  by new and old classes.
    """

    def before_training_epoch(self, strategy: "SupervisedTemplate") -> None:
        self.reset()
        self.update_new_classes(strategy)

    def after_training_iteration(self, strategy: "SupervisedTemplate") -> None:
        if strategy.clock.train_exp_epochs == strategy.train_epochs - 1:
            self.update(strategy)
        super().after_training_iteration(strategy)

    def after_training_epoch(self, strategy: "SupervisedTemplate") -> "MetricResult":
        if strategy.clock.train_exp_epochs == strategy.train_epochs - 1:
            return self._package_result(strategy)
        else:
            return None


class MeanScoresEvalPluginMetric(MeanScoresPluginMetricABC):
    """
    Plugin to show the scores of the true class during evaluation, averaged by
        new and old classes.
    """

    def before_training(self, strategy: "SupervisedTemplate") -> None:
        self.reset()

    def before_training_exp(self, strategy: "SupervisedTemplate") -> None:
        self.update_new_classes(strategy)

    def after_eval_iteration(self, strategy: "SupervisedTemplate") -> None:
        self.update(strategy)
        super().after_eval_iteration(strategy)

    def after_eval(self, strategy: "SupervisedTemplate") -> "MetricResult":
        return self._package_result(strategy)


def mean_scores_metrics(
    *,
    on_train: bool = True,
    on_eval: bool = True,
    image_creator: Optional[MeanScoresImageCreator] = default_mean_scores_image_creator,
) -> List[PluginMetric]:
    """
    Helper to create plugins to show the scores of the true class, averaged by
        new and old classes. The plugins are available during training (for the
        last epoch of each experience) and evaluation.

    :param on_train: If True the train plugin is created
    :param on_eval: If True the eval plugin is created
    :param image_creator: The function to use to create an image of the history
        of the mean scores grouped by old and new classes
    :return: The list of plugins that were specified
    """
    plugins: List[PluginMetric] = []

    if on_eval:
        plugins.append(MeanScoresEvalPluginMetric(image_creator=image_creator))
    if on_train:
        plugins.append(MeanScoresTrainPluginMetric(image_creator=image_creator))

    return plugins


__all__ = [
    "mean_scores_metrics",
    "MeanScoresTrainPluginMetric",
    "MeanScoresEvalPluginMetric",
    "MeanScores",
    "MeanNewOldScores",
    "PerClassMeanScores",
]
