################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import (
    Dict,
    Optional,
    Union,
    Iterable,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    List,
    Callable,
    Any,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray, arange
from torch import Tensor

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
    from avalanche.benchmarks.scenarios import ClassificationExperience
    from avalanche.evaluation import PluginMetric


EVAL = "eval"
TRAIN = "train"


def default_cm_image_creator(
    confusion_matrix_tensor: Tensor,
    display_labels: Optional[Iterable[Any]] = None,
    include_values=False,
    xticks_rotation=0,
    yticks_rotation=0,
    values_format=None,
    cmap="viridis",
    image_title="",
):
    """
    The default Confusion Matrix image creator.
    Code adapted from
    `Scikit learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html>`_ # noqa

    :param confusion_matrix_tensor: The tensor describing the confusion matrix.
        This can be easily obtained through Scikit-learn `confusion_matrix`
        utility.
    :param display_labels: Target names used for plotting. By default, `labels`
        will be used if it is defined, otherwise the values will be inferred by
        the matrix tensor.
    :param include_values: Includes values in confusion matrix. Defaults to
        `False`.
    :param xticks_rotation: Rotation of xtick labels. Valid values are
        float point value. Defaults to 0.
    :param yticks_rotation: Rotation of ytick labels. Valid values are
        float point value. Defaults to 0.
    :param values_format: Format specification for values in confusion matrix.
        Defaults to `None`, which means that the format specification is
        'd' or '.2g', whichever is shorter.
    :param cmap: Must be a str or a Colormap recognized by matplotlib.
        Defaults to 'viridis'.
    :param image_title: The title of the image. Defaults to an empty string.
    :return: The Confusion Matrix as a PIL Image.
    """

    fig, ax = plt.subplots()

    cm = confusion_matrix_tensor.numpy()
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0

        for i in range(n_classes):
            for j in range(n_classes):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], ".2g")
                    if cm.dtype.kind != "f":
                        text_d = format(cm[i, j], "d")
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                text_[i, j] = ax.text(
                    j, i, text_cm, ha="center", va="center", color=color
                )

    if display_labels is None:
        display_labels = np.arange(n_classes)

    fig.colorbar(im_, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    if image_title != "":
        ax.set_title(image_title)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    plt.setp(ax.get_yticklabels(), rotation=yticks_rotation)

    fig.tight_layout()
    return fig


SEABORN_COLORS = (
    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
    (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
    (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
    (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
    (0.8, 0.7254901960784313, 0.4549019607843137),
    (0.39215686274509803, 0.7098039215686275, 0.803921568627451),
)


def repartition_pie_chart_image_creator(
    label2counts: Dict[int, List[int]],
    counters: List[int],
    colors: Union[ndarray, Iterable, int, float] = SEABORN_COLORS,
    fmt: str = "%1.1f%%",
):
    """
    Create a pie chart representing the labels repartition.

    :param label2counts: A dict holding the counts for each label, of the form
        {label: [count_at_step_0, count_at_step_1, ...]}. Only the last count of
        each label is used here.
    :param counters: (unused) The steps the counts were taken at.
    :param colors: The colors to use in the chart.
    :param fmt: Formatting used to display the text values in the chart.
    """
    ax: Axes
    fig, ax = plt.subplots()

    labels, counts = zip(*((label, c[-1]) for label, c in label2counts.items()))

    ax.pie(counts, labels=labels, autopct=fmt, colors=colors)

    fig.tight_layout()
    return fig


def repartition_bar_chart_image_creator(
    label2counts: Dict[int, List[int]],
    counters: List[int],
    colors: Union[ndarray, Iterable, int, float] = SEABORN_COLORS,
):
    """
    Create a bar chart representing the labels repartition.

    :param label2counts: A dict holding the counts for each label, of the form
        {label: [count_at_step_0, count_at_step_1, ...]}. Only the last count of
        each label is used here.
    :param counters: (unused) The steps the counts were taken at.
    :param colors: The colors to use in the chart.
    """
    ax: Axes
    fig, ax = plt.subplots()

    y = -arange(len(label2counts))
    labels, counts = zip(*((label, c[-1]) for label, c in label2counts.items()))
    total = sum(counts)

    ax.barh(y, width=counts, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Number of exemplars")
    ax.set_ylabel("Class")

    for i, count in enumerate(counts):
        ax.text(count / 2, -i, f"{count/total:.1%}", va="center", ha="center")

    fig.tight_layout()
    return fig


def default_history_repartition_image_creator(
    label2counts: Dict[int, List[int]],
    counters: List[int],
    colors: Union[ndarray, Iterable, int, float] = SEABORN_COLORS,
):
    """
    Create a stack plot representing the labels repartition with their history.

    :param label2counts: A dict holding the counts for each label, of the form
        {label: [count_at_step_0, count_at_step_1, ...]}.
    :param counters: The steps the counts were taken at.
    :param colors: The colors to use in the chart.
    """
    ax: Axes
    fig, ax = plt.subplots()

    ax.stackplot(
        counters,
        label2counts.values(),
        labels=label2counts.keys(),
        colors=colors,
    )
    ax.legend(loc="upper left")
    ax.set_ylabel("Number of examples")
    ax.set_xlabel("step")

    fig.tight_layout()
    return fig


def stream_type(experience: "ClassificationExperience") -> str:
    """
    Returns the stream name from which the experience belongs to.
    e.g. the experience can be part of train or test stream.

    :param experience: the instance of the experience
    """

    return experience.origin_stream.name


def phase_and_task(strategy: "SupervisedTemplate") -> Tuple[str, int]:
    """
    Returns the current phase name and the associated task label.

    The current task label depends on the phase. During the training
    phase, the task label is the one defined in the "train_task_label"
    field. On the contrary, during the eval phase the task label is the one
    defined in the "eval_task_label" field.

    :param strategy: The strategy instance to get the task label from.
    :return: The current phase name as either "Train" or "Task" and the
        associated task label.
    """
    task_labels = getattr(strategy.experience, "task_labels", None)
    if task_labels is not None:
        task = task_labels
        if len(task) > 1:
            task = None  # task labels per patterns
        else:
            task = task[0]
    else:
        task = None

    if strategy.is_eval:
        return EVAL, task
    else:
        return TRAIN, task


def bytes2human(n):
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ("K", "M", "G", "T", "P", "E", "Z", "Y")
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return "%.1f%s" % (value, s)
    return "%sB" % n


def default_metric_name_template(metric_info: Dict[str, Any]):
    add_task = metric_info.get("task_label", None) is not None
    add_phase = metric_info.get("phase_name", None) is not None
    add_experience = metric_info.get("experience_id", None) is not None
    add_stream = metric_info.get("stream_name", None) is not None

    if "metric_name" not in metric_info:
        raise RuntimeError("Key metric_name missing from value dictionary.")

    result_template = "{metric_name}/"
    if add_phase:
        result_template += "{phase_name}_phase/"

    if add_stream:
        result_template += "{stream_name}_stream/"

    if add_task:
        result_template += "Task{task_label:03}/"

    if add_experience:
        result_template += "Exp{experience_id:03}/"

    return result_template[:-1]


def generic_get_metric_name(
    value_name_template: Union[str, Callable[[Dict[str, Any]], str]],
    metric_info: Dict[str, Any],
):
    if isinstance(value_name_template, str):
        name_template = value_name_template
    else:
        name_template = value_name_template(metric_info)

    # https://stackoverflow.com/a/17895844
    return name_template.format(**metric_info)


def get_metric_name(
    metric: Union["PluginMetric", str],
    strategy: "SupervisedTemplate",
    add_experience=False,
    add_task=True,
):
    """
    Return the complete metric name used to report its current value.
    The name is composed by:
    metric string representation /phase type/stream type/task id
    where metric string representation is a synthetic string
    describing the metric, phase type describe if the user
    is training (train) or evaluating (eval), stream type describes
    the type of stream the current experience belongs to (e.g. train, test)
    and task id is the current task label.

    :param metric: the metric object for which return the complete name
    :param strategy: the current strategy object
    :param add_experience: if True, add eval_exp_id to the main metric name.
            Default to False.
    :param add_task: if True the main metric name will include the task
        information. If False, no task label will be displayed.
        If an int, that value will be used as task label for the metric name.
    """
    task_label: Optional[int]
    phase_name, task_label = phase_and_task(strategy)
    assert strategy.experience is not None
    stream = stream_type(strategy.experience)
    experience_id = strategy.experience.current_experience
    if type(add_task) == bool and add_task is False:
        task_label = None
    elif type(add_task) == int:
        task_label = add_task

    if not add_experience:
        experience_id = None

    return generic_get_metric_name(
        default_metric_name_template,
        {
            "metric_name": str(metric),
            "task_label": task_label,
            "phase_name": phase_name,
            "experience_id": experience_id,
            "stream_name": stream,
        },
    )


__all__ = [
    "default_cm_image_creator",
    "phase_and_task",
    "default_metric_name_template",
    "generic_get_metric_name",
    "get_metric_name",
    "stream_type",
    "bytes2human",
    "default_history_repartition_image_creator",
    "repartition_pie_chart_image_creator",
    "repartition_bar_chart_image_creator",
]
