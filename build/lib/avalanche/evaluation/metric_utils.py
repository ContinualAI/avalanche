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

import io
from typing import Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy
    from avalanche.benchmarks.scenarios import Experience
    from avalanche.evaluation import PluginMetric


EVAL = "eval"
TRAIN = "train"


def default_cm_image_creator(confusion_matrix_tensor: Tensor,
                             display_labels=None,
                             include_values=False,
                             xticks_rotation=0,
                             yticks_rotation=0,
                             values_format=None,
                             cmap='viridis',
                             image_title=''):
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
        `True`.
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
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0

        for i in range(n_classes):
            for j in range(n_classes):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], '.2g')
                    if cm.dtype.kind != 'f':
                        text_d = format(cm[i, j], 'd')
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                text_[i, j] = ax.text(
                    j, i, text_cm,
                    ha="center", va="center",
                    color=color)

    if display_labels is None:
        display_labels = np.arange(n_classes)

    fig.colorbar(im_, ax=ax)

    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           ylabel="True label",
           xlabel="Predicted label")

    if image_title != '':
        ax.set_title(image_title)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    plt.setp(ax.get_yticklabels(), rotation=yticks_rotation)

    fig.tight_layout()
    return fig


def get_task_label(strategy: 'BaseStrategy') -> int:
    """
    Returns the current task label.

    The current task label depends on the phase. During the training
    phase, the task label is the one defined in the "train_task_label"
    field. On the contrary, during the eval phase the task label is the one
    defined in the "eval_task_label" field.

    :param strategy: The strategy instance to get the task label from.
    :return: The current train or eval task label.
    """
    return strategy.experience.task_label


def stream_type(experience: 'Experience') -> str:
    """
    Returns the stream name from which the experience belongs to.
    e.g. the experience can be part of train or test stream.

    :param experience: the instance of the experience
    """

    return experience.origin_stream.name


def phase_and_task(strategy: 'BaseStrategy') -> Tuple[str, int]:
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

    if strategy.is_eval:
        return EVAL, strategy.experience.task_label

    return TRAIN, strategy.experience.task_label


def bytes2human(n):
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n


def get_metric_name(metric: 'PluginMetric',
                    strategy: 'BaseStrategy',
                    add_experience=False):
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
    """

    phase_name, task_label = phase_and_task(strategy)
    stream = stream_type(strategy.experience)
    if add_experience:
        experience_label = strategy.experience.current_experience
        metric_name = '{}/{}_phase/{}_stream/Task{:03}/Exp{:03}' \
            .format(str(metric),
                    phase_name,
                    stream,
                    task_label,
                    experience_label)
    else:
        metric_name = '{}/{}_phase/{}_stream/Task{:03}' \
            .format(str(metric),
                    phase_name,
                    stream,
                    task_label)

    return metric_name


__all__ = [
    'default_cm_image_creator',
    'get_task_label',
    'phase_and_task',
    'get_metric_name',
    'stream_type',
    'bytes2human'
]
