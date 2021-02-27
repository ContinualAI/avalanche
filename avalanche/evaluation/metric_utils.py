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
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor

if TYPE_CHECKING:
    from avalanche.training.plugins import PluggableStrategy
    from avalanche.benchmarks.scenarios import IExperience
    from avalanche.evaluation import PluginMetric


EVAL = "eval"
TRAIN = "train"


def default_cm_image_creator(confusion_matrix_tensor: Tensor,
                             display_labels=None,
                             include_values=True,
                             xticks_rotation='horizontal',
                             values_format=None,
                             cmap='viridis',
                             dpi=100,
                             image_title=''):
    """
    The default Confusion Matrix image creator. This utility uses Scikit-learn
    `ConfusionMatrixDisplay` to create the matplotlib figure. The figure
    is then converted to a PIL `Image`.

    For more info about the accepted graphic parameters, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html#sklearn.metrics.plot_confusion_matrix.

    :param confusion_matrix_tensor: The tensor describing the confusion matrix.
        This can be easily obtained through Scikit-learn `confusion_matrix`
        utility.
    :param display_labels: Target names used for plotting. By default, `labels`
        will be used if it is defined, otherwise the values will be inferred by
        the matrix tensor.
    :param include_values: Includes values in confusion matrix. Defaults to
        `True`.
    :param xticks_rotation: Rotation of xtick labels. Valid values are
        'vertical', 'horizontal' or a float point value. Defaults to
        'horizontal'.
    :param values_format: Format specification for values in confusion matrix.
        Defaults to `None`, which means that the format specification is
        'd' or '.2g', whichever is shorter.
    :param cmap: Must be a str or a Colormap recognized by matplotlib.
        Defaults to 'viridis'.
    :param dpi: The dpi to use to save the image.
    :param image_title: The title of the image. Defaults to an empty string.
    :return: The Confusion Matrix as a PIL Image.
    """

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_tensor.numpy(),
        display_labels=display_labels)
    display.plot(include_values=include_values, cmap=cmap,
                 xticks_rotation=xticks_rotation, values_format=values_format)

    display.ax_.set_title(image_title)

    fig = display.figure_
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    return image


def get_task_label(strategy: 'PluggableStrategy') -> int:
    """
    Returns the current task label.

    The current task label depends on the phase. During the training
    phase, the task label is the one defined in the "train_task_label"
    field. On the contrary, during the eval phase the task label is the one
    defined in the "eval_task_label" field.

    :param strategy: The strategy instance to get the task label from.
    :return: The current train or eval task label.
    """

    if strategy.is_eval:
        return strategy.eval_task_label

    return strategy.train_task_label


def stream_type(experience: 'IExperience') -> str:
    """
    Returns the stream name from which the experience belongs to.
    e.g. the experience can be part of train or test stream.

    :param experience: the instance of the experience
    """

    return experience.origin_stream.name


def phase_and_task(strategy: 'PluggableStrategy') -> Tuple[str, int]:
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
        return EVAL, strategy.eval_task_label

    return TRAIN, strategy.train_task_label


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
                    strategy: 'PluggableStrategy',
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

        experience_label = strategy.eval_exp_id if phase_name == EVAL \
            else strategy.training_exp_counter
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
    'stream_type',
    'bytes2human',
    'get_metric_name']
