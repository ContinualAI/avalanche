#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

import io
from typing import Union, Sequence, List, Type

import matplotlib.pyplot as plt
from PIL import Image
from numpy import ndarray
from sklearn.metrics import ConfusionMatrixDisplay

from .evaluation_data import EvalData, EvalTestData


def default_cm_image_creator(confusion_matrix_tensor: ndarray,
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
        confusion_matrix=confusion_matrix_tensor, display_labels=display_labels)
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


def get_task_label(eval_data: Union[EvalData, EvalTestData]) -> int:
    """
    Returns the current task label for a evaluation data object.

    The current task label depends on the phase. During the training
    phase, the task label is the one defined in the "training_task_label"
    field. On the contrary, during the test phase the task label is the one
    defined in the "test_task_label" field.

    :param eval_data: The evaluation data to get the task label from.
    :return: The current train or test task label.
    """

    if eval_data.test_phase:
        return eval_data.test_task_label

    return eval_data.training_task_label


def filter_accepted_events(
        event_types: Union[Type[EvalData], Sequence[Type[EvalData]]],
        train: bool, test: bool) -> List[Type[EvalData]]:
    """
    A simple utility method used to filter the types of evaluation events.

    :param event_types: A list of event types.
    :param train: If True, train events will be kept.
    :param test: If True, test events will be kept.
    :return: The event types as filtered according to the "train" and "test"
        parameters.
    """
    if isinstance(event_types, type):
        event_types = [event_types]

    accepted = []
    for event_type in event_types:
        if issubclass(event_type, EvalTestData):
            if test:
                accepted.append(event_type)
        elif issubclass(event_type, EvalData) and train:
            accepted.append(event_type)
    return accepted


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


__all__ = [
    'default_cm_image_creator',
    'get_task_label',
    'filter_accepted_events',
    'bytes2human']
