################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24/07/2021                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import numpy as np
from matplotlib import pyplot as plt


def learning_curves_plot(all_metrics: dict):
    """Creates a plot with separate learning curves for each experience.

    :param all_metrics: Dictionary of metrics as returned by
        EvaluationPlugin.get_all_metrics
    :return: matplotlib figure
    """
    accs_keys = list(filter(lambda x: "Top1_Acc_Exp" in x, all_metrics.keys()))
    fig, ax = plt.subplots()
    for ak in accs_keys:
        k = ak.split("/")[-1]
        x, y = all_metrics[ak]
        plt.plot(x, y, label=k)
    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Experience Accuracy")
    return fig


def plot_metric_matrix(metric_matrix, title, *, ax=None, text_values=True):
    """Plot a matrix of metrics (e.g. forgetting over time).

    :param metric_matrix: 2D accuracy matrix with shape <time, experiences>
    :param title: plot title
    :param ax: axes to use for figure
    :param text_values: (bool) whether to add the value as text in each cell.

    :return: a matplotlib.Figure
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    metric_matrix = np.array(metric_matrix).T
    ax.matshow(metric_matrix)
    ax.set_ylabel("Experience")
    ax.set_xlabel("Time")
    ax.set_title(title)

    if text_values:
        for i in range(len(metric_matrix)):
            for j in range(len(metric_matrix[0])):
                ax.text(
                    j,
                    i,
                    f"{metric_matrix[i][j]:.3f}",
                    ha="center",
                    va="center",
                    color="w",
                )
    return fig
