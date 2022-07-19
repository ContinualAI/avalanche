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
import matplotlib.pyplot as plt


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
