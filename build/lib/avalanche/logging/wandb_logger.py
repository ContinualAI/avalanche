################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 25-11-2020                                                             #
# Author(s): Vincenzo Lomonaco, Lorenzo Pellegrini                             #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" This module handles all the functionalities related to the logging of
Avalanche experiments using Weights & Biases. """
import wandb
from PIL.Image import Image
from torch import Tensor
from matplotlib.pyplot import Figure

from avalanche.evaluation.metric_results import AlternativeValues, MetricValue
from avalanche.logging import StrategyLogger
import numpy as np


class WandBLogger(StrategyLogger):
    """
    The `WandBLogger` provides an easy integration with
    Weights & Biases logging. Each monitored metric is automatically
    logged to a dedicated Weights & Biases project dashboard.
    """

    def __init__(self, init_kwargs: dict, interactive: bool = False):
        """
        Creates an instance of the `WandBLogger`.

        :param init_kwargs: All arguments for wandb.init() function call.:
        """

        super().__init__()
        self.import_wandb()
        self.init_kwargs = init_kwargs
        self.interactive = interactive
        self.before_run()

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def before_run(self):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()

    def log_metric(self, metric_value: MetricValue, callback: str):
        super().log_metric(metric_value, callback)
        name = metric_value.name
        value = metric_value.value

        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(Image, Tensor,
                                               Figure, float, int)

        if not isinstance(value, (Image, Tensor, Figure, float, int)):
            # Unsupported type
            return

        if isinstance(value, Image):
            self.wandb.log({name: self.wandb.Image(value)})

        elif isinstance(value, Tensor):
            value = np.histogram(value.view(-1).numpy())
            self.wandb.log({name: wandb.Histogram(np_histogram=value)})

        elif isinstance(value, (float, int, Figure)):
            self.wandb.log({name: value})


__all__ = [
    'WandBLogger'
]
