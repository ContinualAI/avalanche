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
Avalanche experiments. """

from pathlib import Path

from PIL.Image import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from avalanche.evaluation.metric_results import AlternativeValues, MetricValue
from avalanche.logging import StrategyLogger


class TensorboardLogger(StrategyLogger):
    """
    The `TensorboardLogger` provides an easy integration with
    Tensorboard logging. Each monitored metric is automatically
    logged to Tensorboard.
    The user can inspect results in real time by appropriately launching
    tensorboard with `tensorboard --logdir=/path/to/tb_log_exp_name`.

    If no parameters are provided, the default folder in which tensorboard
    log files are placed is "./runs/".
    .. note::
        We rely on PyTorch implementation of Tensorboard. If you
        don't have Tensorflow installed in your environment,
        tensorboard will tell you that it is running with reduced
        feature set. This should not impact on the logger performance.
    """

    def __init__(self, tb_log_dir="./tb_data", filename_suffix=''):
        """
        Creates an instance of the `TensorboardLogger`.

        :param tb_log_dir: path to the directory where tensorboard log file
            will be stored. Default to "./tb_data".
        :param filename_suffix: string suffix to append at the end of
            tensorboard log file. Default ''.
        """

        super().__init__()
        tb_log_dir = Path(tb_log_dir)
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(tb_log_dir,
                                    filename_suffix=filename_suffix)

    def log_metric(self, metric_value: MetricValue, callback: str):
        super().log_metric(metric_value, callback)
        name = metric_value.name
        value = metric_value.value

        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(Image, float, int)

        if not isinstance(value, (Image, float, int)):
            # Unsupported type
            return

        if isinstance(value, Image):
            self.writer.add_image(name, to_tensor(value),
                                  global_step=metric_value.x_plot)
        elif isinstance(value, (float, int)):
            self.writer.add_scalar(name, value, global_step=metric_value.x_plot)


__all__ = [
    'TensorboardLogger'
]
