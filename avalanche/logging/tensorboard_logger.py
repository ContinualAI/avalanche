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
from typing import Union

from PIL.Image import Image
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from matplotlib.pyplot import Figure
from torchvision.transforms.functional import to_tensor
from avalanche.evaluation.metric_results import AlternativeValues, TensorImage
from avalanche.logging import BaseLogger
import weakref


class TensorboardLogger(BaseLogger):
    """Tensorboard logger.

    The `TensorboardLogger` provides an easy integration with
    Tensorboard logging. Each monitored metric is automatically
    logged to Tensorboard.
    The user can inspect results in real time by appropriately launching
    tensorboard with `tensorboard --logdir=/path/to/tb_log_exp_name`.

    AWS's S3 buckets and (if tensorflow is installed) GCloud storage url are
    supported.

    If no parameters are provided, the default folder in which tensorboard
    log files are placed is "./runs/".

    .. note::

        We rely on PyTorch implementation of Tensorboard. If you
        don't have Tensorflow installed in your environment,
        tensorboard will tell you that it is running with reduced
        feature set. This should not impact on the logger performance.
    """

    def __init__(
        self,
        tb_log_dir: Union[str, Path] = "./tb_data",
        filename_suffix: str = "",
    ):
        """Creates an instance of the `TensorboardLogger`.

        :param tb_log_dir: path to the directory where tensorboard log file
            will be stored. Default to "./tb_data".
        :param filename_suffix: string suffix to append at the end of
            tensorboard log file. Default ''.
        """

        super().__init__()
        tb_log_dir = _make_path_if_local(tb_log_dir)
        self.writer = self._make_writer(tb_log_dir, filename_suffix)

    def log_single_metric(self, name, value, x_plot):
        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(
                Image, Tensor, TensorImage, Figure, float, int
            )

        if isinstance(value, Figure):
            self.writer.add_figure(name, value, global_step=x_plot)

        elif isinstance(value, Image):
            self.writer.add_image(name, to_tensor(value), global_step=x_plot)

        elif isinstance(value, Tensor):
            self.writer.add_histogram(name, value, global_step=x_plot)

        elif isinstance(value, (float, int)):
            self.writer.add_scalar(name, value, global_step=x_plot)

        elif isinstance(value, TensorImage):
            self.writer.add_image(name, value.image, global_step=x_plot)

    def _make_writer(self, tb_log_dir, filename_suffix):
        writer = SummaryWriter(tb_log_dir, filename_suffix=filename_suffix)

        # Shuts down the writer gracefully on process exit
        # or when this logger gets GCed. Fixes issue #864.
        # For more info see:
        # https://docs.python.org/3/library/weakref.html#comparing-finalizers-with-del-methods
        weakref.finalize(self, SummaryWriter.close, writer)
        return writer

    def __getstate__(self):
        return self.writer.log_dir, self.writer.filename_suffix

    def __setstate__(self, state):
        self.writer = self._make_writer(*state)


def _make_path_if_local(tb_log_dir: Union[str, Path]) -> Union[str, Path]:
    if isinstance(tb_log_dir, str) and _is_aws_or_gcloud_path(tb_log_dir):
        return tb_log_dir

    tb_log_dir = Path(tb_log_dir)
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    return tb_log_dir


def _is_aws_or_gcloud_path(tb_log_dir: str) -> bool:
    return tb_log_dir.startswith("gs://") or tb_log_dir.startswith("s3://")


__all__ = ["TensorboardLogger"]
