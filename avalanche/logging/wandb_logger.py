################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 25-11-2020                                                             #
# Author(s): Diganta Misra, Andrea Cossu, Lorenzo Pellegrini                   #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################
""" This module handles all the functionalities related to the logging of
Avalanche experiments using Weights & Biases. """

import re
from typing import Optional, Union, List, TYPE_CHECKING
from pathlib import Path
import os
import warnings

import numpy as np
from numpy import array
from torch import Tensor

from PIL.Image import Image
from matplotlib.pyplot import Figure

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metric_results import (
    AlternativeValues,
    MetricValue,
    TensorImage,
)
from avalanche.logging import BaseLogger

if TYPE_CHECKING:
    from avalanche.evaluation.metric_results import MetricValue
    from avalanche.training.templates import SupervisedTemplate


CHECKPOINT_METRIC_NAME = re.compile(
    r"^WeightCheckpoint\/(?P<phase_name>\S+)_phase\/(?P<stream_name>\S+)_"
    r"stream(\/Task(?P<task_id>\d+))?\/Exp(?P<experience_id>\d+)$"
)


class WandBLogger(BaseLogger, SupervisedPlugin):
    """Weights and Biases logger.

    The `WandBLogger` provides an easy integration with
    Weights & Biases logging. Each monitored metric is automatically
    logged to a dedicated Weights & Biases project dashboard.

    External storage for W&B Artifacts (for instance - AWS S3 and GCS
    buckets) uri are supported.

    The wandb log files are placed by default in "./wandb/" unless specified.

    .. note::

        TensorBoard can be synced on to the W&B dedicated dashboard.
    """

    def __init__(
        self,
        project_name: str = "Avalanche",
        run_name: str = "Test",
        log_artifacts: bool = False,
        path: Union[str, Path] = "Checkpoints",
        uri: Optional[str] = None,
        sync_tfboard: bool = False,
        save_code: bool = True,
        config: Optional[object] = None,
        dir: Optional[Union[str, Path]] = None,
        params: Optional[dict] = None,
    ):
        """Creates an instance of the `WandBLogger`.

        :param project_name: Name of the W&B project.
        :param run_name: Name of the W&B run.
        :param log_artifacts: Option to log model weights as W&B Artifacts.
            Note that, in order for model weights to be logged, the
            :class:`WeightCheckpoint` metric must be added to the
            evaluation plugin.
        :param path: Path to locally save the model checkpoints.
        :param uri: URI identifier for external storage buckets (GCS, S3).
        :param sync_tfboard: Syncs TensorBoard to the W&B dashboard UI.
        :param save_code: Saves the main training script to W&B.
        :param config: Syncs hyper-parameters and config values used to W&B.
        :param dir: Path to the local log directory for W&B logs to be saved at.
        :param params: All arguments for wandb.init() function call. Visit
            https://docs.wandb.ai/ref/python/init to learn about all
            wand.init() parameters.
        """
        super().__init__()
        self.import_wandb()
        self.project_name = project_name
        self.run_name = run_name
        self.log_artifacts = log_artifacts
        self.path = path
        self.uri = uri
        self.sync_tfboard = sync_tfboard
        self.save_code = save_code
        self.config = config
        self.dir = dir
        self.params = params
        self.args_parse()
        self.before_run()
        self.step = 0
        self.exp_count = 0

    def import_wandb(self):
        try:
            import wandb

            assert hasattr(wandb, "__version__")
        except ImportError:
            raise ImportError('Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def args_parse(self):
        self.init_kwargs = {
            "project": self.project_name,
            "name": self.run_name,
            "sync_tensorboard": self.sync_tfboard,
            "dir": self.dir,
            "save_code": self.save_code,
            "config": self.config,
        }
        if self.params:
            self.init_kwargs.update(self.params)

    def before_run(self):
        if self.wandb is None:
            self.import_wandb()

        if self.init_kwargs is None:
            self.init_kwargs = dict()

        run_id = self.init_kwargs.get("id", None)
        if run_id is None:
            run_id = os.environ.get("WANDB_RUN_ID", None)
        if run_id is None:
            run_id = self.wandb.util.generate_id()

        self.init_kwargs["id"] = run_id

        self.wandb.init(**self.init_kwargs)
        self.wandb.run._label(repo="Avalanche")

    def after_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        for val in metric_values:
            self.log_metrics([val])

        self.wandb.log({"TrainingExperience": self.exp_count}, step=self.step)
        self.exp_count += 1

    def log_single_metric(self, name, value, x_plot):
        self.step = x_plot

        if name.startswith("WeightCheckpoint"):
            if self.log_artifacts:
                self._log_checkpoint(name, value, x_plot)
            return

        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(
                Image,
                Tensor,
                TensorImage,
                Figure,
                float,
                int,
                self.wandb.viz.CustomChart,
            )

        if not isinstance(
            value,
            (
                Image,
                TensorImage,
                Tensor,
                Figure,
                float,
                int,
                self.wandb.viz.CustomChart,
            ),
        ):
            # Unsupported type
            return

        if isinstance(value, Image):
            self.wandb.log({name: self.wandb.Image(value)}, step=self.step)

        elif isinstance(value, Tensor):
            value = np.histogram(value.view(-1).numpy())
            self.wandb.log(
                {name: self.wandb.Histogram(np_histogram=value)}, step=self.step
            )

        elif isinstance(value, (float, int, Figure, self.wandb.viz.CustomChart)):
            self.wandb.log({name: value}, step=self.step)

        elif isinstance(value, TensorImage):
            self.wandb.log({name: self.wandb.Image(array(value))}, step=self.step)

    def _log_checkpoint(self, name, value, x_plot):
        assert self.wandb is not None

        # Example: 'WeightCheckpoint/train_phase/train_stream/Task000/Exp000'
        name_match = CHECKPOINT_METRIC_NAME.match(name)
        if name_match is None:
            warnings.warn(f"Checkpoint metric has unsupported name {name}.")
            return
        # phase_name: str = name_match['phase_name']
        # stream_name: str = name_match['stream_name']
        task_id: Optional[int] = (
            int(name_match["task_id"]) if name_match["task_id"] is not None else None
        )
        experience_id: int = int(name_match["experience_id"])
        assert experience_id >= 0

        cwd = Path.cwd()
        checkpoint_directory = cwd / self.path
        checkpoint_directory.mkdir(parents=True, exist_ok=True)

        checkpoint_name = "Model_{}".format(experience_id)
        checkpoint_file_name = checkpoint_name + ".pth"
        checkpoint_path = checkpoint_directory / checkpoint_file_name
        artifact_name = "Models/" + checkpoint_file_name

        # Write the checkpoint blob
        with open(checkpoint_path, "wb") as f:
            f.write(value)

        metadata = {
            "experience": experience_id,
            "x_step": x_plot,
            **({"task_id": task_id} if task_id is not None else {}),
        }

        artifact = self.wandb.Artifact(checkpoint_name, type="model", metadata=metadata)
        artifact.add_file(str(checkpoint_path), name=artifact_name)
        self.wandb.run.log_artifact(artifact)
        if self.uri is not None:
            artifact.add_reference(self.uri, name=artifact_name)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "wandb" in state:
            del state["wandb"]
        return state

    def __setstate__(self, state):
        print("[W&B logger] Resuming from checkpoint...")
        self.__dict__ = state
        if self.init_kwargs is None:
            self.init_kwargs = dict()
        self.init_kwargs["resume"] = "allow"

        self.wandb = None
        self.before_run()


__all__ = ["WandBLogger"]
