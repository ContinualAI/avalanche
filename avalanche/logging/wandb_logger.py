################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 25-11-2020                                                             #
# Author(s): Diganta Misra, Andrea Cossu                                       #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################
""" This module handles all the functionalities related to the logging of
Avalanche experiments using Weights & Biases. """
from PIL.Image import Image
from numpy import array
from torch import Tensor
import torch
from matplotlib.pyplot import Figure
from avalanche.evaluation.metric_results import AlternativeValues, \
    MetricValue, TensorImage
from avalanche.logging import StrategyLogger
import numpy as np
import os
import errno


class WandBLogger(StrategyLogger):
    """
    The `WandBLogger` provides an easy integration with
    Weights & Biases logging. Each monitored metric is automatically
    logged to a dedicated Weights & Biases project dashboard.
    """

    def __init__(self, project_name: str = "Avalanche", 
                 run_name: str = "Test", log_artifacts: bool = False,
                 path: str = "Checkpoints", checkpoint: str = "Model.pth", 
                 uri: str = None, params: dict = None):
        """
        Creates an instance of the `WandBLogger`.
        :param project_name: Name of the W&B project.:
        :param run_name: Name of the W&B run.:
        :param log_artifacts: Option to log model weights as W&B Artifacts.:
        :param path: Path to locally save the model checkpoints.:
        :param checkpoint: Name of the model checkpoint file.:
        :param uri: Reference to external URI.:
        :param params: All arguments for wandb.init() function call. 
         Visit https://docs.wandb.ai/ref/python/init to learn about all 
         wand.init() parameters.:
        """
        super().__init__()
        self.import_wandb()
        self.project_name = project_name
        self.run_name = run_name
        self.log_artifacts = log_artifacts
        self.path = path
        self.checkpoint = checkpoint
        self.uri = uri
        self.params = params
        self.args_parse()
        self.before_run()

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def args_parse(self):
        self.init_kwargs = {"project": self.project_name, "name": self.run_name}
        if self.params:
            self.init_kwargs.update(self.params)

    def before_run(self):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()
        self.wandb.run._label(repo="Avalanche")

    def log_metric(self, metric_value: MetricValue, callback: str):
        super().log_metric(metric_value, callback)
        name = metric_value.name
        value = metric_value.value
        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(Image, Tensor, TensorImage,
                                               Figure, float, int,
                                               self.wandb.viz.CustomChart)
        if not isinstance(value, (Image, Tensor, Figure, float, int,
                                  self.wandb.viz.CustomChart)):
            # Unsupported type
            return
        if isinstance(value, Image):
            self.wandb.log({name: self.wandb.Image(value)})
        elif isinstance(value, Tensor):
            value = np.histogram(value.view(-1).numpy())
            self.wandb.log({name: self.wandb.Histogram(np_histogram=value)})
        elif isinstance(value, (float, int, Figure,
                                self.wandb.viz.CustomChart)):
            self.wandb.log({name: value})

        elif isinstance(value, TensorImage):	
            self.wandb.log({name: self.wandb.Image(array(value))})	

    def after_training_exp(self, weights: MetricValue, callback: str):
        super().after_training_exp(weights, callback)
        value = weights.value
        if self.log_artifacts: 
            cwd = os.getcwd()
            ckpt = os.path.join(cwd, self.path)
            try:
                os.makedirs(ckpt)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            dir_name = os.path.join(ckpt, self.checkpoint)
            artifact_name = os.path.join('Models', self.checkpoint)
            if isinstance(value, list):
                torch.save(value, dir_name)
                name = os.path.splittext(self.checkpoint)
                artifact = self.wandb.Artifact(name, type='model')
                artifact.add_file(dir_name, name=artifact_name)
                self.wandb.run.log_artifact(artifact)
                if self.uri is not None:
                    artifact.add_reference(self.uri, name=artifact_name)


__all__ = [
    'WandBLogger'
]
