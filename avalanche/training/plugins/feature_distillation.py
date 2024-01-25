#!/usr/bin/env python3
import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, nn

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.utils import _at_task_boundary, cycle


class FeatureDistillationPlugin(SupervisedPlugin):
    def __init__(self, alpha=1, mode="cosine"):
        """
        Adds a Distillation loss term on the features of the model,
        trying to maximize the cosine similarity between current and old features

        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        """
        super().__init__()
        self.alpha = alpha
        self.prev_model = None
        assert mode in ["mse", "cosine"]
        self.mode = mode

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        if self.prev_model is None:
            return

        with torch.no_grad():
            avalanche_forward(self.prev_model, strategy.mb_x, strategy.mb_task_id)
            old_features = self.prev_model.features

        new_features = strategy.model.features

        if self.mode == "cosine":
            strategy.loss += self.alpha * (
                1 - F.cosine_similarity(new_features, old_features, dim=1).mean()
            )
        elif self.mode == "mse":
            strategy.loss += self.alpha * F.mse_loss(new_features, old_features, dim=1)

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        if _at_task_boundary(strategy.experience, before=False):
            strategy.model.features = None
            self.prev_model = copy.deepcopy(strategy.model)


__all__ = ["FeatureDistillationPlugin"]
