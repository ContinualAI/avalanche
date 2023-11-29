#!/usr/bin/env python3
import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, nn

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.utils import _at_task_boundary, cycle
from avalanche.models.utils import avalanche_forward


class FeatureDistillationPlugin(SupervisedPlugin):
    def __init__(self, alpha=1):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        """
        super().__init__()
        self.alpha = alpha
        self.prev_model = None

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

        strategy.loss += self.alpha * (
            1 - F.cosine_similarity(new_features, old_features, dim=1).mean()
        )

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        if _at_task_boundary(strategy.experience, before=False):
            strategy.model.features = None
            self.prev_model = copy.deepcopy(strategy.model)


class FeatureExtractorModel(nn.Module):
    """
    Feature extractor that additionnaly stores the features
    """

    def __init__(self, feature_extractor, train_classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.train_classifier = train_classifier
        self.features = None

    def forward(self, x):
        self.features = self.feature_extractor(x)
        x = self.train_classifier(self.features)
        return x


class FeatureDataset(torch.utils.data.Dataset):
    """
    Wrapper around features tensor dataset
    Required for compatibility with storage policy
    """

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class FeatureReplayPlugin(SupervisedPlugin):
    """
    Store some features and use them for replay
    """

    def __init__(self, mem_size: int = 10000, batch_size_mem: int = None):
        super().__init__()
        self.storage_policy = ClassBalancedBuffer(max_size=mem_size)
        self.batch_size_mem = batch_size_mem
        self.criterion = nn.CrossEntropyLoss()
        self.replay_loader = None

    def after_training_exp(self, strategy, **kwargs):
        if not hasattr(strategy.model, "feature_extractor"):
            raise AttributeError(
                "FeatureReplayPlugin expects a FeatureExtractorModel object as the strategy model"
            )
        assert hasattr(strategy.model, "train_classifier")
        feature_dataset = self.gather_feature_dataset(strategy)
        self.storage_policy.update_from_dataset(feature_dataset)

        if self.batch_size_mem is None:
            self.batch_size_mem = strategy.train_mb_size

        # Create replay loader
        self.replay_loader = cycle(
            torch.utils.data.DataLoader(
                self.storage_policy.buffer,
                batch_size=self.batch_size_mem,
                shuffle=True,
                drop_last=True,
            )
        )

    def before_backward(self, strategy, **kwargs):
        if self.replay_loader is None:
            return

        batch_feats, batch_y, batch_t = next(self.replay_loader)
        batch_feats, batch_y = batch_feats.to(strategy.device), batch_y.to(
            strategy.device
        )
        out = strategy.model.train_classifier(batch_feats)

        # Select additional outputs from current
        # output to be learned with cross-entropy
        weight_current = 1 / (strategy.experience.current_experience + 1)

        mb_output = strategy.model.train_classifier(strategy.model.features.detach())

        strategy.loss = 0.5 * strategy.loss + 0.5 * (
            (1 - weight_current) * self.criterion(out, batch_y)
            + weight_current * self.criterion(mb_output, strategy.mb_y)
        )

    @torch.no_grad()
    def gather_feature_dataset(self, strategy):
        strategy.model.eval()
        dataloader = torch.utils.data.DataLoader(
            strategy.experience.dataset, batch_size=strategy.train_mb_size, shuffle=True
        )
        all_features = []
        all_labels = []
        for x, y, t in dataloader:
            x, y = x.to(strategy.device), y.to(strategy.device)
            feats = avalanche_forward(strategy.model.feature_extractor, x, t)
            all_features.append(feats.cpu())
            all_labels.append(y.cpu())
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        features_dataset = FeatureDataset(all_features, all_labels)
        return features_dataset


__all__ = ["FeatureDistillationPlugin", "FeatureReplayPlugin", "FeatureExtractorModel"]
