#!/usr/bin/env python3
import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avalanche.benchmarks.utils import concat_datasets
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.storage_policy import ClassBalancedBuffer

from avalanche.models.fecam import compute_means, compute_covariance

class CurrentDataFeCAMUpdate(SupervisedPlugin):
    """
    Updates FeCAM cov and prototypes
    using the current task data
    (at the end of each task)
    """

    def __init__(self):
        super().__init__()

    def after_training_exp(self, strategy, **kwargs):
        assert hasattr(strategy.model, "eval_classifier")
        assert isinstance(strategy.model.eval_classifier, FeCAMClassifier)

        num_workers = kwargs["num_workers"] if "num_workers" in kwargs else 0
        loader = torch.utils.data.DataLoader(
            strategy.adapted_dataset.eval(),
            batch_size=strategy.train_mb_size,
            shuffle=False,
            num_workers=num_workers,
        )

        features = []
        labels = []

        was_training = strategy.model.training
        strategy.model.eval()

        for x, y, t in loader:
            x = x.to(strategy.device)
            y = y.to(strategy.device)

            with torch.no_grad():
                out = strategy.model.feature_extractor(x)

            features.append(out)
            labels.append(y)

        if was_training:
            strategy.model.train()

        features = torch.cat(features)
        labels = torch.cat(labels)

        # Transform
        features = strategy.model.eval_classifier.apply_transforms(features)
        class_means = compute_means(features, labels)
        class_cov = compute_covariance(features, labels)
        class_cov = strategy.model.eval_classifier.apply_cov_transforms(class_cov)

        strategy.model.eval_classifier.update_class_means_dict(class_means)
        strategy.model.eval_classifier.update_class_cov_dict(class_cov)

class MemoryFeCAMUpdate(SupervisedPlugin):
    """
    Updates FeCAM cov and prototypes
    using the current task data
    (at the end of each task)
    """

    def __init__(self, mem_size=2000, storage_policy=None):
        super().__init__()
        if storage_policy is None:
            self.storage_policy = ClassBalancedBuffer(max_size=mem_size)
        else:
            self.storage_policy = storage_policy

    def after_training_exp(self, strategy, **kwargs):
        self.storage_policy.update(strategy)

        num_workers = kwargs["num_workers"] if "num_workers" in kwargs else 0
        loader = torch.utils.data.DataLoader(
            self.storage_policy.buffer.eval(),
            batch_size=strategy.train_mb_size,
            shuffle=False,
            num_workers=num_workers,
        )

        features = []
        labels = []

        was_training = strategy.model.training
        strategy.model.eval()

        for x, y, t in loader:
            x = x.to(strategy.device)
            y = y.to(strategy.device)

            with torch.no_grad():
                out = strategy.model.feature_extractor(x)

            features.append(out)
            labels.append(y)

        if was_training:
            strategy.model.train()

        features = torch.cat(features)
        labels = torch.cat(labels)

        # Transform
        features = strategy.model.eval_classifier.apply_transforms(features)
        class_means = compute_means(features, labels)
        class_cov = compute_covariance(features, labels)
        class_cov = strategy.model.eval_classifier.apply_cov_transforms(class_cov)

        strategy.model.eval_classifier.replace_class_means_dict(class_means)
        strategy.model.eval_classifier.replace_class_cov_dict(class_cov)

class FeCAMOracle(SupervisedPlugin):
    """
    Updates FeCAM cov and prototypes
    using the current task data
    (at the end of each task)
    """

    def __init__(self):
        super().__init__()
        self.all_datasets = []

    def after_training_exp(self, strategy, **kwargs):
        self.all_datasets.append(strategy.experience.dataset)
        full_dataset = concat_datasets(self.all_datasets)
        num_workers = kwargs["num_workers"] if "num_workers" in kwargs else 0
        loader = torch.utils.data.DataLoader(
            full_dataset.eval(),
            batch_size=strategy.train_mb_size,
            shuffle=False,
            num_workers=num_workers,
        )

        features = []
        labels = []

        was_training = strategy.model.training
        strategy.model.eval()

        for x, y, t in loader:
            x = x.to(strategy.device)
            y = y.to(strategy.device)

            with torch.no_grad():
                out = strategy.model.feature_extractor(x)

            features.append(out)
            labels.append(y)

        if was_training:
            strategy.model.train()

        features = torch.cat(features)
        labels = torch.cat(labels)

        # Transform
        features = strategy.model.eval_classifier.apply_transforms(features)
        class_means = compute_means(features, labels)
        class_cov = compute_covariance(features, labels)
        class_cov = strategy.model.eval_classifier.apply_cov_transforms(class_cov)

        strategy.model.eval_classifier.replace_class_means_dict(class_means)
        strategy.model.eval_classifier.replace_class_cov_dict(class_cov)

__all__ = ["CurrentDataFeCAMUpdate", "MemoryFeCAMUpdate", "FeCAMOracle"]
