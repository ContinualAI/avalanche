#!/usr/bin/env python3
import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avalanche.benchmarks.utils import concat_datasets
from avalanche.models import FeCAMClassifier
from avalanche.models.fecam import compute_covariance, compute_means
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate


def _gather_means_and_cov(model, dataset, batch_size, device, **kwargs):
    num_workers = kwargs["num_workers"] if "num_workers" in kwargs else 0
    loader = torch.utils.data.DataLoader(
        dataset.eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    features = []
    labels = []

    was_training = model.training
    model.eval()

    for x, y, t in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            out = model.feature_extractor(x)

        features.append(out)
        labels.append(y)

    if was_training:
        model.train()

    features = torch.cat(features)
    labels = torch.cat(labels)

    # Transform
    features = model.eval_classifier.apply_transforms(features)
    class_means = compute_means(features, labels)
    class_cov = compute_covariance(features, labels)
    class_cov = model.eval_classifier.apply_cov_transforms(class_cov)

    return class_means, class_cov


def _check_has_fecam(model):
    assert hasattr(model, "eval_classifier")
    assert isinstance(model.eval_classifier, FeCAMClassifier)


class CurrentDataFeCAMUpdate(SupervisedPlugin):
    """
    Updates FeCAM cov and prototypes
    using the current task data
    (at the end of each task)
    """

    def __init__(self):
        super().__init__()

    def after_training_exp(self, strategy, **kwargs):
        _check_has_fecam(strategy.model)

        class_means, class_cov = _gather_means_and_cov(
            strategy.model,
            strategy.experience.dataset,
            strategy.train_mb_size,
            strategy.device,
            **kwargs
        )

        strategy.model.eval_classifier.update_class_means_dict(class_means)
        strategy.model.eval_classifier.update_class_cov_dict(class_cov)


class MemoryFeCAMUpdate(SupervisedPlugin):
    """
    Updates FeCAM cov and prototypes
    using the data contained inside a memory buffer
    """

    def __init__(self, mem_size=2000, storage_policy=None):
        super().__init__()
        if storage_policy is None:
            self.storage_policy = ClassBalancedBuffer(max_size=mem_size)
        else:
            self.storage_policy = storage_policy

    def after_training_exp(self, strategy, **kwargs):
        _check_has_fecam(strategy.model)

        self.storage_policy.update(strategy)

        class_means, class_cov = _gather_means_and_cov(
            strategy.model,
            self.storage_policy.buffer.eval(),
            strategy.train_mb_size,
            strategy.device,
            **kwargs
        )

        strategy.model.eval_classifier.update_class_means_dict(class_means)
        strategy.model.eval_classifier.update_class_cov_dict(class_cov)


class FeCAMOracle(SupervisedPlugin):
    """
    Updates FeCAM cov and prototypes
    using all the data seen so far
    WARNING: This is an oracle,
    and thus breaks assumptions usually made
    in continual learning algorithms i
    (storage of full dataset)
    This is meant to be used as an upper bound
    for FeCAM based methods
    (i.e when trying to estimate prototype and covariance drift)
    """

    def __init__(self):
        super().__init__()
        self.all_datasets = []

    def after_training_exp(self, strategy, **kwargs):
        _check_has_fecam(strategy.model)

        self.all_datasets.append(strategy.experience.dataset)
        full_dataset = concat_datasets(self.all_datasets)

        class_means, class_cov = _gather_means_and_cov(
            strategy.model,
            full_dataset,
            strategy.train_mb_size,
            strategy.device,
            **kwargs
        )

        strategy.model.eval_classifier.update_class_means_dict(class_means)
        strategy.model.eval_classifier.update_class_cov_dict(class_cov)


__all__ = ["CurrentDataFeCAMUpdate", "MemoryFeCAMUpdate", "FeCAMOracle"]
