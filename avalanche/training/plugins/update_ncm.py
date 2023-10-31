#!/usr/bin/env python3
import copy
from typing import Dict
import collections

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, nn

from avalanche.benchmarks.utils import concat_datasets
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate


@torch.no_grad()
def compute_class_means(model, dataset, batch_size, normalize, device, **kwargs):
    class_means_dict = collections.defaultdict(list())
    class_counts = collections.defaultdict(lambda: 0)
    num_workers = kwargs["num_workers"] if "num_workers" in kwargs else 0
    loader = torch.utils.data.DataLoader(
        dataset.eval(), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.eval()

    for x, y, t in loader:
        x = x.to(device)
        for class_idx in torch.unique(y):
            mask = y == class_idx
            out = model.feature_extractor(x[mask])
            class_means_dict[int(class_idx)].append(out)
            class_counts[int(class_idx)] += len(x[mask])

    for k, v in class_means_dict.items():
        v = torch.cat(v)
        if normalize:
            class_means_dict[k] = (
                torch.sum(v / torch.norm(v, dim=1, keepdim=True), dim=0)
                / class_counts[k]
            )
        else:
            class_means_dict[k] = torch.sum(v, dim=0) / class_counts[k]

        if normalize:
            class_means_dict[k] = class_means_dict[k] / class_means_dict[k].norm()

    model.train()

    return class_means_dict


class CurrentDataNCMUpdate(SupervisedPlugin):
    def __init__(self):
        super().__init__()

    # Maybe change with before_eval
    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        class_means_dict = compute_class_means(
            strategy.model,
            strategy.adapted_dataset,
            strategy.train_mb_size,
            normalize=strategy.model.eval_classifier.normalize,
            device=strategy.device,
        )
        strategy.model.eval_classifier.update_class_means_dict(class_means_dict)


class MemoryNCMUpdate(SupervisedPlugin):
    """
    Updates NCM prototypes
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
        class_means_dict = compute_class_means(
            strategy.model,
            self.storage_policy.buffer.eval(),
            batch_size=strategy.train_mb_size,
            normalize=strategy.model.eval_classifier.normalize,
            device=strategy.device,
        )
        strategy.model.eval_classifier.replace_class_means_dict(class_means_dict)


class NCMOracle(SupervisedPlugin):
    def __init__(self):
        super().__init__()
        self.all_datasets = []

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        self.all_datasets.append(strategy.experience.dataset)
        accumulated_dataset = concat_datasets(self.all_datasets)

        class_means_dict = compute_class_means(
            strategy.model,
            accumulated_dataset,
            strategy.train_mb_size,
            normalize=strategy.model.eval_classifier.normalize,
            device=strategy.device,
        )

        strategy.model.eval_classifier.replace_class_means_dict(class_means_dict)


__all__ = ["CurrentDataNCMUpdate", "MemoryNCMUpdate", "NCMOracle"]
