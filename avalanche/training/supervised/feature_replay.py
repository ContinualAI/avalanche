#!/usr/bin/env python3
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from avalanche.core import SupervisedPlugin
from avalanche.models.utils import FeatureExtractorModel, avalanche_forward
from avalanche.training import ACECriterion
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.utils import cycle
from avalanche.training.losses import MaskedCrossEntropy


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


class FeatureReplay(SupervisedTemplate):
    """
    Store some last layer features and use them for replay

    Replay is performed following the PR-ACE protocol
    defined in Magistri et al. https://openreview.net/forum?id=7D9X2cFnt1

    Training the current task with masked cross entropy for current task classes
    and training the classifier with cross entropy
    criterion over all previously seen classes
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion=MaskedCrossEntropy(),
        last_layer_name: str = "classifier",
        mem_size: int = 200,
        batch_size_mem: int = 10,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.replay_loader = None

        # Criterion used when doing feature replay
        # self._criterion is used only on current data
        self.full_criterion = nn.CrossEntropyLoss()

        # Turn the model into feature extractor
        last_layer = getattr(self.model, last_layer_name)
        setattr(self.model, last_layer_name, nn.Identity())
        self.model = FeatureExtractorModel(self.model, last_layer)

    def _after_training_exp(self, **kwargs):
        super()._after_training_exp(**kwargs)
        feature_dataset = self.gather_feature_dataset()
        self.storage_policy.update_from_dataset(feature_dataset)

        if self.batch_size_mem is None:
            self.batch_size_mem = self.train_mb_size

        # Create replay loader
        self.replay_loader = cycle(
            torch.utils.data.DataLoader(
                self.storage_policy.buffer,
                batch_size=self.batch_size_mem,
                shuffle=True,
                drop_last=True,
            )
        )

    def criterion(self, **kwargs):
        if self.replay_loader is None:
            return self._criterion(self.mb_output, self.mb_y)

        batch_feats, batch_y, batch_t = next(self.replay_loader)
        batch_feats, batch_y = batch_feats.to(self.device), batch_y.to(self.device)
        out = self.model.train_classifier(batch_feats)

        # Select additional outputs from current
        # output to be learned with cross-entropy
        weight_current = 1 / (self.experience.current_experience + 1)

        mb_output = self.model.train_classifier(self.model.features.detach())

        loss = 0.5 * self._criterion(self.mb_output, self.mb_y) + 0.5 * (
            (1 - weight_current) * self.full_criterion(out, batch_y)
            + weight_current * self.full_criterion(mb_output, self.mb_y)
        )
        return loss

    @torch.no_grad()
    def gather_feature_dataset(self):
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            self.experience.dataset, batch_size=self.train_mb_size, shuffle=True
        )
        all_features = []
        all_labels = []
        for x, y, t in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            feats = avalanche_forward(self.model.feature_extractor, x, t)
            all_features.append(feats.cpu())
            all_labels.append(y.cpu())
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        features_dataset = FeatureDataset(all_features, all_labels)
        return features_dataset


__all__ = ["FeatureReplay"]
