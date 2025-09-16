################################################################################
# Copyright (c) 2025.
# Copyrights licensed under the MIT License.
# See the accompanying LICENSE file for terms.
#
# Date: 2025-09-13
# Author(s): Muhammad Aniq, GPT-5 assistant
# Website: avalanche.continualai.org
################################################################################
from copy import deepcopy
from typing import List

import torch
from torch.utils.data import TensorDataset  # noqa: F401 (kept for reference)

from avalanche.benchmarks.utils.classification_dataset import (
    _make_taskaware_tensor_classification_dataset,
)
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class PseudoAnnotationPlugin(SupervisedPlugin):
    """
    Simple pseudo-annotation plugin for classification benchmarks.

    Before training on a new experience, use the previous model to generate
    high-confidence predictions on current data for classes seen so far and
    append those pseudo-labeled samples to the adapted dataset.

    Note: This is a minimal classification-oriented implementation (not object
    detection). It assumes inputs are tensors after transforms and that the
    benchmark exposes class timelines.
    """

    def __init__(self, confidence_thresh: float = 0.9):
        super().__init__()
        self.confidence_thresh = float(confidence_thresh)
        self._prev_model = None
        self._seen_classes: set[int] = set()

    def after_training_exp(self, strategy, **kwargs):
        self._prev_model = deepcopy(strategy.model)
        self._prev_model.to(strategy.device)
        self._prev_model.eval()

        if hasattr(strategy.experience, "classes_in_this_experience"):
            self._seen_classes.update(
                map(int, strategy.experience.classes_in_this_experience)
            )

    @torch.no_grad()
    def after_train_dataset_adaptation(self, strategy, **kwargs):
        if self._prev_model is None:
            return

        if not hasattr(strategy.experience, "classes_in_this_experience"):
            return

        new_classes = set(map(int, strategy.experience.classes_in_this_experience))
        old_classes = sorted(list(self._seen_classes - new_classes))
        if len(old_classes) == 0:
            return

        # Use the adapted dataset (already set to train transforms)
        from avalanche.training.utils import load_all_dataset

        assert strategy.adapted_dataset is not None
        x, *rest = load_all_dataset(strategy.adapted_dataset)
        x = x.to(strategy.device)

        logits = self._prev_model(x)
        probs = torch.softmax(logits, dim=1)

        xs: List[torch.Tensor] = []
        ys: List[int] = []
        for idx in range(probs.shape[0]):
            p = probs[idx]
            for c in old_classes:
                if c < p.numel() and float(p[c]) >= self.confidence_thresh:
                    xs.append(x[idx].detach().cpu())
                    ys.append(int(c))

        if len(ys) == 0:
            return

        x_t = torch.stack(xs, dim=0)
        y_t = torch.tensor(ys, dtype=torch.long)
        pseudo_ds = _make_taskaware_tensor_classification_dataset(
            x_t, y_t, task_labels=0
        )

        strategy.adapted_dataset = strategy.adapted_dataset.concat(pseudo_ds)


__all__ = ["PseudoAnnotationPlugin"]
