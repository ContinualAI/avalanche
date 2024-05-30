import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCELoss

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.regularization import cross_entropy_with_oh_targets
from avalanche._annotations import deprecated


class ICaRLLossPlugin(SupervisedPlugin):
    """
    ICaRLLossPlugin
    Similar to the Knowledge Distillation Loss. Works as follows:
        The target is constructed by taking the one-hot vector target for the
        current sample and assigning to the position corresponding to the
        past classes the output of the old model on the current sample.
        Doesn't work if classes observed in previous experiences might be
        observed again in future training experiences.
    """

    def __init__(self):
        super().__init__()
        self.criterion = BCELoss()

        self.old_classes = []
        self.old_model = None
        self.old_logits = None

    def before_forward(self, strategy, **kwargs):
        if self.old_model is not None:
            with torch.no_grad():
                self.old_logits = self.old_model(strategy.mb_x)

    def __call__(self, logits, targets):
        predictions = torch.sigmoid(logits)

        one_hot = torch.zeros(
            targets.shape[0],
            logits.shape[1],
            dtype=torch.float,
            device=logits.device,
        )
        one_hot[range(len(targets)), targets.long()] = 1

        if self.old_logits is not None:
            old_predictions = torch.sigmoid(self.old_logits)
            one_hot[:, self.old_classes] = old_predictions[:, self.old_classes]
            self.old_logits = None

        return self.criterion(predictions, one_hot)

    def after_training_exp(self, strategy, **kwargs):
        if self.old_model is None:
            old_model = copy.deepcopy(strategy.model)
            self.old_model = old_model.to(strategy.device)

        self.old_model.load_state_dict(strategy.model.state_dict())

        self.old_classes += np.unique(strategy.experience.dataset.targets).tolist()


class SCRLoss(torch.nn.Module):
    """
    Supervised Contrastive Replay Loss as defined in Eq. 5 of
    https://arxiv.org/pdf/2103.13885.pdf.

    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    Original GitHub repository: https://github.com/HobbitLong/SupContrast/
    LICENSE: BSD 2-Clause License
    """

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        features: [bsz, n_views, f_dim]
        `n_views` is the number of crops from each image, better
        be L2 normalized in f_dim dimension

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class MaskedCrossEntropy(SupervisedPlugin):
    """
    Masked Cross Entropy

    This criterion can be used for instance in Class Incremental
    Learning Problems when no examplars are used
    (i.e LwF in Class Incremental Learning would need to use mask="new").
    """

    def __init__(self, classes=None, mask="seen", reduction="mean"):
        """
        param: classes: Initial value for current classes
        param: mask: "all" normal cross entropy, uses all the classes seen so far
                     "old" cross entropy only on the old classes
                     "new" cross entropy only on the new classes
        param: reduction: "mean" or "none", average or per-sample loss
        """
        super().__init__()
        assert mask in ["seen", "new", "old", "all"]
        if classes is not None:
            self.current_classes = set(classes)
        else:
            self.current_classes = set()

        self.old_classes = set()
        self.reduction = reduction
        self.mask = mask

    def __call__(self, logits, targets):
        oh_targets = F.one_hot(targets, num_classes=logits.shape[1])

        oh_targets = oh_targets[:, self.current_mask(logits.shape[1])]
        logits = logits[:, self.current_mask(logits.shape[1])]

        return cross_entropy_with_oh_targets(
            logits,
            oh_targets.float(),
            reduction=self.reduction,
        )

    def current_mask(self, logit_shape):
        if self.mask == "seen":
            return list(self.current_classes.union(self.old_classes))
        elif self.mask == "new":
            return list(self.current_classes)
        elif self.mask == "old":
            return list(self.old_classes)
        elif self.mask == "all":
            return list(range(int(logit_shape)))

    def _adaptation(self, new_classes):
        self.old_classes = self.old_classes.union(self.current_classes)
        self.current_classes = set(new_classes)

    def pre_adapt(self, agent, exp):
        self._adaptation(exp.classes_in_this_experience)

    @deprecated(0.7, "Please switch to the `pre_adapt`or `_adaptation` methods.")
    def adaptation(self, new_classes):
        self._adaptation(new_classes)

    @deprecated(0.7, "Please switch to the `pre_adapt` method.")
    def before_training_exp(self, strategy, **kwargs):
        self._adaptation(strategy.experience.classes_in_this_experience)


__all__ = ["ICaRLLossPlugin", "SCRLoss", "MaskedCrossEntropy"]
