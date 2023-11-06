import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, nn

from avalanche.models import DynamicModule


class FeCAMClassifier(DynamicModule):
    """
    FeCAMClassifier

    Similar to NCM but uses malahanobis distance instead of l2 distance

    This approach has been proposed for continual learning in
    "FeCAM: Exploiting the Heterogeneity of Class Distributions
    in Exemplar-Free Continual Learning" Goswami et. al.
    (Neurips 2023)

    This requires the storage of full per-class covariance matrices
    """

    def __init__(
        self,
        tukey=True,
        shrinkage=True,
        shrink1: float = 1.0,
        shrink2: float = 1.0,
        tukey1: float = 0.5,
        covnorm=True,
    ):
        """
        :param tukey: whether to use the tukey transforms
                      (help get the distribution closer
                       to multivariate gaussian)
        :param shrinkage: whether to shrink the covariance matrices
        :param shrink1:
        :param shrink2:
        :param tukey1: power in tukey transforms
        :param covnorm: whether to normalize the covariance matrix
        """
        super().__init__()
        self.class_means_dict = {}
        self.class_cov_dict = {}

        self.register_buffer("class_means", None)
        self.register_buffer("class_covs", None)

        self.tukey = tukey
        self.shrinkage = shrinkage
        self.covnorm = covnorm
        self.shrink1 = shrink1
        self.shrink2 = shrink2
        self.tukey1 = tukey1

        self.max_class = -1

    @torch.no_grad()
    def forward(self, x):
        """
        :param x: (batch_size, feature_size)

        Returns a tensor of size (batch_size, num_classes) with
        negative distance of each element in the mini-batch
        with respect to each class.
        """
        if self.class_means_dict == {}:
            self.init_missing_classes(range(self.max_class + 1), x.shape[1], x.device)

        assert self.class_means_dict != {}, "no class means available."

        if self.tukey:
            x = self._tukey_transforms(x)

        maha_dist = []
        for class_id, prototype in self.class_means_dict.items():
            cov = self.class_cov_dict[class_id]
            dist = self._mahalanobis(x, prototype, cov)
            maha_dist.append(dist)

        # n_classes, batch_size
        maha_dis = torch.stack(maha_dist).T

        # (batch_size, num_classes)
        return -maha_dis

    def _mahalanobis(self, vectors, class_means, cov):
        x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(
            class_means, p=2, dim=-1
        )
        inv_covmat = torch.linalg.pinv(cov).float().to(vectors.device)
        left_term = torch.matmul(x_minus_mu, inv_covmat)
        mahal = torch.matmul(left_term, x_minus_mu.T)
        return torch.diagonal(mahal, 0)

    def _tukey_transforms(self, x):
        x = torch.tensor(x)
        if self.tukey1 == 0:
            return torch.log(x)
        else:
            return torch.pow(x, self.tukey1)

    def _tukey_invert_transforms(self, x):
        x = torch.tensor(x)
        if self.tukey1 == 0:
            return torch.exp(x)
        else:
            return torch.pow(x, 1 / self.tukey1)

    def _shrink_cov(self, cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag * mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0]).to(cov.device)
        cov_ = (
            cov
            + (self.shrink1 * diag_mean * iden)
            + (self.shrink2 * off_diag_mean * (1 - iden))
        )
        return cov_

    def _vectorize_means_dict(self):
        if self.class_means_dict == {}:
            return

        max_class = max(self.class_means_dict.keys())
        self.max_class = max(max_class, self.max_class)
        first_mean = list(self.class_means_dict.values())[0]
        feature_size = first_mean.size(0)
        device = first_mean.device
        self.class_means = torch.zeros(self.max_class + 1, feature_size).to(device)

        for k, v in self.class_means_dict.items():
            self.class_means[k] = self.class_means_dict[k].clone()

    def _vectorize_cov_dict(self):
        if self.class_cov_dict == {}:
            return

        max_class = max(self.class_cov_dict.keys())
        self.max_class = max(max_class, self.max_class)
        first_mean = list(self.class_cov_dict.values())[0]
        feature_size = first_mean.size(0)
        device = first_mean.device
        self.class_covs = torch.zeros(
            self.max_class + 1, feature_size, feature_size
        ).to(device)

        for k, v in self.class_cov_dict.items():
            self.class_covs[k] = self.class_cov_dict[k].clone()

    def _normalize_cov(self, cov_mat):
        norm_cov_mat = {}
        for key, cov in cov_mat.items():
            sd = torch.sqrt(torch.diagonal(cov))  # standard deviations of the variables
            cov = cov / (torch.matmul(sd.unsqueeze(1), sd.unsqueeze(0)))
            norm_cov_mat[key] = cov

        return norm_cov_mat

    def update_class_means_dict(
        self, class_means_dict: Dict[int, Tensor], momentum: float = 0.5
    ):
        assert momentum <= 1 and momentum >= 0
        assert isinstance(class_means_dict, dict), (
            "class_means_dict must be a dictionary mapping class_id " "to mean vector"
        )
        for k, v in class_means_dict.items():
            if k not in self.class_means_dict or (self.class_means_dict[k] == 0).all():
                self.class_means_dict[k] = class_means_dict[k].clone()
            else:
                device = self.class_means_dict[k].device
                self.class_means_dict[k] = (
                    momentum * class_means_dict[k].to(device)
                    + (1 - momentum) * self.class_means_dict[k]
                )
        self._vectorize_means_dict()

    def update_class_cov_dict(
        self, class_cov_dict: Dict[int, Tensor], momentum: float = 0.5
    ):
        assert momentum <= 1 and momentum >= 0
        assert isinstance(class_cov_dict, dict), (
            "class_cov_dict must be a dictionary mapping class_id " "to mean vector"
        )
        for k, v in class_cov_dict.items():
            if k not in self.class_cov_dict or (self.class_cov_dict[k] == 0).all():
                self.class_cov_dict[k] = class_cov_dict[k].clone()
            else:
                device = self.class_cov_dict[k].device
                self.class_cov_dict[k] = (
                    momentum * class_cov_dict[k].to(device)
                    + (1 - momentum) * self.class_cov_dict[k]
                )
        self._vectorize_cov_dict()

    def replace_class_means_dict(
        self,
        class_means_dict: Dict[int, Tensor],
    ):
        self.class_means_dict = class_means_dict
        self._vectorize_means_dict()

    def replace_class_cov_dict(
        self,
        class_cov_dict: Dict[int, Tensor],
    ):
        self.class_cov_dict = class_cov_dict
        self._vectorize_cov_dict()

    def init_missing_classes(self, classes, class_size, device):
        for k in classes:
            if k not in self.class_means_dict:
                self.class_means_dict[k] = torch.zeros(class_size).to(device)
                self.class_cov_dict[k] = torch.eye(class_size).to(device)

    def eval_adaptation(self, experience):
        classes = experience.classes_in_this_experience
        for k in classes:
            self.max_class = max(k, self.max_class)

        if len(self.class_means_dict) > 0:
            self.init_missing_classes(
                classes,
                list(self.class_means_dict.values())[0].shape[0],
                list(self.class_means_dict.values())[0].device,
            )

    def apply_transforms(self, features):
        if self.tukey:
            features = self._tukey_transforms(features)
        return features

    def apply_invert_transforms(self, features):
        if self.tukey:
            features = self._tukey_invert_transforms(features)
        return features

    def apply_cov_transforms(self, class_cov):
        if self.shrinkage:
            for key, cov in class_cov.items():
                class_cov[key] = self._shrink_cov(cov)
                class_cov[key] = self._shrink_cov(class_cov[key])
        if self.covnorm:
            class_cov = self._normalize_cov(class_cov)
        return class_cov

    def load_state_dict(self, state_dict, strict: bool = True):
        self.class_means = state_dict["class_means"]
        self.class_covs = state_dict["class_covs"]

        super().load_state_dict(state_dict, strict)

        # fill dictionary
        if self.class_means is not None:
            for i in range(self.class_means.shape[0]):
                if (self.class_means[i] != 0).any():
                    self.class_means_dict[i] = self.class_means[i].clone()

        self.max_class = max(self.class_means_dict.keys())

        if self.class_covs is not None:
            for i in range(self.class_covs.shape[0]):
                if (self.class_covs[i] != 0).any():
                    self.class_cov_dict[i] = self.class_covs[i].clone()


def compute_covariance(features, labels) -> Dict:
    class_cov = {}
    for class_id in list(torch.unique(labels).cpu().int().numpy()):
        mask = labels == class_id
        class_features = features[mask]
        cov = torch.cov(class_features.T)
        class_cov[class_id] = cov
    return class_cov


def compute_means(features, labels) -> Dict:
    class_means = {}
    for class_id in list(torch.unique(labels).cpu().int().numpy()):
        mask = labels == class_id
        class_features = features[mask]
        prototype = torch.mean(class_features, dim=0)
        class_means[class_id] = prototype
    return class_means
