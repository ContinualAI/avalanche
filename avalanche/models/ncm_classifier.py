import torch
from torch import nn


class NCMClassifier(nn.Module):
    """
    NCM Classifier.
    NCMClassifier performs nearest class mean classification
    measuring the distance between the input tensor and the
    ones stored in 'self.class_means'.
    """

    def __init__(self, class_mean=None, normalize=True):
        """
        :param class_mean: tensor of dimension (num_classes x feature_size)
            used to classify input patterns.
        :param normalize: whether to normalize the input with
            2-norm = 1 before computing the distance.
        """
        super().__init__()
        self.class_means = class_mean
        self.normalize = normalize

    def forward(self, x):
        if self.normalize:
            x = (x.T / torch.norm(x.T, dim=0)).T
        sqd = torch.cdist(self.class_means[:, :].T, x)
        return (-sqd).T


__all__ = ["NCMClassifier"]
