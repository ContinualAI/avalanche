import torch
from torch import nn


class NCMClassifier(nn.Module):
    """
    NCM Classifier.
    NCMClassifier performs nearest class mean classification
    measuring the distance between the input tensor and the
    ones stored in 'self.class_means'.
    """

    def __init__(self, class_means_dict={}, normalize=True):
        """
        :param class_means_dict: dictionary mapping class_id to mean vector.
            classes must be zero-indexed.
        :param normalize: whether to normalize the input with
            2-norm = 1 before computing the distance.
        """
        super().__init__()
        assert isinstance(class_means_dict, dict), \
            "class_means_dict must be a dictionary mapping class_id " \
            "to mean vector"
        self.class_means_dict = class_means_dict
        # vectorized version of class means
        self.class_means: torch.Tensor = None
        self.normalize = normalize

        self._vectorize_means_dict()

    def _vectorize_means_dict(self):
        """
        Transform the dictionary of {class_id: mean vector}
        into a tensor of shape (num_classes, feature_size), where
        `feature_size` is the size of the mean vector.
        The `num_classes` parameter is the maximum class_id found
        in the dictionary. Missing class means are treated as zero vectors.
        """
        if self.class_means_dict == {}:
            return

        max_class = max(self.class_means_dict.keys()) + 1
        first_mean = list(self.class_means_dict.values())[0]
        feature_size = first_mean.size(0)
        device = first_mean.device
        self.class_means = torch.zeros(max_class, feature_size).to(device)

        for k, v in self.class_means_dict.items():
            self.class_means[k] = self.class_means_dict[k].clone()

    @torch.no_grad()
    def forward(self, x):
        """
        :param x: (batch_size, feature_size)

        Returns a tensor of size (batch_size, num_classes) with
        negative distance of each element in the mini-batch
        with respect to each class.
        """

        assert self.class_means_dict != {}, "empty dictionary of class means."
        if self.normalize:
            # normalize across feature_size
            x = (x.T / torch.norm(x, dim=1)).T

        # (num_classes, batch_size)
        sqd = torch.cdist(self.class_means.to(x.device), x)
        # (batch_size, num_classes)
        return (-sqd).T

    def update_class_means_dict(self, class_means_dict):
        """
        Update dictionary of class means.
        If class already exists, the average of the two mean vectors
        will be computed.

        :param class_means_dict: a dictionary mapping class id
            to class mean tensor.
        """
        for k, v in class_means_dict.items():
            if k not in self.class_means_dict:
                self.class_means_dict[k] = class_means_dict[k].clone()
            else:
                device = self.class_means_dict[k].device
                self.class_means_dict[k] += class_means_dict[k].to(device)
                self.class_means_dict[k] /= 2

        self._vectorize_means_dict()


__all__ = ["NCMClassifier"]
