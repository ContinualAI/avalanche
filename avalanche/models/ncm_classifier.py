from typing import Dict

import torch
from torch import Tensor, nn

from avalanche.models import DynamicModule


class NCMClassifier(DynamicModule):
    """
    NCM Classifier.
    NCMClassifier performs nearest class mean classification
    measuring the distance between the input tensor and the
    ones stored in 'self.class_means'.

    Before being used for inference, NCMClassifier needs to
    be updated with a mean vector per class, by calling
    `update_class_means_dict`.

    This class registers a `class_means` buffer that stores
    the class means in a single tensor of shape
    [max_class_id_seen, feature_size]. Classes with ID smaller
    than `max_class_id_seen` are associated with a 0-vector.
    """

    def __init__(self, normalize: bool = True):
        """
        :param normalize: whether to normalize the input with
            2-norm = 1 before computing the distance.
        """
        super().__init__()
        # vectorized version of class means
        self.register_buffer("class_means", None)
        self.class_means_dict = {}

        self.normalize = normalize
        self.max_class = -1

    def load_state_dict(self, state_dict, strict: bool = True):
        self.class_means = state_dict["class_means"]
        super().load_state_dict(state_dict, strict)
        # fill dictionary
        if self.class_means is not None:
            for i in range(self.class_means.shape[0]):
                if (self.class_means[i] != 0).any():
                    self.class_means_dict[i] = self.class_means[i].clone()
        self.max_class = max(self.class_means_dict.keys())

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

        max_class = max(self.class_means_dict.keys())
        self.max_class = max(max_class, self.max_class)
        first_mean = list(self.class_means_dict.values())[0]
        feature_size = first_mean.size(0)
        device = first_mean.device
        self.class_means = torch.zeros(self.max_class + 1, feature_size).to(device)

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
        if self.class_means_dict == {}:
            self.init_missing_classes(range(self.max_class + 1), x.shape[1], x.device)

        assert self.class_means_dict != {}, "no class means available."
        if self.normalize:
            # normalize across feature_size
            x = (x.T / torch.norm(x.T, dim=0)).T

        # (num_classes, batch_size)
        sqd = torch.cdist(self.class_means.to(x.device), x)
        # (batch_size, num_classes)
        return (-sqd).T

    def update_class_means_dict(
        self, class_means_dict: Dict[int, Tensor], momentum: float = 0.5
    ):
        """
        Update dictionary of class means.
        If class already exists, the average of the two mean vectors
        will be computed.

        :param class_means_dict: a dictionary mapping class id
            to class mean tensor.
        :param momentum: Weighting of the new means vs old means
                         in the update. 1 = replace, 0 = don't update
        """
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

    def replace_class_means_dict(self, class_means_dict: Dict[int, Tensor]):
        """
        Replace existing dictionary of means with a given dictionary.
        """
        assert isinstance(class_means_dict, dict), (
            "class_means_dict must be a dictionary mapping class_id " "to mean vector"
        )
        self.class_means_dict = {k: v.clone() for k, v in class_means_dict.items()}
        self._vectorize_means_dict()

    def init_missing_classes(self, classes, class_size, device):
        for k in classes:
            if k not in self.class_means_dict:
                self.class_means_dict[k] = torch.zeros(class_size).to(device)
        self._vectorize_means_dict()

    def eval_adaptation(self, experience):
        classes = experience.classes_in_this_experience
        for k in classes:
            self.max_class = max(k, self.max_class)
        if self.class_means is not None:
            self.init_missing_classes(
                classes, self.class_means.shape[1], self.class_means.device
            )


__all__ = ["NCMClassifier"]
