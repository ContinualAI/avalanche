#!/usr/bin/env python3
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from avalanche.models import DynamicModule


"""
Implementation of Cosine layer taken and modified from https://github.com/G-U-N/PyCIL
"""


class CosineLinear(nn.Module):
    """
    Cosine layer defined in
    "Learning a Unified Classifier Incrementally via Rebalancing"
    by Saihui Hou et al.

    Implementation modified from https://github.com/G-U-N/PyCIL

    This layer is aimed at countering the task-recency bias by removing the bias
    in the classifier and normalizing the weight and the input feature before
    computing the weight-feature product
    """

    def __init__(self, in_features, out_features, sigma=True):
        """
        :param in_features: number of input features
        :param out_features: number of classes
        :param sigma: learnable output scaling factor
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter("sigma", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(
            F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1)
        )
        if self.sigma is not None:
            out = self.sigma * out

        return out


class SplitCosineLinear(nn.Module):
    """
    This class keeps two Cosine Linear layers, without sigma scaling,
    and handles the sigma parameter that is common for the two of them.
    One CosineLinear is for the old classes and the other
    one is for the new classes
    """

    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter("sigma", None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1, out2), dim=1)

        if self.sigma is not None:
            out = self.sigma * out

        return out


class CosineIncrementalClassifier(DynamicModule):
    """
    Equivalent to IncrementalClassifier but using the cosine layer
    described in "Learning a Unified Classifier Incrementally via Rebalancing"
    by Saihui Hou et al.
    """

    def __init__(self, in_features, num_classes=0):
        """
        :param in_features: Number of input features
        :param num_classes: Number of initial classes (default=0)
                            If set to more than 0, the initial logits
                            will be mapped to the corresponding sequence of
                            classes starting from 0.
        """
        super().__init__()
        self.class_order = []
        self.classes = set()

        if num_classes == 0:
            self.fc = None
        else:
            self.fc = CosineLinear(in_features, num_classes, sigma=True)
            for i in range(num_classes):
                self.class_order.append(i)
            self.classes = set(range(5))

        self.feature_dim = in_features

    def adaptation(self, experience):
        num_classes = len(experience.classes_in_this_experience)

        new_classes = set(experience.classes_in_this_experience) - set(self.classes)

        if len(new_classes) == 0:
            # Do not adapt
            return

        self.classes = self.classes.union(new_classes)

        for c in list(new_classes):
            self.class_order.append(c)

        max_index = len(self.class_order)

        if self.fc is None:
            self.fc = CosineLinear(self.feature_dim, max_index, sigma=True)
            return

        fc = self._generate_fc(self.feature_dim, max_index)

        if isinstance(self.fc, CosineLinear):
            # First exp self.fc is CosineLinear
            # while it is SplitCosineLinear for subsequent exps
            fc.fc1.weight.data = self.fc.weight.data
            fc.sigma.data = self.fc.sigma.data
        elif isinstance(self.fc, SplitCosineLinear):
            prev_out_features1 = self.fc.fc1.out_features
            fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
            fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
            fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def forward(self, x):
        unmapped_logits = self.fc(x)

        # Mask by default unseen classes
        mapped_logits = (
            torch.ones(len(unmapped_logits), np.max(self.class_order) + 1) * -1000
        )
        mapped_logits.to(x.device)

        # Now map to classes
        mapped_logits[:, self.class_order] = unmapped_logits

        return mapped_logits

    def _generate_fc(self, in_dim, out_dim):
        fc = SplitCosineLinear(
            in_dim, self.fc.out_features, out_dim - self.fc.out_features
        )
        return fc


__all__ = ["CosineLinear", "CosineIncrementalClassifier"]
