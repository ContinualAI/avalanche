#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from avalanche.models import DynamicModule


"""
Implementation of Cosine layer taken and modified from https://github.com/G-U-N/PyCIL
"""


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
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
    This class keeps two Cosine Linear layers, without sigma, and handles the sigma parameter
    that is common for the two of them. One CosineLinear is for the old classes and the other
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
    # WARNING Maybe does not work with initial evaluation
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = CosineLinear(in_features, num_classes)
        self.num_current_classes = num_classes
        self.feature_dim = in_features

    def adaptation(self, experience):
        max_class = torch.max(experience.classes_in_this_experience)[0]
        if max_class <= self.num_current_classes:
            # Do not adapt
            return
        self.num_current_classes = max_class
        fc = self.generate_fc(self.feature_dim, max_class + 1)
        if experience.current_experience == 1:
            fc.fc1.weight.data = self.fc.weight.data
            fc.sigma.data = self.fc.sigma.data
        else:
            prev_out_features1 = self.fc.fc1.out_features
            fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
            fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
            fc.sigma.data = self.fc.sigma.data
        del self.fc
        self.fc = fc

    def forward(self, x):
        return self.fc(x)

    def generate_fc(self, in_dim, out_dim):
        fc = SplitCosineLinear(
            in_dim, self.fc.out_features, out_dim - self.fc.out_features
        )
        return fc
