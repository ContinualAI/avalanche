################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is the definition od the Mid-caffenet high resolution in Pytorch.
"""

# Python 2-3 compatible

import torch.nn as nn

from avalanche.models.dynamic_modules import MultiTaskModule, MultiHeadClassifier


class SimpleMLP(nn.Module):
    def __init__(self, num_classes=10, input_size=28*28, hidden_size=512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x


class MTSimpleMLP(nn.Module, MultiTaskModule):
    def __init__(self, input_size=28*28, hidden_size=512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = MultiHeadClassifier(hidden_size)
        self._input_size = input_size

    def forward(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x


__all__ = [
    'SimpleMLP',
    'MTSimpleMLP'
]