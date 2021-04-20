################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco, Antonio Carta                                  #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import torch.nn as nn

from avalanche.models.dynamic_modules import MultiTaskModule, \
    MultiHeadClassifier


class SimpleMLP(nn.Module):
    def __init__(self, num_classes=10, input_size=28 * 28,
                 hidden_size=512, hidden_layers=1):
        super().__init__()

        self.features = nn.Sequential(
            *(nn.Linear(input_size, hidden_size),
              nn.ReLU(inplace=True),
              nn.Dropout(),
              ) * hidden_layers
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
    def __init__(self, input_size=28 * 28, hidden_size=512):
        """
            Multi-task MLP with multi-head classifier.
        """
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
