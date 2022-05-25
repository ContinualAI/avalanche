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

from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)


class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network

    **Example**::

        >>> from avalanche.models import SimpleCNN
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleCNN(num_classes=n_classes)
        >>> print(model) # View model details
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MTSimpleCNN(SimpleCNN, MultiTaskModule):
    """
    Convolutional Neural Network
    with multi-head classifier
    """

    def __init__(self):
        super().__init__()
        self.classifier = MultiHeadClassifier(64)

    def forward(self, x, task_labels):
        x = self.features(x)
        x = x.squeeze()
        x = self.classifier(x, task_labels)
        return x


__all__ = ["SimpleCNN", "MTSimpleCNN"]
