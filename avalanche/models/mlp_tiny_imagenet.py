################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 3-02-2021                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

""" Basic Multi-Layer Perceptron (MLP) used in TinyImageNet Experiments. """

import torch.nn as nn


class SimpleMLP_TinyImageNet(nn.Module):
    """Multi-layer Perceptron for TinyImageNet benchmark."""

    def __init__(self, num_classes=200, num_channels=3):
        """
        :param num_classes: model output size
        :param num_channels: number of input channels
        """
        super(SimpleMLP_TinyImageNet, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(num_channels * 64 * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x
