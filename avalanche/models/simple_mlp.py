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

This is the definition od the Mid-caffenet high resolution in Pythorch

"""

# Python 2-3 compatible

import torch.nn as nn


class SimpleMLP(nn.Module):

    def __init__(self, num_classes=10, input_size=28*28, hidden_size=512):
        super(SimpleMLP, self).__init__()

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


if __name__ == "__main__":

    kwargs = {'num_classes': 10}
    model = SimpleMLP(**kwargs)

    for name, module in model.named_parameters():
        print(name)
