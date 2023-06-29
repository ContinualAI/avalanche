import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, n_classes, input_channels):
        """LeNet5 architecture.

        :param n_classes:
        :param input_channels:
        """
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=6,
                kernel_size=5,
                stride=1,
            ),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
        )
        self.ff = nn.Sequential(nn.Linear(in_features=120, out_features=84), nn.Tanh())

        self.classifier = nn.Sequential(
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.ff(x)
        logits = self.classifier(x)
        return logits
