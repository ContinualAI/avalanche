import torch.nn as nn


class SimpleMLP_TinyImageNet(nn.Module):

    def __init__(self, num_classes=200, num_channels=3):
        super(SimpleMLP_TinyImageNet, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(num_channels*64*64, 1024),
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
