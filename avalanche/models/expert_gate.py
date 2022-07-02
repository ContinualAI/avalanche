import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss

from torchvision import transforms
import torchvision.models as models

from .utils import FeatureExtractorBackbone

from avalanche.models import MultiTaskModule
from avalanche.models.utils import Flatten
from avalanche.benchmarks.scenarios.generic_scenario import CLExperience


def AE_loss(target, reconstruction):
    reconstruction_loss = mse_loss(
        input=reconstruction, target=target, reduction="sum")
    return reconstruction_loss


class Autoencoder(nn.Module):
    def __init__(self, shape, 
                 latent_dim, 
                 arch="alexnet",
                 pretrained_flag=True,
                 device="cpu",
                 output_layer_name="features"):

        super().__init__()

        # Select pretrained model
        base_template = (models.__dict__[arch](
            pretrained=pretrained_flag).to(device))

        self.feature_module = FeatureExtractorBackbone(
                base_template, "features")

        self.shape = shape

        # Encoder Linear -> ReLU
        flattened_size = torch.Size(shape).numel()
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(flattened_size, latent_dim),
            nn.ReLU()
        )

        # Decoder Linear -> Sigmoid
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flattened_size), 
            nn.Sigmoid()
        )

    def forward(self, x):

        # Encode input
        x = self.encoder(x)

        # Reconstruction
        x = self.decoder(x)

        return x.view(-1, *self.shape)


class ExpertModel(nn.Module):
    def __init__(self, 
                 num_classes, 
                 arch, 
                 device, 
                 pretrained_flag, 
                 feature_template=None):
        super().__init__()

        self.num_classes = num_classes

        # Select pretrained model
        base_template = (models.__dict__[arch](
            pretrained=pretrained_flag).to(device))

        # Set the feature module
        if (feature_template):
            self.feature_module = feature_template.feature_module

        else: 
            self.feature_module = FeatureExtractorBackbone(
                base_template, "features")

        # Set avgpool layer
        self.avg_pool = base_template._modules['avgpool']

        # Flattener
        self.flatten = Flatten()

        # Set the classifier module
        self.classifier_module = base_template._modules['classifier']

        # Customize final layer for number of classes
        original_classifier_input_dim = self.classifier_module[-1].in_features
        self.classifier_module[-1] = nn.Linear(
            original_classifier_input_dim, self.num_classes)

    def forward(self, x):
        x = self.feature_module(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier_module(x)
        return x


class ExpertGate(nn.Module):
    def __init__(
        self,
        shape,
        num_classes,
        rel_thresh=0.85,
        arch="alexnet",
        pretrained_flag=True,
        device="cpu",
        output_layer_name="features"
    ):
        super().__init__()

        # Store variables
        self.shape = shape
        self.num_classes = num_classes
        self.rel_thresh = rel_thresh
        self.arch = arch
        self.pretrained_flag = pretrained_flag
        self.device = device

        # Dict for autoencoders
        # {task, autoencoder}
        self.autoencoder_dict = nn.ModuleDict()

        # Dict for experts
        # {task, expert}
        self.expert_dict = nn.ModuleDict()

        self.expert = (
            models.__dict__[arch](pretrained=pretrained_flag)
            .to(device)
            .eval()
        )

    def forward(self, x):
        return self.expert(x)
