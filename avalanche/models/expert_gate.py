from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch.nn.functional import mse_loss, softmax

from torchvision import transforms
import torchvision.models as models

from .utils import FeatureExtractorBackbone

from avalanche.models import MultiTaskModule
from avalanche.models.utils import Flatten
from avalanche.benchmarks.scenarios.generic_scenario import CLExperience


def AE_loss(target, reconstruction):
    """Calculates the MSE loss for the autoencoder by comparing the 
    reconstruction to the pre-processed input. 
    """
    reconstruction_loss = mse_loss(
        input=reconstruction, target=target, reduction="sum")
    return reconstruction_loss


class ExpertAutoencoder(nn.Module):
    """The expert autoencoder that determines which expert classifier to select
    for the incoming data.
    """

    def __init__(self, 
                 shape, 
                 latent_dim, 
                 device,
                 arch="alexnet",
                 pretrained_flag=True,
                 output_layer_name="features"):

        super().__init__()

        # Select pretrained AlexNet for preprocessing input 
        base_template = (models.__dict__[arch](
            weights=('AlexNet_Weights.IMAGENET1K_V1' 
                     if pretrained_flag 
                     else 'AlexNet_Weights.NONE'))
            .to(device))

        self.feature_module = FeatureExtractorBackbone(
                base_template, output_layer_name)

        self.feature_module.to(device)

        self.shape = shape
        self.device = device

        # Freeze the feature module
        for param in self.feature_module.parameters():
            param.requires_grad = False

        # Flatten input
        # Encoder Linear -> ReLU
        flattened_size = torch.Size(shape).numel()
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(flattened_size, latent_dim),
            nn.ReLU()
        ).to(device)

        # Decoder Linear -> Sigmoid
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flattened_size), 
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):

        # Preprocessing step
        x = x.to(self.device)
        x = self.feature_module(x)
        x = sigmoid(x)

        # Encode input
        x = self.encoder(x)

        # Reconstruction
        x = self.decoder(x)

        return x.view(-1, *self.shape)


class ExpertModel(nn.Module):
    """The expert classifier behind the autoencoder that is trained for a
    specific task.
    """

    def __init__(self, 
                 num_classes, 
                 arch, 
                 device, 
                 pretrained_flag, 
                 feature_template=None):
        super().__init__()

        self.device = device
        self.num_classes = num_classes

        # Select pretrained AlexNet for feature backbone
        base_template = (models.__dict__[arch](
            weights=('AlexNet_Weights.IMAGENET1K_V1' 
                     if pretrained_flag 
                     else 'AlexNet_Weights.NONE'))
            .to(device))

        # Set the feature module from provided template 
        if (feature_template):
            self.feature_module = feature_template.feature_module

        # Use base template if nothing provided
        else: 
            self.feature_module = FeatureExtractorBackbone(
                base_template, "features")

        # Set avgpool layer
        self.avg_pool = base_template._modules['avgpool']

        # Flattener
        self.flatten = Flatten()

        # Classifier module
        self.classifier_module = base_template._modules['classifier']

        # Customize final layer for  the number of classes in the data
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
    """Overall parent module that holds the dictionary of expert autoencoders 
    and expert classifiers. 
    """

    def __init__(
        self,
        shape,
        device,
        arch="alexnet",
        pretrained_flag=True,
        output_layer_name="features"
    ):
        super().__init__()

        # Store variables
        self.shape = shape
        self.arch = arch
        self.pretrained_flag = pretrained_flag
        self.device = device

        # Dictionary for autoencoders
        # {task, autoencoder}
        self.autoencoder_dict = nn.ModuleDict()

        # Dictionary for experts
        # {task, expert}
        self.expert_dict = nn.ModuleDict()

        # Initialize an expert with pretrained AlexNet
        self.expert = (
            models.__dict__[arch](
                weights=('AlexNet_Weights.IMAGENET1K_V1' 
                         if pretrained_flag 
                         else 'AlexNet_Weights.NONE'))
            .to(device)
            .eval()
        )

    def _get_average_reconstruction_error(self, autoencoder_id, x):

        # Select autoencoder with the given ID
        autoencoder = self.autoencoder_dict[str(autoencoder_id)]

        # Run input through autoencoder to get reconstruction
        reconstruction = autoencoder(x)

        # Process input for target
        target = sigmoid(autoencoder.feature_module(x))

        # Error between reconstruction and input
        error = AE_loss(target=target, reconstruction=reconstruction)

        return error

    def forward(self, x):

        # If not in training mode, select the best expert for the input data
        if (not self.training):

            # Build an error tensor to hold errors for all autoencoders
            all_errors = [None]*len(self.autoencoder_dict)

            # Iterate through all autoencoders to populate error tensor
            for autoencoder_id in self.autoencoder_dict:
                error = self._get_average_reconstruction_error(
                    autoencoder_id, x)
                error = -error/self.temp
                all_errors[int(autoencoder_id)] = torch.tensor(error)

            # Softmax to get probabilites
            probabilities = softmax(torch.Tensor(all_errors), dim=-1)

            # Select an expert for this input using the most likely autoencoder
            most_relevant_expert_key = torch.argmax(probabilities)
            self.expert = self.expert_dict[str(most_relevant_expert_key.item())]

        x = x.to(self.device)
        self.expert = self.expert.to(self.device)

        return self.expert(x)
