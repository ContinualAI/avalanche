import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from .utils import FeatureExtractorBackbone
def AE_loss(input, reconstruction):
    loss_method = MSELoss(reduction="sum")
    reconstruction_loss = loss_method(input, reconstruction)
    return reconstruction_loss


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Encoder Linear -> ReLU
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder Linear -> Sigmoid
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # Reconstruct input
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction


class ExpertModel(nn.Module):
    def __init__(self, arch, num_classes, pretrained_flag=True, ):
        super().__init__()

        # Select pretrained model
        self.model = models.__dict__[arch](pretrained=pretrained_flag)

        # Replace the final layer to work for the task
        original_classifier_input_dim = self.model._modules['classifier'][-1].in_features
        self.model._modules['classifier'][-1] = nn.Linear(
            original_classifier_input_dim, num_classes)

    def forward(self, x):
        return self.model(x)


class ExpertGate(MultiTaskModule):
    def __init__(
        self,
        num_classes,
        arch="alexnet",
        pretrained_flag=True,
        device="cpu",
        output_layer_name="feature",
    ):
        super().__init__()

        # Store variables
        self.arch = arch
        self.num_classes = num_classes

        # Select the pre-trained backbone to extract features from 
        # (defaults to arch=AlexNet)
        feature_extractor_model = (
            models.__dict__[arch](pretrained=pretrained_flag)
            .to(device)
            .eval()
        )

        # Module to extract features from given backbone and layer
        self.feature_extraction_wrapper = FeatureExtractorBackbone(
            feature_extractor_model, output_layer_name
        ).eval()

        # Dict for autoencoders
        # {task, autoencoder}
        self.autoencoder_dict = nn.ModuleDict()

        # Dict for experts
        # {task, expert}
        self.expert_dict = nn.ModuleDict()

    def add_autoencoder(self, task_num, input_dim, latent_dim=100):
        # Build a new autoencoder
        new_autoencoder = Autoencoder(
            input_dim=input_dim, latent_dim=latent_dim)

        # Store autoencoder with task number
        self.autoencoder_dict.update({task_num, new_autoencoder})

    def add_expert(self, task_num):
        # Build a new expert
        new_expert = ExpertModel(arch=self.arch, num_classes=self.num_classes)

        # Store expert with task number
        self.expert_dict.update({task_num, new_expert})

    def forward_single_task(self, x, task_label):
        # your forward goes here.
        # task_label is a single integer
        # the mini-batch is split by task-id inside the forward method.
        pass
