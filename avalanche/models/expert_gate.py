import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from .utils import FeatureExtractorBackbone
from avalanche.models import MultiTaskModule, DynamicModule
from avalanche.models import MultiHeadClassifier

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

class ExpertGate(MultiTaskModule):
    def __init__(
        arch="alexnet",
        pretrained_flag=True,
        device="cpu",
        output_layer_name=None,
    ):
        super().__init__()

    
        # Select the pre-trained backbone to extract features from (defaults to arch=AlexNet)
        feature_extractor_model = (
            models.__dict__[arch](pretrained=pretrained_flag)
            .to(device)
            .eval()
        )

        # Module to extract features from given backbone and layer
        self.feature_extraction_wrapper = FeatureExtractorBackbone(
            feature_extractor_model, output_layer_name
        ).eval()

    def adaptation(self, dataset):
        super().adaptation(dataset)
        # your adaptation goes here

    def forward_single_task(self, x, task_label):
        # your forward goes here.
        # task_label is a single integer
        # the mini-batch is split by task-id inside the forward method.
        pass