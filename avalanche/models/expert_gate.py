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
