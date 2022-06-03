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
    def __init__(self, shape, latent_dim):
        super().__init__()
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

        self.invTrans = transforms.Compose(
                [transforms.Normalize((0.1307,), (0.3081,))]
                )

    def forward(self, x):
        # Reconstruct input
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)

        return self.invTrans(reconstruction.view(-1, *self.shape))


class ExpertModel(nn.Module):
    def __init__(self, num_classes, arch, device, pretrained_flag, feature_template=None):
        super().__init__()

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
            original_classifier_input_dim, num_classes)

    def forward(self, x):
        x = self.feature_module(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier_module(x)
        return x


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
