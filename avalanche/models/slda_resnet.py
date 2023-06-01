from typing import Union
import torch
import torch.nn as nn
import torchvision.models as models
from .utils import FeatureExtractorBackbone


class SLDAResNetModel(nn.Module):
    """
    This is a model wrapper to reproduce experiments from the original
    paper of Deep Streaming Linear Discriminant Analysis by using
    a pretrained ResNet model.
    """

    def __init__(
        self,
        arch="resnet18",
        output_layer_name="layer4.1",
        imagenet_pretrained=True,
        device: Union[str, torch.device] = "cpu",
    ):
        """Init.

        :param arch: backbone architecture. Default is resnet-18, but others
            can be used by modifying layer for
            feature extraction in ``self.feature_extraction_wrapper``.
        :param imagenet_pretrained: True if initializing backbone with imagenet
            pre-trained weights else False
        :param output_layer_name: name of the layer from feature extractor
        :param device: cpu, gpu or other device
        """

        super(SLDAResNetModel, self).__init__()

        feat_extractor = (
            models.__dict__[arch](pretrained=imagenet_pretrained).to(device).eval()
        )
        self.feature_extraction_wrapper = FeatureExtractorBackbone(
            feat_extractor, output_layer_name
        ).eval()

    @staticmethod
    def pool_feat(features):
        feat_size = features.shape[-1]
        num_channels = features.shape[1]
        features2 = features.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x
        # num_channels
        features3 = torch.reshape(
            features2, (features.shape[0], feat_size * feat_size, num_channels)
        )
        feat = features3.mean(1)  # mb x num_channels
        return feat

    def forward(self, x):
        """
        :param x: raw x data
        """
        feat = self.feature_extraction_wrapper(x)
        feat = SLDAResNetModel.pool_feat(feat)
        return feat
