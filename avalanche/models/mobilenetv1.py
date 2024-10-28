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

from typing import List
import torch.nn as nn
import torch

from pytorchcv.models.mobilenet import mobilenet_w1

try:
    from pytorchcv.models.mobilenet import DwsConvBlock
except ImportError:
    try:
        from pytorchcv.models.common import DwsConvBlock
    except ImportError:
        # pytorchcv >= 0.0.68
        from pytorchcv.models.common.conv import DwsConvBlock


def remove_sequential(network: nn.Module, all_layers: List[nn.Module]):
    for layer in network.children():
        # if sequential layer, apply recursively to layers in sequential layer
        if isinstance(layer, nn.Sequential):
            # print(layer)
            remove_sequential(layer, all_layers)
        else:  # if leaf node, add it to list
            # print(layer)
            all_layers.append(layer)


def remove_DwsConvBlock(cur_layers):
    all_layers = []
    for layer in cur_layers:
        if isinstance(layer, DwsConvBlock):
            # print("helloooo: ", layer)
            for ch in layer.children():
                all_layers.append(ch)
        else:
            all_layers.append(layer)
    return all_layers


class MobilenetV1(nn.Module):
    """MobileNet v1 implementation. This model
    can be instantiated from a pretrained network."""

    def __init__(self, pretrained=True, latent_layer_num=20):
        super().__init__()

        model = mobilenet_w1(pretrained=pretrained)
        model.features.final_pool = nn.AvgPool2d(4)

        all_layers: List[nn.Module] = []
        remove_sequential(model, all_layers)
        all_layers = remove_DwsConvBlock(all_layers)

        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers[:-1]):
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

        self.output = nn.Linear(1024, 50, bias=False)

    def forward(self, x, latent_input=None, return_lat_acts=False):
        if latent_input is not None:
            with torch.no_grad():
                orig_acts = self.lat_features(x)
            lat_acts = torch.cat((orig_acts, latent_input), 0)
        else:
            orig_acts = self.lat_features(x)
            lat_acts = orig_acts

        x = self.end_features(lat_acts)
        x = x.view(x.size(0), -1)
        logits = self.output(x)

        if return_lat_acts:
            return logits, orig_acts
        else:
            return logits


if __name__ == "__main__":
    model = MobilenetV1(pretrained=True)
    for name, param in model.named_parameters():
        print(name)
