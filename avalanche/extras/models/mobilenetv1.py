#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco. All rights reserved.                  #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 7-12-2017                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""

This is the definition od the Mid-caffenet high resolution in Pythorch

"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch.nn as nn
import torch
from avalanche.extras.models.pytorchcv.mobilenet import mobilenet_w1
from avalanche.extras.models.pytorchcv.common import DwsConvBlock

def remove_sequential(network, all_layers):

    for layer in network.children():
        if isinstance(layer, nn.Sequential): # if sequential layer, apply recursively to layers in sequential layer
            #print(layer)
            remove_sequential(layer, all_layers)
        else: # if leaf node, add it to list
            # print(layer)
            all_layers.append(layer)

def remove_DwsConvBlock(cur_layers):

    all_layers = []
    for layer in cur_layers:
        if isinstance(layer, DwsConvBlock):
           #  print("helloooo: ", layer)
            for ch in layer.children():
                all_layers.append(ch)
        else:
            all_layers.append(layer)
    return all_layers

class MobilenetV1(nn.Module):
    def __init__(self, pretrained=True, latent_layer_num=20):
        super().__init__()

        model = mobilenet_w1(pretrained=pretrained)
        model.features.final_pool = nn.AvgPool2d(4)

        all_layers = []
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
