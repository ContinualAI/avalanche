#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

"""
General utils for the synaptic intelligence algorithm.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import torch

def create_syn_data(model):
    size = 0
    print('Creating Syn data for Optimal params and their Fisher info')

    for name, param in model.named_parameters():
        if "bn" not in name and "output" not in name:
            print(name, param.flatten().size(0))
            size += param.flatten().size(0)

    # The first array returned is a 2D array: the first component contains
    # the params at loss minimum, the second the parameter importance
    # The second array is a dictionary with the synData
    synData = {}
    synData['old_theta'] = torch.zeros(size, dtype=torch.float32)
    synData['new_theta'] = torch.zeros(size, dtype=torch.float32)
    synData['grad'] = torch.zeros(size, dtype=torch.float32)
    synData['trajectory'] = torch.zeros(size, dtype=torch.float32)
    synData['cum_trajectory'] = torch.zeros(size, dtype=torch.float32)

    return torch.zeros((2, size), dtype=torch.float32), synData


def extract_weights(model, target):

    with torch.no_grad():
        weights_vector= None
        for name, param in model.named_parameters():
            if "bn" not in name and "output" not in name:
                # print(name, param.flatten())
                if weights_vector is None:
                    weights_vector = param.flatten()
                else:
                    weights_vector = torch.cat(
                        (weights_vector, param.flatten()), 0)

        target[...] = weights_vector.cpu()


def extract_grad(model, target):
    # Store the gradients into target
    with torch.no_grad():
        grad_vector= None
        for name, param in model.named_parameters():
            if "bn" not in name and "output" not in name:
                # print(name, param.flatten())
                if grad_vector is None:
                    grad_vector = param.grad.flatten()
                else:
                    grad_vector = torch.cat(
                        (grad_vector, param.grad.flatten()), 0)

        target[...] = grad_vector.cpu()


def init_batch(net, ewcData, synData):
    extract_weights(net, ewcData[0])  # Keep initial weights
    synData['trajectory'] = 0


def pre_update(net, synData):
    extract_weights(net, synData['old_theta'])


def post_update(net, synData):
    extract_weights(net, synData['new_theta'])
    extract_grad(net, synData['grad'])

    synData['trajectory'] += synData['grad'] * (
                    synData['new_theta'] - synData['old_theta'])


def update_ewc_data(net, ewcData, synData, clip_to, c=0.0015):
    extract_weights(net, synData['new_theta'])
    eps = 0.0000001  # 0.001 in few task - 0.1 used in a more complex setup

    synData['cum_trajectory'] += c * synData['trajectory'] / (
                    np.square(synData['new_theta'] - ewcData[0]) + eps)

    ewcData[1] = torch.empty_like(synData['cum_trajectory'])\
        .copy_(-synData['cum_trajectory'])
    # change sign here because the Ewc regularization
    # in Caffe (theta - thetaold) is inverted w.r.t. syn equation [4]
    # (thetaold - theta)
    ewcData[1] = torch.clamp(ewcData[1], max=clip_to)
    # (except CWR)
    ewcData[0] = synData['new_theta'].clone().detach()


def compute_ewc_loss(model, ewcData, lambd=0, device=None):

    weights_vector = None
    for name, param in model.named_parameters():
        if "bn" not in name and "output" not in name:
            # print(name, param.flatten())
            if weights_vector is None:
                weights_vector = param.flatten()
            else:
                weights_vector = torch.cat(
                    (weights_vector, param.flatten()), 0)

    ewcData = ewcData.to(device)
    loss = (lambd / 2) * torch.dot(ewcData[1], (weights_vector - ewcData[0])**2)
    return loss


