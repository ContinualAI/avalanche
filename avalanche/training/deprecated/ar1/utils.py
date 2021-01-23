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
General useful functions for machine learning with Pytorch.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import torch
from avalanche.models.batch_renorm import BatchRenorm2D


def replace_bn_with_brn(
        m, name="", momentum=0.1, r_d_max_inc_step=0.0001, r_max=1.0,
        d_max=0.0, max_r_max=3.0, max_d_max=5.0):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            # print('replaced: ', name, attr_str)
            setattr(m, attr_str,
                    BatchRenorm2D(
                        target_attr.num_features,
                        gamma=target_attr.weight,
                        beta=target_attr.bias,
                        running_mean=target_attr.running_mean,
                        running_var=target_attr.running_var,
                        eps=target_attr.eps,
                        momentum=momentum,
                        r_d_max_inc_step=r_d_max_inc_step,
                        r_max=r_max,
                        d_max=d_max,
                        max_r_max=max_r_max,
                        max_d_max=max_d_max
                        )
                    )
    for n, ch in m.named_children():
        replace_bn_with_brn(ch, n, momentum, r_d_max_inc_step, r_max, d_max,
                            max_r_max, max_d_max)


def change_brn_pars(
        m, name="", momentum=0.1, r_d_max_inc_step=0.0001, r_max=1.0,
        d_max=0.0):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == BatchRenorm2D:
            target_attr.momentum = torch.tensor((momentum), requires_grad=False)
            target_attr.r_max = torch.tensor(r_max, requires_grad=False)
            target_attr.d_max = torch.tensor(d_max, requires_grad=False)
            target_attr.r_d_max_inc_step = r_d_max_inc_step

    for n, ch in m.named_children():
        change_brn_pars(ch, n, momentum, r_d_max_inc_step, r_max, d_max)


def consolidate_weights(model, cur_clas):
    """ Mean-shift for the target layer weights"""

    with torch.no_grad():

        globavg = np.average(model.output.weight.detach()
                             .cpu().numpy()[cur_clas])
        for c in cur_clas:
            w = model.output.weight.detach().cpu().numpy()[c]

            if c in cur_clas:
                new_w = w - globavg
                if c in model.saved_weights.keys():
                    wpast_j = np.sqrt(model.past_j[c] / model.cur_j[c])
                    # wpast_j = model.past_j[c] / model.cur_j[c]
                    model.saved_weights[c] = (model.saved_weights[c] * wpast_j
                                              + new_w) / (wpast_j + 1)
                else:
                    model.saved_weights[c] = new_w


def set_consolidate_weights(model):
    """ set trained weights """

    with torch.no_grad():
        for c, w in model.saved_weights.items():
            model.output.weight[c].copy_(
                torch.from_numpy(model.saved_weights[c])
            )


def reset_weights(model, cur_clas):
    """ reset weights"""

    with torch.no_grad():
        model.output.weight.fill_(0.0)
        # model.output.weight.copy_(
        #     torch.zeros(model.output.weight.size())
        # )
        for c, w in model.saved_weights.items():
            if c in cur_clas:
                model.output.weight[c].copy_(
                    torch.from_numpy(model.saved_weights[c])
                )


def examples_per_class(train_y):
    count = {i: 0 for i in range(50)}
    for y in train_y:
        count[int(y)] += 1

    return count


def set_brn_to_train(m, name=""):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == BatchRenorm2D:
            target_attr.train()
            # print("setting to train..")
    for n, ch in m.named_children():
        set_brn_to_train(ch, n)


def set_brn_to_eval(m, name=""):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == BatchRenorm2D:
            target_attr.eval()
            # print("setting to train..")
    for n, ch in m.named_children():
        set_brn_to_train(ch, n)


def freeze_up_to(model, freeze_below_layer):
    for name, param in model.named_parameters():
        # tells whether we want to use gradients for a given parameter
        if "bn" not in name:
            param.requires_grad = False
            print("Freezing parameter " + name)
        if name == freeze_below_layer:
            break


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
        weights_vector = None
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
        grad_vector = None
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


if __name__ == "__main__":
    from avalanche.models import MobilenetV1
    model = MobilenetV1(pretrained=True)
    replace_bn_with_brn(model, "net")

    ewcData, synData = create_syn_data(model)
    extract_weights(model, ewcData[0])
