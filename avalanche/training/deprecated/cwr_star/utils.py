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


def consolidate_weights(model, cur_clas):
    """ Mean-shift for the target layer weights"""

    with torch.no_grad():

        globavg = np.average(model.classifier.weight.detach()
                             .cpu().numpy()[cur_clas])
        for c in cur_clas:
            w = model.classifier.weight.detach().cpu().numpy()[c]

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
            model.classifier.weight[c].copy_(
                torch.from_numpy(model.saved_weights[c])
            )


def reset_weights(model, cur_clas):
    """ reset weights"""

    with torch.no_grad():
        model.classifier.weight.fill_(0.0)
        # model.classifier.weight.copy_(
        #     torch.zeros(model.classifier.weight.size())
        # )
        for c, w in model.saved_weights.items():
            if c in cur_clas:
                model.classifier.weight[c].copy_(
                    torch.from_numpy(model.saved_weights[c])
                )


def examples_per_class(train_y):
    count = {i: 0 for i in range(50)}
    for y in train_y:
        count[int(y)] += 1

    return count


def freeze_up_to(model, freeze_below_layer):
    for name, param in model.named_parameters():
        # tells whether we want to use gradients for a given parameter
        param.requires_grad = False
        print("Freezing parameter " + name)
        if name == freeze_below_layer:
            break
