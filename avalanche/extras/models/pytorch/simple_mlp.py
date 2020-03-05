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


class SimpleMLP(nn.Module):

    def __init__(self, num_classes=10):
        super(SimpleMLP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), 28 * 28)
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    kwargs = {'num_classes': 10}
    print(SimpleMLP(**kwargs))
