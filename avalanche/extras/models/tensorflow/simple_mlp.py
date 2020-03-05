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

This is the definition of a simple MLP in Tensorflow

"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf


class SimpleMLP(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(SimpleMLP, self).__init__()

        self.features = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.1)
        ])

        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    kwargs = {'num_classes': 10}
    print(SimpleMLP(**kwargs))
