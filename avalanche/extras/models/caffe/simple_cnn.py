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
This is the definition of a simple MLP for Caffe
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import caffe
from caffe import layers as L
from caffe import params as P


class SimpleCNN(object):

    def __init__(self, num_classes=10):

        self.num_classes = num_classes

    def get_proto(self):
        """ This net can be used for whatever RGB image """

        net = caffe.NetSpec()

        # Input Layer with Input
        net.data = L.Input(
            shape=[dict(dim=[256, 3, 128, 128])], ntop=1,
            include=dict(phase=caffe.TRAIN)
        )
        net.test_data = L.Input(
            shape=[dict(dim=[100, 3, 128, 128])], ntop=1,
            include=dict(phase=caffe.TEST)
        )
        net.label = L.Input(
            shape=[dict(dim=[256])], ntop=1,
            include=dict(phase=caffe.TRAIN)
        )
        net.test_label = L.Input(
            shape=[dict(dim=[100])], ntop=1,
            include=dict(phase=caffe.TEST)
        )
        net.target = L.Input(
            shape=[dict(dim=[256, self.num_classes])], ntop=1,
            include=dict(phase=caffe.TRAIN)
        )
        # Processing Layers
        net.conv1 = L.Convolution(
            net.data, kernel_size=3, stride=2,
            num_output=96, param=[dict(lr_mult=1), dict(lr_mult=2)],
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0)
        )
        net.relu1 = L.ReLU(net.conv1, in_place=True)
        net.conv2 = L.Convolution(
            net.relu1, kernel_size=3, stride=2,
            num_output=120, param=[dict(lr_mult=1), dict(lr_mult=2)],
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0)
        )
        net.relu2 = L.ReLU(net.conv2, in_place=True)
        net.conv3 = L.Convolution(
            net.relu2, kernel_size=3, stride=2,
            num_output=60, param=[dict(lr_mult=1), dict(lr_mult=2)],
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0)
        )
        net.relu3 = L.ReLU(net.conv3, in_place=True)
        net.conv4 = L.Convolution(
            net.relu3, kernel_size=3, stride=2,
            num_output=self.num_classes,
            param=[dict(lr_mult=1), dict(lr_mult=2)],
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0)
        )
        net.relu4 = L.ReLU(net.conv4, in_place=True)
        net.out = L.Pooling(
            net.relu4, pooling_param=dict(
                pool=P.Pooling.AVE, global_pooling=True
            )
        )

        # Accuracy and Loss
        net.accuracy = L.Accuracy(net.out, net.test_label)
        net.softmax = L.Softmax(net.out, include=dict(phase=caffe.TRAIN))
        net.loss = L.MultinomialLogisticLossTarget(
            net.softmax, net.target, include=dict(phase=caffe.TRAIN)
        )

        # convert to string and fix the test data layer
        proto = str(net.to_proto())
        proto = proto.replace('test_data', 'data')\
            .replace('test_label', 'label')

        # returning the proto
        return proto


if __name__ == "__main__":

    kwargs = {'num_classes': 10}
    print(SimpleCNN(**kwargs).get_proto())
