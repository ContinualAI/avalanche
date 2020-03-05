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


class SimpleMLP(object):

    def __init__(self, hidden_units=512, num_classes=10,
                ewc_input=False, tot_num_params=5592010):

        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.ewc_input = ewc_input
        self.tot_num_params = tot_num_params

    def get_proto(self):
        """ This net can be used in the Permuted MNIST experiments for
            Continual Learning. """

        net = caffe.NetSpec()

        # Input Layers
        net.data = L.Input(
            shape=[dict(dim=[256, 1, 28, 28, ])], ntop=1,
            include=dict(phase=caffe.TRAIN)
        )
        net.test_data = L.Input(
            shape=[dict(dim=[100, 1, 28, 28, ])], ntop=1,
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
            shape=[dict(dim=[256, 10])], ntop=1,
            include=dict(phase=caffe.TRAIN)
        )
        if self.ewc_input:
            net.ewc = L.Input(
                shape=[dict(dim=[2, self.tot_num_params])], ntop=1,
                include=dict(phase=caffe.TRAIN)
            )

        # Processing Layers
        if self.ewc_input:
            net.ewc_silence = L.Silence(
                net.ewc, include=dict(phase=caffe.TRAIN), ntop=0
            )

        net.fc1 = L.InnerProduct(
            net.data, num_output=self.hidden_units,
            param=[dict(lr_mult=1), dict(lr_mult=2)],
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0)
        )
        net.relu1 = L.ReLU(net.fc1, in_place=True)
        net.drop1 = L.Dropout(net.relu1, dropout_ratio=0.5, in_place=True)
        net.fc2 = L.InnerProduct(
            net.drop1, num_output=self.hidden_units,
            param=[dict(lr_mult=1), dict(lr_mult=2)],
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=1)
        )
        net.relu2 = L.ReLU(net.fc2, in_place=True)
        net.drop2 = L.Dropout(net.relu2, dropout_ratio=0.5, in_place=True)
        net.out = L.InnerProduct(
            net.drop2, num_output=self.num_classes,
            param=[dict(lr_mult=1), dict(lr_mult=2)],
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0)
        )
        # Accuracy and Loss
        net.accuracy = L.Accuracy(net.out, net.test_label)
        net.softmax = L.Softmax(net.out, include=dict(phase=caffe.TRAIN))
        net.loss = L.MultinomialLogisticLossTarget(
            net.softmax, net.target, include=dict(phase=caffe.TRAIN)
        )

        # convert to string and fix the test data layer
        proto = str(net.to_proto())
        proto = proto.replace('test_data', 'data').replace('test_label',
                                                           'label')

        # returning the proto
        return proto


if __name__ == "__main__":

    kwargs = {'num_classes': 10}
    print(SimpleMLP(**kwargs).get_proto())
