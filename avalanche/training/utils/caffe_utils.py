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
General useful functions for caffe
"""

from caffe.proto import caffe_pb2
from google.protobuf import text_format
from avalanche.constants import HWBackEnd
import numpy as np

def create_solver(
    net,
    base_lr,
    random_seed=0,
    lr_policy="step",
    gamma=0.1,
    stepsize=100000000,
    momentum=0.9,
    weight_decay=0.0005,
    test_iter=0,
    test_interval=1000,
    display=20,
    max_iter=None,
    snapshot=None,
    snapshot_prefix=None,
    regularization_type=None,
    momentum2=None,
    type=None,
    solver_mode=HWBackEnd.GPU
):

    solver_config = caffe_pb2.SolverParameter()

    # Optional Parameters
    if max_iter is not None:
        solver_config.max_iter = max_iter

    if random_seed is not None:
        solver_config.random_seed = random_seed

    if snapshot is not None:
        solver_config.snapshot = snapshot

    if snapshot_prefix is not None:
        solver_config.snapshot_prefix = snapshot_prefix

    if regularization_type is not None:
        solver_config.regularization_type = regularization_type

    if momentum2 is not None:
        solver_config.momentum2 = momentum2

    if display is not None:
        solver_config.display = display

    if type is not None:
        solver_config.type = type

    # Set defaults in case of None Values
    if test_iter is None:
        solver_config.test_iter.append(1)
    else:
        solver_config.test_iter.append(test_iter)

    if test_interval is None:
        solver_config.test_interval = 1
    else:
        solver_config.test_interval = test_interval

    # Other parameters
    solver_config.net = net
    solver_config.base_lr = base_lr
    solver_config.lr_policy = lr_policy
    solver_config.gamma = gamma
    solver_config.stepsize = stepsize
    solver_config.momentum = momentum
    solver_config.weight_decay = weight_decay
    solver_config.snapshot_format = caffe_pb2.SolverParameter.HDF5

    if solver_mode is HWBackEnd.GPU:
        solver_mode = caffe_pb2.SolverParameter.GPU
    else:
        solver_mode = caffe_pb2.SolverParameter.CPU
    solver_config.solver_mode = solver_mode

    # Casting to String
    solver_config = text_format.MessageToString(
        solver_config, float_format='.6g'
    )

    return solver_config


def get_classifier(model, clas=None):
    """ Change weights and biases of the last layer in the network. """

    if clas is not None:
        weigth = model.params['out'][0].data[clas]
        bias = model.params['out'][1].data[clas]
    else:
        weigth = model.params['out'][0].data[...]
        bias = model.params['out'][1].data[...]

    return weigth, bias


def set_classifier(model, weigth, bias, clas=None):
    """ Change weights and biases of the last layer in the network. """

    # WARNING: train-test net shares weights
    # It seems train and test weights are shared if you change them from
    # python, so it is like having a single net (remove test net at this
    #  point)!

    if clas is not None:
        model.params['out'][0].data[clas] = weigth
        model.params['out'][1].data[clas] = bias
    else:
        model.params['out'][0].data[...] = weigth
        model.params['out'][1].data[...] = bias


def reset_classifier(model, val=0, std=None):
    """ Set weights and biases of the last layer in the network to zero. """

    weights = np.zeros_like(model.params['out'][0].data)
    biases = np.zeros_like(model.params['out'][1].data)

    if std:
        # print(val, std, opt.net.params['out'][0].data.shape)
        weights = np.random.normal(
            val, std, model.net.params['out'][0].data.shape
        )
    else:
        weights.fill(val)

    biases.fill(0)
    set_classifier(model, weights, biases)


def compute_one_hot(train_y, class_count):
    """ Compute one-hot from labels. """

    target_y = np.zeros((train_y.shape[0], class_count), dtype=np.float32)
    target_y[np.arange(train_y.shape[0]), train_y.astype(np.int8)] = 1

    return target_y


def preprocess_image(img):
    """ Pre-process images like caffe. It may be need adjustements depending
        on the pre-trained model since it is training dependent. """

    # we assume img is a 3-channel image loaded with plt.imread
    # so img has dim (w, h, c)
    # scale each pixel to 255
    img = img * 255
    # Swap RGB to BRG
    img = img[:, :, ::-1]
    # Subtract channel average
    img[:, :, 0] -= 104
    img[:, :, 1] -= 117
    img[:, :, 2] -= 123
    # Swap channel dimension to fit the caffe format (c, w, h)
    img = np.transpose(img, (2, 0, 1))
    return img


