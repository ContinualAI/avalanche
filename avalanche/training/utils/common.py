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

General useful functions.

"""
# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import random as rn
from copy import deepcopy


def shuffle_in_unison(dataset, seed, in_place=False):
    """ Shuffle two (or more) list in unison. """

    np.random.seed(seed)
    rng_state = np.random.get_state()
    new_dataset = []
    for x in dataset:
        if in_place:
            np.random.shuffle(x)
        else:
            new_dataset.append(np.random.permutation(x))
        np.random.set_state(rng_state)

    if not in_place:
        return new_dataset


def softmax(x):
    """ Compute softmax values for each sets of scores in x. """

    f = x - np.max(x)
    return np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)
    # If you do not care about stability use line above:
    # return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def count_lines(fpath):
    """ Count line in file. """

    num_imgs = 0
    with open(fpath, 'r') as f:
        for line in f:
            if '/' in line:
                num_imgs += 1
    return num_imgs


def pad_data(dataset, mb_size):
    """ Padding all the matrices contained in dataset to suit the mini-batch
        size. We assume they have the same shape. """

    num_set = len(dataset)
    x = dataset[0]
    # computing test_iters
    n_missing = x.shape[0] % mb_size
    if n_missing > 0:
        surplus = 1
    else:
        surplus = 0
    it = x.shape[0] // mb_size + surplus

    # padding data to fix batch dimentions
    if n_missing > 0:
        n_to_add = mb_size - n_missing
        for i, data in enumerate(dataset):
            dataset[i] = np.concatenate((data[:n_to_add], data))
    if num_set == 1:
        dataset = dataset[0]

    return dataset, it


def compute_one_hot(train_y, class_count):
    """ Compute one-hot from labels. """

    target_y = np.zeros((train_y.shape[0], class_count), dtype=np.float32)
    target_y[np.arange(train_y.shape[0]), train_y.astype(np.int8)] = 1

    return target_y


def imagenet_batch_preproc(img_batch, rgb_swap=True, channel_first=True,
                        avg_sub=True):
    """ Pre-process batch of PIL img for Imagenet pre-trained models with caffe.
        It may be need adjustements depending on the pre-trained model
        since it is training dependent. """

    # we assume img is a 3-channel image loaded with PIL
    # so img has dim (w, h, c)

    if rgb_swap:
        # Swap RGB to BRG
        img_batch = img_batch[:, :, :, ::-1]

    if avg_sub:
        # Subtract channel average
        img_batch[:, :, :, 0] -= 104
        img_batch[:, :, :, 1] -= 117
        img_batch[:, :, :, 2] -= 123

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch


