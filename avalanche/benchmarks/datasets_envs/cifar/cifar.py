#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco. All rights reserved.                  #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-04-2017                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Different utils which can be used for CIFAR10 and CIFAR100 """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os
import glob
import pickle as pkl
from PIL import Image

from avalanche.benchmarks.utils import remove_some_labels
from avalanche.training.utils import shuffle_in_unison

def read_data_from_png(params):
    """ Extract the images and pickle them. Values are rescaled from [0, 255]
    down to [-0.5, 0.5]. If the number of imgs is too high this function flush
    it in batches. It returns the final lists of pixels, qualities and labels.
    """

    for dir in ['train', 'test']:
        if os.path.isfile(params['dpath'] + dir + '.npz'):
            with open(params['dpath'] + dir + '.npz', 'rb') as f:
                npzfile = np.load(f)
                if dir == 'train':
                    train_data, train_labels = npzfile['imgs'], \
                                               npzfile['labels']
                else:
                    test_data, test_labels = npzfile['imgs'], \
                                             npzfile['labels']
        else:
            print('Extracting data from: ', params['dpath'] + dir)

            # matrices initialization
            glob_files = params['dpath'] + dir + "/*/*"
            imgs = np.empty(shape=(params[dir + '_size'],
                                   params['img_size'] * params['img_size'] * 3),
                            dtype=np.float32)
            labels = np.empty(shape=(params[dir + '_size']),
                              dtype=np.int8)

            imgs_proc = 0
            for i, file_path in enumerate(
                    sorted(glob.glob(glob_files))):
                # print("filename:", file_path)

                tail, name = os.path.split(file_path)
                tail, class_name = os.path.split(tail)
                clas = int(class_name)

                img = Image.open(file_path)
                pixels_ch0 = [f[0] for f in list(img.getdata())]
                pixels_ch1 = [f[1] for f in list(img.getdata())]
                pixels_ch2 = [f[2] for f in list(img.getdata())]

                pixels_ch0[:] = [(pixel - (params['pixel_depth'] / 2.0)) /
                                 params['pixel_depth'] for pixel in pixels_ch0]
                pixels_ch1[:] = [(pixel - (params['pixel_depth'] / 2.0)) /
                                 params['pixel_depth'] for pixel in pixels_ch1]
                pixels_ch2[:] = [(pixel - (params['pixel_depth'] / 2.0)) /
                                 params['pixel_depth'] for pixel in pixels_ch2]

                imgs[i] = np.asarray(pixels_ch0 + pixels_ch1 + pixels_ch2,
                                     dtype=np.float32)

                labels[i] = clas
                imgs_proc += 1

                print("num imgs processed:", imgs_proc, end="\r")

            imgs, labels = shuffle_in_unison([imgs, labels],
                                             params['seed'])
            imgs = np.asarray(imgs, np.float32)
            labels = np.asarray(labels, np.int64)
            imgs = imgs.reshape(imgs_proc, params['img_size'],
                                params['img_size'], 3)

            with open(params['dpath'] + dir + '.npz', 'wb') as f:
                np.savez(f, imgs=imgs, labels=labels)

            if dir == 'train':
                train_data = imgs
                train_labels = labels
            else:
                test_data = imgs
                test_labels = labels

    return [(train_data, train_labels), (test_data, test_labels)]


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pkl.load(fo, encoding='bytes')
    return dict


def read_data_from_pickled(dpath, train_size, test_size, img_size):
    if '100' not in dpath:
        for i in range(1, 6):
            batch = unpickle(dpath + 'data_batch_' + str(i))
            if i == 1:
                train_data = batch[b'data']
                train_labels = batch[b'labels']
            else:
                train_labels = np.concatenate((train_labels, batch[b'labels']))
                train_data = np.vstack((train_data, batch[b'data']))

        test_name = 'test_batch'
        batch = unpickle(dpath + test_name)
        test_data = batch[b'data']
        test_labels = np.asarray(batch[b'labels'])

    else:
        batch = unpickle(dpath + 'train')
        train_data = batch[b'data']
        train_labels = np.asarray(batch[b'fine_labels'])
        test_name = 'test'

        batch = unpickle(dpath + test_name)
        test_data = batch[b'data']
        test_labels = np.asarray(batch[b'fine_labels'])

    train_data = train_data.reshape(train_size, 3, img_size, img_size)
    test_data = test_data.reshape(test_size, 3, img_size, img_size)
    train_data = np.transpose(train_data, (0, 2, 3, 1))
    test_data = np.transpose(test_data, (0, 2, 3, 1))

    return [(train_data, train_labels), (test_data, test_labels)]


def get_merged_cifar10_and_100(cifar10_root, cifar100_root):

    # load cifar10 images
    cifar10_train, cifar10_test = read_data_from_pickled(
        cifar10_root, 50000, 10000, 32
    )
    print("Cifar10 loaded.")

    # load cifar100 images
    cifar100_train, cifar100_test = read_data_from_pickled(
        cifar100_root, 50000, 10000, 32
    )
    print("Cifar100 loaded.")

    # shifting cifar100 labels (starting from 10)
    offset = 10
    for i, label in enumerate(cifar100_train[1]):
        cifar100_train[1][i] = label + offset

    for i, label in enumerate(cifar100_test[1]):
        cifar100_test[1][i] = label + offset

    train_data = np.vstack((cifar10_train[0], cifar100_train[0]))
    train_labels = np.concatenate((cifar10_train[1], cifar100_train[1]))

    test_data = np.vstack((cifar10_test[0], cifar100_test[0]))
    test_labels = np.concatenate((cifar10_test[1], cifar100_test[1]))

    return [(train_data, train_labels), (test_data, test_labels)]


if __name__ == '__main__':
    pixel_depth = 255
    seed = 0

    train_set, test_set = get_merged_cifar10_and_100(
        '/home/admin/data/cifar10/cifar-10-batches-py/',
         '/home/admin/data/cifar100/cifar-100-python/'
    )

    # split at convenience
    train_set = remove_some_labels(train_set, [0, 1, 2])

    # normalize
    train_data, train_labels = train_set
    test_data, test_labels = test_set
    shuffle_in_unison([train_data, train_labels], seed)
    test_data = (test_data - (pixel_depth // 2.0)) / pixel_depth
    train_data = (train_data - (pixel_depth // 2.0)) / pixel_depth

    print('Done.')
