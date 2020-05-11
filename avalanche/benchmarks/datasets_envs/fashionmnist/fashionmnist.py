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

""" Different utils which can be used for fashionMNIST """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import gzip
import pickle
from copy import deepcopy
from PIL import Image
import os
import matplotlib.pyplot as plt
import sys

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


class fashionMNIST(object):

    """MNIST static dataset and basic utilities"""

    def __init__(self, data_loc='data/'):

        if os.path.isabs(data_loc):
            path = data_loc
        else:
            path = os.path.join(os.path.dirname(__file__), data_loc)

        self.data_loc = path
        self.train_set = None
        self.test_set = None

        try:
            # Create target Directory for fashionMNIST data
            os.mkdir(self.data_loc)
            print("Directory ", self.data_loc, " Created ")
            self.download_fmnist()
            self.save_fmnist()
            self.load()

        except OSError:
            print("Directory ", self.data_loc, " already exists")
            self.load()

    def load(self):

        with open(self.data_loc + "fmnist.pkl", 'rb') as f:
            fmnist = pickle.load(f)

        self.train_set = [fmnist["training_images"], fmnist["training_labels"]]
        self.test_set = [fmnist["test_images"], fmnist["test_labels"]]

    def get_data(self):

        return [self.train_set, self.test_set]

    def permute_fmnist(self, seed):
        """
        Given the train and test set (no labels), permute pixels of each img
        the same way.
        """

        # we take only the images
        fmnist_imgs = [self.train_set[0], self.test_set[0]]

        np.random.seed(seed)
        print("starting permutation...")
        # print(fmnist_imgs[0].shape)
        h, w = fmnist_imgs[0].shape[2], fmnist_imgs[0].shape[3]
        perm_inds = list(range(h*w))
        np.random.shuffle(perm_inds)
        # print(perm_inds)
        perm_fmnist = []
        for set in fmnist_imgs:
            num_img = set.shape[0]
            # print(num_img, w, h)
            flat_set = set.reshape(num_img, w * h)
            perm_fmnist.append(flat_set[:, perm_inds].reshape(num_img, w, h))
        return perm_fmnist

    def rotate_fmnist(self, rotation=20):
        """
        Given the train and test set (no labels), rotate each img the
        same way.
        """

        # we take only the images
        fmnist_imgs = [self.train_set[0], self.test_set[0]]
        rot_fmnist = deepcopy(fmnist_imgs)

        print("starting rotation...")
        for i, set in enumerate(rot_fmnist):
            for j in range(set.shape[0]):
                img = Image.fromarray(np.squeeze(set[j]), mode='L')
                rot_fmnist[i][j] = img.rotate(rotation)

        return rot_fmnist

    def download_fmnist(self):

        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        for name in filename:
            print("Downloading " + name[1]+"...")
            urlretrieve(base_url + name[1], self.data_loc + name[1])
        print("Download complete.")

    def save_fmnist(self):

        fmnist = {}
        for name in filename[:2]:
            with gzip.open(self.data_loc + name[1], 'rb') as f:
                tmp = np.frombuffer(f.read(), np.uint8, offset=16)
                fmnist[name[0]] = tmp.reshape(-1, 1, 28, 28)\
                                     .astype(np.float32) / 255
        for name in filename[-2:]:
            with gzip.open(self.data_loc + name[1], 'rb') as f:
                fmnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(self.data_loc + "fmnist.pkl", 'wb') as f:
            pickle.dump(fmnist, f)
        print("Save complete.")


if __name__ == '__main__':

    # np.set_printoptions(threshold=np.nan)
    fmnist = fashionMNIST()

    # try permutation
    fmnist_data = fmnist.get_data()
    perm_fmnist = fmnist.permute_fmnist(seed=0)

    # let's see some images
    #print(mnist_data[0][0][0].shape)
    #print(perm_mnist[0][0].shape)
    #imgplot = plt.imshow(fmnist_data[0][0][300].squeeze(), cmap='gray')
    # imgplot = plt.imshow(perm_fmnist[0][0], cmap='gray')
    #plt.show()

