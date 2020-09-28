#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" This module conveniently wraps Pytorch Datasets for using a clean and
comprehensive Avalanche API."""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from torchvision.datasets import MNIST as torchMNIST
from torchvision.datasets import FashionMNIST as torchFashionMNIST
from torchvision.datasets import KMNIST as torchKMNIST
from torchvision.datasets import EMNIST as torchEMNIST
from torchvision.datasets import QMNIST as torchQMNIST
from torchvision.datasets import FakeData as torchFakeData


def MNIST(*args, **kwargs):
    return torchMNIST(*args, **kwargs)


def FashionMNIST(*args, **kwargs):
    return torchFashionMNIST(*args, **kwargs)


def KMNIST(*args, **kwargs):
    return torchKMNIST(*args, **kwargs)


def EMNIST(*args, **kwargs):
    return torchEMNIST(*args, **kwargs)


def QMNIST(*args, **kwargs):
    return torchQMNIST(*args, **kwargs)


def FakeData(*args, **kwargs):
    return torchFakeData(*args, **kwargs)


if __name__ == "__main__":

    mnist = MNIST(".", download=True)
