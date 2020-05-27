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

""" Useful constants for the Avalanche package """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Other imports
from enum import Enum, unique


@unique
class Scenario(Enum):
    NI = 'ni'
    NC = 'nc'
    NIC = 'nic'
    NIC8 = 'nic8'
    NC25 = 'nc25'


@unique
class Strategy(Enum):
    LWF = 'lwf'
    NAIVE = 'naive'
    FROM_SCRATCH = 'from_scratch'
    REHEARSAL = 'rehearsal'
    CWR = 'cwr'
    CW = 'cw'
    CWRP = 'cwr+'
    FREEZE_FC8 = 'freeze_fc8'
    SST_A_D = 'sst_a_d'
    EWC = 'ewc'
    SYN = 'syn'
    CCWR = 'ccwr'


@unique
class DataBackEnd(Enum):
    LMDB = 'lmdb'
    IMAGE_DATA = 'image_data'
    MEMORY_DATA = 'memory_data'


@unique
class DLBackEnd(Enum):
    CAFFE = 'caffe'
    PYTORCH = 'pytorch'
    TENSORFLOW = 'tensorflow'


@unique
class HWBackEnd(Enum):
    GPU = 'gpu'
    CPU = 'cpu'


@unique
class Phase(Enum):
    TRAIN = 1
    TEST = 2


@unique
class Dataset(Enum):
    FAKE_DATASET = 'fake_dataset'
    CORE50 = 'core50'
    CMNIST = 'cmnist'
    MNIST_SPLIT = 'mnist_split'
    MNIST_PERM = 'mnist_perm'
    MNIST_ROT = 'mnist_rot'
    CIFAR_SPLIT = 'cifar_split'
    CIFAR_SPLIT_FULL = 'cifar_split_full'
    ICifar100 = 'icifar100'
    ICIFAR10 = 'icifar10'
    CFASHION_MNIST = 'cfashionmnist'

@unique
class Net(Enum):
    MID_CAFFENET = 'mid_caffenet'
    CIFAR_SPLIT_NET = 'cifar_split_net'
    NIN = 'nin'

@unique
class EvalProtocol(Enum):
    FIXED_TEST_SET = 'fixed_test_set'
    GROWING_TEST_SET = 'growing_test_set'
