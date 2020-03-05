#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. ContinualAI. All rights reserved.                        #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-07-2019                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" Avalanche usage examples """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from avalanche.benchmarks import CMNIST, CORE50
from avalanche.evaluation.metrics import ACC, CF, RAMU
# from avalanche.extras.models.pytorch import SimpleMLP
# from avalanche.extras.models.tensorflow import SimpleMLP
from avalanche.extras.models.caffe import SimpleMLP, SimpleCNN
from avalanche.training.utils.common import imagenet_batch_preproc
from avalanche.training.strategies import Naive
from avalanche.evaluation import EvalProtocol

# load the model with PyTorch/Tensorflow/Caffe for example
model = SimpleCNN(num_classes=50)

# load the benchmark as a python iterator object
# cdata = CMNIST()
cdata = CORE50(root='/home/admin/Ior50N/128/', scenario="nicv2_79")

# adding the CL strategy
clmodel = Naive({'model': model, 'preproc': imagenet_batch_preproc,
                 'train_ep': 4})

# Eval Protocol
evalp = EvalProtocol(metrics=[ACC, CF, RAMU])

# getting full test set beforehand
test_full = cdata.get_full_testset()

results = []

# loop over the training incremental batches
for i, (x, y, t) in enumerate(cdata):

    # training over the batch
    print("Batch {0}, task {1}".format(i, t))
    clmodel.train(x, y, t)

    # here we could get the growing test set too
    # test_grow = cdata.get_growing_testset()

    # testing
    results.append(clmodel.test(test_full, eval_protocol=evalp))

