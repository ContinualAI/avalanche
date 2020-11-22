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

""" Avalanche usage examples """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from avalanche.benchmarks import CORE50
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleCNN
from avalanche.benchmarks.utils import imagenet_batch_preproc
from avalanche.training.strategies import Naive
from avalanche.evaluation import EvalProtocol

import torch

# load the model
model = SimpleCNN(num_classes=50)

# load the benchmark as a python iterator object
cdata = CORE50(scenario="nicv2_391")

# Eval Protocol
evalp = EvalProtocol(
    metrics=[ACC(), CF(), RAMU(), CM()], tb_logdir='../logs/core_test_391'
)

# adding the CL strategy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

clmodel = Naive(
    model, optimizer=optimizer, eval_protocol=evalp,
    preproc=imagenet_batch_preproc, train_ep=4, mb_size=128,
    device=device
)

# getting full test set beforehand
test_full = cdata.get_full_testset()

# loop over the training incremental batches
for i, (x, y, t) in enumerate(cdata):

    # training over the batch
    print("Batch {0}, task {1}".format(i, t))
    clmodel.train(x, y, t)

    # testing
    clmodel.test(test_full)


