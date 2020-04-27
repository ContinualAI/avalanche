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

from avalanche.benchmarks import CMNIST
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Rehearsal
from avalanche.evaluation import EvalProtocol
import torch
from torch.utils.tensorboard import SummaryWriter

# Tensorboard setup
exp_name = "mnist_rehe_test"
log_dir = '../logs/' + exp_name
writer = SummaryWriter(log_dir)

# load the model with PyTorch for example
model = SimpleMLP()

# load the benchmark as a python iterator object
cdata = CMNIST()

# Eval Protocol
evalp = EvalProtocol(metrics=[ACC], tb_writer=writer)

# adding the CL strategy
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
clmodel = Rehearsal(
    model, optimizer=optimizer, rm_sz=1500, train_ep=4, eval_protocol=evalp
)

# getting full test set beforehand
test_full = cdata.get_full_testset()

results = []

# loop over the training incremental batches
for i, (x, y, t) in enumerate(cdata):

    # training over the batch
    print("Batch {0}, task {1}".format(i, t))
    clmodel.train(x, y, t)

    # testing
    clmodel.test(test_full)

writer.close()

