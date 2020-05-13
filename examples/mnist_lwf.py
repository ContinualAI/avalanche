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

from avalanche.benchmarks import CMNIST
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import LearningWithoutForgetting
from avalanche.evaluation import AccEvalProtocol
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn


# Tensorboard setup
exp_name = "mnist_lwf"
log_dir = '../logs/' + exp_name
writer = SummaryWriter(log_dir)
num_class = 10
mode = 'perm'  # one of 'perm' or 'split'


model = SimpleMLP()
if mode == 'perm':
    cdata = CMNIST(num_batch=10, mode='perm')
    evalp = AccEvalProtocol()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    clmodel = LearningWithoutForgetting(
        model, classes_per_task=10, optimizer=optimizer, alpha=1,
        warmup_epochs=0, train_ep=10, eval_protocol=evalp
    )
elif mode == 'split':
    cdata = CMNIST(num_batch=5, mode='split')
    evalp = AccEvalProtocol()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    clmodel = LearningWithoutForgetting(
        model, classes_per_task=2, optimizer=optimizer,
        alpha=[0, 1/2, 2*2/3, 3*3/4, 4*4/5], warmup_epochs=2,
        train_ep=10, distillation_loss_T=1, eval_protocol=evalp
    )
else:
    assert False

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

