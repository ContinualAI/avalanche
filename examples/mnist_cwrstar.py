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

""" CWR* usage examples """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import pdb;pdb.set_trace()
from avalanche.benchmarks import CMNIST
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import CWRStar, Naive
from avalanche.evaluation import EvalProtocol

strat = "naive" # "cwrstar"

# load the model with PyTorch for example
model = SimpleMLP()

# load the benchmark as a python iterator object
cdata = CMNIST(mode="split", num_batch=5)

# Eval Protocol
evalp = EvalProtocol()

# adding the CL strategy
device = torch.device("cpu")

if strat == "cwrstar":
    clmodel = CWRStar(
        model, eval_protocol=evalp, device=device,
        second_last_layer_name="features.0.bias"
    )
else:
    clmodel = Naive(model, eval_protocol=evalp, device=device)

# getting full test set beforehand
test_full = cdata.get_full_testset()

results = []

# loop over the training incremental batches
for i, (x, y, t) in enumerate(cdata):

    # training over the batch
    print("Batch {0}, task {1}".format(i, t))
    clmodel.train(x, y, t)

    # testing
    results.append(clmodel.test(test_full))
