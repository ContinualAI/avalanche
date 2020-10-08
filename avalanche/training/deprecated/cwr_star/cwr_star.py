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

""" CopyWeight with Re-Init Strategy """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from avalanche.training.deprecated.strategy import Strategy
from avalanche.evaluation.eval_protocol import EvalProtocol
from avalanche.evaluation.metrics import ACC
from avalanche.training.deprecated.cwr_star.utils import freeze_up_to, \
    examples_per_class, reset_weights, consolidate_weights,  \
    set_consolidate_weights
import torch


class CWRStar(Strategy):

    def __init__(self, model, optimizer=None,
                 criterion=torch.nn.CrossEntropyLoss(), mb_size=128,
                 train_ep=2, multi_head=False, device=None, preproc=None,
                 eval_protocol=EvalProtocol(metrics=[ACC()]), lr=0.001,
                 momentum=0.9, l2=0.0005, second_last_layer_name=None):
        """ CWR* Strategy.

        :param model: pytorch basic model.
        :param optimizer: pytorch optimizer.
        :param criterion: pytorch optimization criterion.
        :param int mb_size: mini-batch size for SGD.
        :param int train_ep: training epochs for each task/batch
        :param multi_head: multi-head or not.
        :param device: device on which to run the script.
        :param preproc: preprocessing function.
        :param eval_protocol: avalanche evaluation protocol.
        :param lr: learning rate.
        :param momentum: momentum.
        :param l2: l2 decay coefficient.
        :param second_last_layer_name: name of the second to last layer.
        """

        super(CWRStar, self).__init__(
            model, optimizer, criterion, mb_size, train_ep, multi_head,
            device, preproc, eval_protocol)

        if second_last_layer_name is None:
            raise Exception("To use CWR* you need to specify the second last "
                            "layer name of the model you are using.")
        else:
            self.second_last_layer_name = second_last_layer_name

        if optimizer is None:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=l2
            )

        # Model setup
        self.model.saved_weights = {}
        self.model.past_j = {i: 0 for i in range(50)}
        self.model.cur_j = {i: 0 for i in range(50)}

        super(CWRStar, self).__init__(
            model, optimizer, criterion, mb_size, train_ep, multi_head,
            device, preproc, eval_protocol
        )

        self.lr = lr
        self.momentum = momentum
        self.l2 = l2

        # to be updated
        self.cur_class = None

    def after_train(self):
        consolidate_weights(self.model, self.cur_class)

    def before_train(self):

        if self.batch_processed == 1:
            freeze_up_to(self.model, self.second_last_layer_name)

        self.cur_class = [int(o) for o in set(self.y)]
        self.model.cur_j = examples_per_class(self.y)
        reset_weights(self.model, self.cur_class)

    def before_test(self):
        set_consolidate_weights(self.model)
