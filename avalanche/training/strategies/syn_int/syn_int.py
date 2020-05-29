#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

""" Synaptic Intelligence Strategy """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from ..strategy import Strategy
from avalanche.evaluation.eval_protocol import EvalProtocol
from avalanche.evaluation.metrics import ACC
from .utils import create_syn_data, init_batch, compute_ewc_loss, pre_update,\
    post_update, update_ewc_data
import torch


class SynInt(Strategy):

    def __init__(self, model, optimizer=None,
                 criterion=torch.nn.CrossEntropyLoss(), mb_size=128,
                 train_ep=2, multi_head=False, device=None, preproc=None,
                 eval_protocol=EvalProtocol(metrics=[ACC()]), lr=0.001,
                 momentum=0.9, l2=0.0005, si_lambda=0):
        """ Synaptic Intelligence Strategy.

        This is the Synaptic Intelligence pytorch implementation of the
        algorithm described in the paper "Continual Learning Through Synaptic
        Intelligence" (https://arxiv.org/abs/1703.04200)

        :param model: pytorch basic model.
        :param optimizer: pytorch optimizer.
        :param criterion: pytorch optimization criterion.
        :param int mb_size: mini-batch size for SGD.
        :param int train_ep: training epochs for each task/batch
        :param bool multi_head: multi-head or not
        :param device device: device on which to run the script.
        :param preproc: prepocessing function.
        :param eval_protocol: avalanche evaluation protocol.
        :param lr: learning rate for the optimizer.
        :param momentum: momentum used
        :param l2: weights decay regularization.
        :param si_lambda: Synaptic Intellgence lambda term.
        """

        if optimizer is None:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=l2
            )

        self.lr = lr
        self.momentum = momentum
        self.l2 = l2

        super(SynInt, self).__init__(
            model, optimizer, criterion, mb_size, train_ep, multi_head,
            device, preproc, eval_protocol
        )

        self.si_lambda = si_lambda

        # to be updated
        self.ewcData, self.synData = create_syn_data(model)

    def before_train(self):

        init_batch(self.model, self.ewcData, self.synData)

    def compute_loss(self, logits, y_mb):

        loss = self.criterion(logits, y_mb)
        loss += compute_ewc_loss(self.model, self.ewcData,
                                 lambd=self.si_lambda,
                                 device=self.device)
        return loss

    def before_iteration(self):

        pre_update(self.model, self.synData)

    def after_iter_ended(self):

        post_update(self.model, self.synData)

    def after_train(self):

        update_ewc_data(self.model, self.ewcData, self.synData, 0.001, 1)
