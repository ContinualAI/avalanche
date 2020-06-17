#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): ContinualAI                                                       #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

""" Rehearsal Strategy Implementation """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from avalanche.training.strategies.strategy import Strategy
from avalanche.evaluation.eval_protocol import EvalProtocol
from avalanche.evaluation.metrics import ACC
import torch
import torch.nn.functional as F
import copy


def distillation_loss(y_pred, y_teacher, temperature):
    """ Distillation loss. """
    # kl_div is normalized by element instead of observation
    scale = y_teacher.shape[-1]
    log_p = F.log_softmax(y_pred / temperature, dim=1)
    q = F.softmax(y_teacher / temperature, dim=1)
    res = scale * F.kl_div(log_p, q, reduction='mean')
    return res


class LearningWithoutForgetting(Strategy):
    def __init__(self, model, classes_per_task, alpha=0.5,
                 distillation_loss_T=2, warmup_epochs=0, optimizer=None,
                 criterion=torch.nn.CrossEntropyLoss(), mb_size=256,
                 train_ep=2, device=None, preproc=None,
                 eval_protocol=EvalProtocol(metrics=[ACC()])):
        """ Learning without Forgetting Strategy.

        paper: https://arxiv.org/abs/1606.09282
        original implementation (Matlab):
            https://github.com/lizhitwo/LearningWithoutForgetting
        reference implementation (pytorch):
            https://github.com/arunmallya/packnet/blob/master/src/lwf.py

        :param model: pytorch basic model.
        :param classes_per_task: number of classes for each task.
        :param alpha: distillation loss coefficient.
        :param distillation_loss_T: distillation loss temperature.
        :param warmup_epochs: number of warmup epochs training only
            the new parameters.
        :param optimizer: pytorch optimizer.
        :param criterion: pytorch optimization criterion.
        :param mb_size: mini-batch size for SGD.
        :param train_ep: training epochs for each task/batch.
        :param device: device on which to run the script.
        :param preproc: preprocessing function.
        :param eval_protocol: avalanche evaluation protocol.
        """

        super(LearningWithoutForgetting, self).__init__(
            model, optimizer, criterion, mb_size, train_ep, multi_head=False,
            device=device, preproc=preproc, eval_protocol=eval_protocol
        )

        # LwF parameters
        self.classes_per_task = classes_per_task
        self.prev_model = None
        self.distillation_loss_T = distillation_loss_T
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs

    def warmup_train(self):
        """ Train only the new parameters for the first epochs. """
        # add only the last layer to the trainable parameters
        opt = torch.optim.SGD(lr=0.01,
                              params=self.model.classifier.parameters())

        train_x, train_y, it_x_ep = \
            self.preproc_batch_data(self.x, self.y, self.t)
        model = self.model.to(self.device)

        train_x = torch.tensor(train_x, dtype=torch.float)
        train_y = torch.tensor(train_y, dtype=torch.long)
        for ep in range(self.warmup_epochs):
            for it in range(it_x_ep):
                start = it * self.mb_size
                end = (it + 1) * self.mb_size

                self.optimizer.zero_grad()
                x_mb = train_x[start:end].to(self.device)
                y_mb = train_y[start:end].to(self.device)
                logits = model(x_mb)
                # loss computed only on the new classes
                loss = self.criterion(
                    logits[:, self.t*self.classes_per_task:(self.t+1)*self.classes_per_task],  # noqa
                    y_mb - self.t*self.classes_per_task)
                loss.backward()
                opt.step()

    def compute_loss(self, logits, y_mb):
        dist_loss = 0
        if self.prev_model is not None:
            y_prev = self.prev_model(self.x_mb).detach()
            loss = self.criterion(logits, y_mb)
            dist_loss += distillation_loss(logits, y_prev,
                                           self.distillation_loss_T)

            if isinstance(self.alpha, list):
                loss = loss + self.alpha[self.t] * dist_loss
            else:
                loss = loss + self.alpha * dist_loss
        else:
            loss = self.criterion(logits, y_mb)
        return loss

    def before_train(self):
        self.warmup_train()

    def after_train(self):
        self.prev_model = copy.deepcopy(self.model)
