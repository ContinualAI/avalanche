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

""" Common training utils. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from avalanche.training.strategies.skeletons.strategy \
    import Strategy
from avalanche.evaluation.eval_protocol import EvalProtocol
from avalanche.evaluation.metrics import ACC
from avalanche.extras.models.mobilenetv1 import MobilenetV1
from avalanche.training.strategies.ar1.utils import replace_bn_with_brn, \
    create_syn_data, init_batch, freeze_up_to, change_brn_pars, \
    examples_per_class, reset_weights, pre_update, compute_ewc_loss, \
    consolidate_weights, update_ewc_data, set_consolidate_weights
from avalanche.training.utils import pad_data, shuffle_in_unison
import torch
import copy
import numpy as np


class AR1(Strategy):
    """
    Naive Strategy: PyTorch implementation.
    """

    def __init__(self, model, optimizer=None,
                 criterion=torch.nn.CrossEntropyLoss(), mb_size=128,
                 train_ep=2, multi_head=False, device=None, preproc=None,
                 eval_protocol=EvalProtocol(metrics=[ACC]), lr=0.001,
                 init_update_rate=0.01, inc_update_rate=0.00005, max_r_max=1.25,
                 max_d_max=0.5, inc_step=4.1e-05, rm_sz=1500, momentum=0.9,
                 l2 = 0.0005, freeze_below_layer="lat_features.19.bn.beta",
                 latent_layer_num=19, ewc_lambda=0):

        super(AR1, self).__init__(model, optimizer, criterion, mb_size, train_ep,
                                  multi_head, device, preproc, eval_protocol)

        if optimizer is None:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=l2
            )

        # Model setup
        model = MobilenetV1(pretrained=True, latent_layer_num=latent_layer_num)
        replace_bn_with_brn(
            model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
            max_r_max=max_r_max, max_d_max=max_d_max
        )

        model.saved_weights = {}
        model.past_j = {i: 0 for i in range(50)}
        model.cur_j = {i: 0 for i in range(50)}
        if ewc_lambda != 0:
            self.ewcData, self.synData = create_syn_data(model)

        super(AR1, self).__init__(
            model, optimizer, criterion, mb_size, train_ep, multi_head,
            device, preproc, eval_protocol
        )

        self.ewc_lambda = ewc_lambda
        self.freeze_below_layer = freeze_below_layer
        self.rm_sz = rm_sz
        self.inc_update_rate = inc_update_rate
        self.max_r_max = max_r_max
        self.max_d_max = max_d_max
        self.lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.rm = None

    def train(self, x, y, t):

        self.cur_ep = 0
        self.cur_train_t = t

        if self.preproc:
            x = self.preproc(x)

        if self.ewc_lambda != 0:
            init_batch(self.model, self.ewcData, self.synData)

        if self.batch_processed == 1:
            freeze_up_to(self.model, self.freeze_below_layer)
            change_brn_pars(
                self.model, momentum=self.inc_update_rate, r_d_max_inc_step=0,
                r_max=self.max_r_max, d_max=self.max_d_max)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=self.momentum,
                weight_decay=self.l2
            )

        train_x, train_y = x, y
        if self.preproc:
            train_x = self.preproc(train_x)

        if self.batch_processed == 0:
            cur_class = [int(o) for o in set(train_y)]
            self.model.cur_j = examples_per_class(train_y)
        else:
            cur_class = [int(o) for o in set(train_y).union(set(self.rm[1]))]
            self.model.cur_j = examples_per_class(list(train_y) +
                                                  list(self.rm[1]))

        self.model.eval()
        self.model.end_features.train()
        self.model.output.train()

        reset_weights(self.model, cur_class)
        cur_ep = 0

        if self.batch_processed == 0:
            (train_x, train_y), it_x_ep = pad_data([train_x, train_y],
                                                   self.mb_size)
        shuffle_in_unison([train_x, train_y], in_place=True)

        self.model = self.model.to(self.device)
        acc = None
        ave_loss = 0

        train_x = torch.tensor(train_x, dtype=torch.float)
        train_y = torch.tensor(train_y, dtype=torch.long)

        for ep in range(self.train_ep):

            print("training ep: ", ep)
            correct_cnt, ave_loss = 0, 0

            if self.batch_processed > 0:
                cur_sz = train_x.size(0) // (
                            (train_x.size(0) + self.rm_sz) // self.mb_size)
                it_x_ep = train_x.size(0) // cur_sz
                n2inject = max(0, self.mb_size - cur_sz)
            else:
                n2inject = 0
            print("total sz:", train_x.size(0) + self.rm_sz)
            print("n2inject", n2inject)
            print("it x ep: ", it_x_ep)

            for it in range(it_x_ep):

                if self.ewc_lambda != 0:
                    pre_update(self.model, self.synData)

                start = it * (self.mb_size - n2inject)
                end = (it + 1) * (self.mb_size - n2inject)

                self.optimizer.zero_grad()

                x_mb = train_x[start:end].to(self.device)

                if self.batch_processed == 0:
                    lat_mb_x = None
                    y_mb = train_y[start:end].to(self.device)

                else:
                    lat_mb_x = self.rm[0][it * n2inject: (it + 1) * n2inject]
                    lat_mb_y = self.rm[1][it * n2inject: (it + 1) * n2inject]
                    y_mb = torch.cat((train_y[start:end], lat_mb_y), 0).to(self.device)
                    lat_mb_x = lat_mb_x.to(self.device)

                logits, lat_acts = self.model(
                    x_mb, latent_input=lat_mb_x, return_lat_acts=True)

                # collect latent volumes only for the first ep
                if ep == 0:
                    lat_acts = lat_acts.cpu().detach()
                    if it == 0:
                        cur_acts = copy.deepcopy(lat_acts)
                    else:
                        cur_acts = torch.cat((cur_acts, lat_acts), 0)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()

                loss = self.criterion(logits, y_mb)
                if self.ewc_lambda != 0:
                    loss += compute_ewc_loss(self.model, self.ewcData,
                                             lambd=self.ewc_lambda,
                                             device=self.device)
                ave_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                if self.ewc_lambda != 0:
                    self.post_update(self.model, self.synData)

                acc = correct_cnt.item() / \
                      ((it + 1) * y_mb.size(0))
                ave_loss /= ((it + 1) * y_mb.size(0))

                if it % 100 == 0:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'.format(it, ave_loss, acc)
                    )
                    self.eval_protocol.update_tb_train(
                        ave_loss, acc, self.total_it_processed,
                        torch.unique(train_y), self.cur_train_t
                    )

                self.total_it_processed += 1

            self.cur_ep += 1

        consolidate_weights(self.model, cur_class)
        if self.ewc_lambda != 0:
            update_ewc_data(self.model, self.ewcData, self.synData, 0.001, 1)

        # how many patterns to save for next iter
        h = min(self.rm_sz // (self.batch_processed + 1), cur_acts.size(0))
        print("h", h)

        print("cur_acts sz:", cur_acts.size(0))
        idxs_cur = np.random.choice(
            cur_acts.size(0), h, replace=False
        )
        rm_add = [cur_acts[idxs_cur], train_y[idxs_cur]]
        print("rm_add size", rm_add[0].size(0))

        # replace patterns in random memory
        if self.batch_processed == 0:
            self.rm = copy.deepcopy(rm_add)
        else:
            idxs_2_replace = np.random.choice(
                self.rm[0].size(0), h, replace=False
            )
            for j, idx in enumerate(idxs_2_replace):
                self.rm[0][idx] = copy.deepcopy(rm_add[0][j])
                self.rm[1][idx] = copy.deepcopy(rm_add[1][j])

        set_consolidate_weights(self.model)

        # update number examples encountered over time
        for c, n in self.model.cur_j.items():
            self.model.past_j[c] += n

        self.batch_processed +=1

        return ave_loss, acc

    def before_test(self):
        pass

    def after_test(self):
        pass

    def before_task_test(self):
        pass

    def after_task_test(self):
        pass
