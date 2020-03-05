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

""" Common training utils. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from avalanche.evaluation.metrics import ACC
from avalanche.training.strategies.skeletons import Strategy
from avalanche.training.utils.common import pad_data, shuffle_in_unison
from avalanche.training.utils.pytorch_utils import maybe_cuda
import torch


class StrategyPytorch(Strategy):

    def __init__(self, model, optimizer=None,
                 criterion=torch.nn.CrossEntropyLoss(), mb_size=256,
                 train_ep=2, multi_head=False, use_cuda=False, preproc=None):

        self.model = model
        if optimizer is None:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=0.01
            )
        else:
            self.optimizer = optimizer

        self.preproc = preproc
        self.criterion = criterion
        self.mb_size = mb_size
        self.train_ep = train_ep
        self.multi_head = multi_head
        self.use_cuda = use_cuda

        # to be updated
        self.cur_ep = None
        self.cur_train_t = None
        self.cur_test_t = None
        self.num_class = None

        super(StrategyPytorch, self).__init__()

    def train(self, x, y, t):

        self.cur_ep = 0
        self.cur_train_t = t

        if self.preproc:
            x = self.preproc(x)

        (train_x, train_y), it_x_ep = pad_data(
            [x, y], self.mb_size
        )

        shuffle_in_unison(
            [train_x, train_y], 0, in_place=True
        )

        correct_cnt, ave_loss = 0, 0
        model = maybe_cuda(self.model, use_cuda=self.use_cuda)
        acc = None

        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.LongTensor)

        for ep in range(self.train_ep):
            self.before_epoch()

            for it in range(it_x_ep):
                self.before_iteration()

                start = it * self.mb_size
                end = (it + 1) * self.mb_size

                self.optimizer.zero_grad()

                x_mb = maybe_cuda(train_x[start:end], use_cuda=self.use_cuda)
                y_mb = maybe_cuda(train_y[start:end], use_cuda=self.use_cuda)
                logits = model(x_mb)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()

                loss = self.criterion(logits, y_mb)
                ave_loss += loss.item()

                self.before_weights_update()
                loss.backward()
                self.optimizer.step()

                acc = correct_cnt.item() / \
                      ((it + 1) * y_mb.size(0) + ep * x.shape[0])
                ave_loss /= ((it + 1) * y_mb.size(0) + ep * x.shape[0])

                if it % 100 == 0:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(it, ave_loss, acc)
                    )

                self.after_iter_ended()

            self.after_epoch_ended()
            self.cur_ep += 1

        return ave_loss, acc

    def test(self, test_set, eval_protocol=[ACC]):

        self.before_test()

        res = {}
        ave_loss = 0
        acc = None
        accs = None

        for (x, y), t in test_set:

            if self.preproc:
                x = self.preproc(x)

            self.cur_test_t = t
            self.before_task_test()

            (test_x, test_y), it_x_ep = pad_data(
                [x, y], self.mb_size
            )

            test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
            test_y = torch.from_numpy(test_y).type(torch.LongTensor)

            model = maybe_cuda(self.model, use_cuda=self.use_cuda)

            res[t] = []
            y_hat = []
            true_y = []

            for i in range(it_x_ep):

                # indexing
                start = i * self.mb_size
                end = (i + 1) * self.mb_size

                x_mb = maybe_cuda(test_x[start:end], use_cuda=self.use_cuda)
                y_mb = maybe_cuda(test_y[start:end], use_cuda=self.use_cuda)

                logits = model(x_mb)

                loss = self.criterion(logits, y_mb)
                ave_loss += loss.item()

                _, pred_label = torch.max(logits, 1)

                y_hat.append(pred_label.numpy())
                true_y.append(y_mb.numpy())

            if eval_protocol:
                results = eval_protocol.get_results(true_y, y_hat, t)
                acc, accs = results

            ave_loss /= test_y.size(0)

            print("Task {0}: Avg Loss {1}; Avg Acc {2}"
                  .format(t, ave_loss, acc))
            res[t].append((ave_loss, acc, accs))

            self.after_task_test()

        self.after_test()

        return res

    def before_epoch(self):
        super(StrategyPytorch, self).before_epoch()

    def before_iteration(self):
        super(StrategyPytorch, self).before_iteration()

    def before_weights_update(self):
        super(StrategyPytorch, self).before_weights_update()

    def after_iter_ended(self):
        super(StrategyPytorch, self).after_iter_ended()

    def after_epoch_ended(self):
        super(StrategyPytorch, self).after_epoch_ended()

    def before_test(self):
        super(StrategyPytorch, self).before_test()

    def after_test(self):
        super(StrategyPytorch, self).after_test()

    def before_task_test(self):
        super(StrategyPytorch, self).before_task_test()

    def after_task_test(self):
        super(StrategyPytorch, self).after_task_test()


