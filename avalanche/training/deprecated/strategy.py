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

import torch
from torch.utils.data import Dataset

from avalanche.training.utils import pad_data, shuffle_in_unison, \
    load_all_dataset


class Strategy(object):

    def __init__(self, model, optimizer=None,
                 criterion=torch.nn.CrossEntropyLoss(), mb_size=256,
                 train_ep=2, multi_head=False, device=None, preproc=None):

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
        self.device = device

        # to be updated
        self.cur_ep = None
        self.cur_train_t = None
        self.cur_test_t = None
        self.num_class = None
        self.batch_processed = 0
        self.total_it_processed = 0

        super(Strategy, self).__init__()

    def train_using_dataset(self, dataset_and_t_label, num_workers=8):
        dataset, t = dataset_and_t_label
        x, y = load_all_dataset(dataset, num_workers=num_workers)
        return self.train(x, y, t)

    def train(self, x, y, t):
        self.x, self.y, self.t = x, y, t
        self.before_train()

        self.cur_ep = 0
        self.cur_train_t = t
        train_x, train_y, it_x_ep = self.preproc_batch_data(x, y, t)

        correct_cnt, ave_loss = 0, 0
        acc = None

        # Differently from .tensor(...), .as_tensor(...) will not make a
        # copy of the data if not strictly needed!
        train_x = torch.as_tensor(train_x, dtype=torch.float)
        train_y = torch.as_tensor(train_y, dtype=torch.long)

        for ep in range(self.train_ep):
            self.before_epoch()

            for it in range(it_x_ep):
                self.before_iteration()

                start = it * self.mb_size
                end = (it + 1) * self.mb_size

                self.optimizer.zero_grad()

                model = self.model.to(self.device)
                self.x_mb = train_x[start:end].to(self.device)
                y_mb = train_y[start:end].to(self.device)
                logits = model(self.x_mb)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum().item()

                loss = self.compute_loss(logits, y_mb)
                ave_loss += loss.item()

                self.before_weights_update()
                loss.backward()
                self.optimizer.step()

                acc = correct_cnt / ((it + 1) * y_mb.size(0) + ep * x.shape[0])
                ave_loss /= ((it + 1) * y_mb.size(0) + ep * x.shape[0])

                if it % 100 == 0:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'.format(it, ave_loss, acc)
                    )
                    self.eval_protocol.update_tb_train(
                        ave_loss, acc, self.total_it_processed,
                        torch.unique(train_y), self.cur_train_t
                    )

                self.after_iter_ended()
                self.total_it_processed += 1

            self.after_epoch_ended()
            self.cur_ep += 1

        self.after_train()
        self.batch_processed += 1
        return ave_loss, acc

    def test(self, test_set, num_workers=8):
        self.before_test()
        res = {}
        ave_loss = 0
        for dataset, t in test_set:
            # In this way dataset can be both a tuple (x, y) and a Dataset
            if isinstance(dataset, Dataset):
                x, y = load_all_dataset(dataset, num_workers=num_workers)
            else:
                x, y = dataset[:]

            if self.preproc:
                x = self.preproc(x)

            self.cur_test_t = t
            self.before_task_test()

            (test_x, test_y), it_x_ep = pad_data(
                [x, y], self.mb_size
            )

            # Differently from .tensor(...), .as_tensor(...) will not make a
            # copy of the data if not strictly needed!
            test_x = torch.as_tensor(test_x, dtype=torch.float)
            test_y = torch.as_tensor(test_y, dtype=torch.long)

            y_hat = []
            true_y = []

            for i in range(it_x_ep):
                # indexing
                start = i * self.mb_size
                end = (i + 1) * self.mb_size

                model = self.model.to(self.device)
                x_mb = test_x[start:end].to(self.device)
                y_mb = test_y[start:end].to(self.device)

                logits = model(x_mb)

                loss = self.compute_loss(logits, y_mb)
                ave_loss += loss.item()

                _, pred_label = torch.max(logits, 1)

                y_hat.append(pred_label.cpu().numpy())
                true_y.append(y_mb.cpu().numpy())

            results = self.eval_protocol.get_results(
                true_y, y_hat, self.cur_train_t, self.cur_test_t
            )
            acc, accs = results[ACC]

            ave_loss /= test_y.size(0)

            print("Task {0}: Avg Loss {1}; Avg Acc {2}"
                  .format(t, ave_loss, acc))
            res[t] = (ave_loss, acc, accs, results)

            self.after_task_test()

        self.eval_protocol.update_tb_test(res, self.batch_processed)

        self.after_test()
        return res

    def compute_loss(self, logits, y_mb):
        return self.criterion(logits, y_mb)

    def before_train(self):
        pass

    def preproc_batch_data(self, x, y, t):

        if self.preproc:
            x = self.preproc(x)

        (train_x, train_y), it_x_ep = pad_data(
            [x, y], self.mb_size
        )

        shuffle_in_unison(
            [train_x, train_y], 0, in_place=True
        )

        return train_x, train_y, it_x_ep

    def before_epoch(self):
        pass

    def before_iteration(self):
        pass

    def before_weights_update(self):
        pass

    def after_iter_ended(self):
        pass

    def after_epoch_ended(self):
        pass

    def after_train(self):
        pass

    def before_test(self):
        pass

    def after_test(self):
        pass

    def before_task_test(self):
        pass

    def after_task_test(self):
        pass
