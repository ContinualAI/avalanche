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
import tensorflow as tf


class StrategyTensorflow(Strategy):

    def __init__(self, model, optimizer=None, criterion=None, mb_size=256,
                 train_ep=2, multi_head=False, use_cuda=False, preproc=None):

        self.model = model
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = tf.keras.losses.\
                 SparseCategoricalCrossentropy(from_logits=True)
        else:
            self.criterion = criterion

        self.preproc = preproc
        self.mb_size = mb_size
        self.train_ep = train_ep
        self.multi_head = multi_head
        self.use_cuda = use_cuda

        # to be updated
        self.cur_ep = None
        self.cur_train_t = None
        self.cur_test_t = None
        self.num_class = None

        super(StrategyTensorflow, self).__init__()

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
        acc = None

        for ep in range(self.train_ep):
            self.before_epoch()

            for it in range(it_x_ep):
                self.before_iteration()

                start = it * self.mb_size
                end = (it + 1) * self.mb_size

                x_mb = train_x[start:end]
                y_mb = train_y[start:end]

                with tf.GradientTape() as tape:
                    logits = self.model(x_mb)
                    pred_label = tf.argmax(logits, axis=1)
                    loss = self.criterion(y_mb, logits)

                ave_loss += loss.numpy().mean()

                for i, lab in enumerate(pred_label):
                    if int(lab) == y_mb[i]:
                        correct_cnt += 1

                self.before_weights_update()

                grads = tape.gradient(
                    loss, self.model.trainable_variables
                )
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

                acc = float(correct_cnt) / \
                      ((it + 1) * y_mb.shape[0] + ep * x.shape[0])
                ave_loss /= ((it + 1) * y_mb.shape[0] + ep * x.shape[0])

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

    def test(self, test_set, eval_protocol=None):

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

            res[t] = []
            y_hat = []
            true_y = []

            for i in range(it_x_ep):

                # indexing
                start = i * self.mb_size
                end = (i + 1) * self.mb_size

                x_mb = test_x[start:end]
                y_mb = test_y[start:end]

                logits = self.model(x_mb)

                loss = self.criterion(y_mb, logits)
                ave_loss += loss

                pred_label = tf.argmax(logits, axis=1)

                y_hat.append(pred_label.numpy())
                true_y.append(y_mb)

            if eval_protocol:
                results = eval_protocol.get_results(true_y, y_hat, t)
                acc, accs = results

            ave_loss /= test_y.shape[0]

            print("Task {0}: Avg Loss {1}; Avg Acc {2}"
                  .format(t, ave_loss, acc))
            res[t].append((ave_loss, acc, accs))

            self.after_task_test()

        self.after_test()

        return res

    def before_epoch(self):
        super(StrategyTensorflow, self).before_epoch()

    def before_iteration(self):
        super(StrategyTensorflow, self).before_iteration()

    def before_weights_update(self):
        super(StrategyTensorflow, self).before_weights_update()

    def after_iter_ended(self):
        super(StrategyTensorflow, self).after_iter_ended()

    def after_epoch_ended(self):
        super(StrategyTensorflow, self).after_epoch_ended()

    def before_test(self):
        super(StrategyTensorflow, self).before_test()

    def after_test(self):
        super(StrategyTensorflow, self).after_test()

    def before_task_test(self):
        super(StrategyTensorflow, self).before_task_test()

    def after_task_test(self):
        super(StrategyTensorflow, self).after_task_test()


