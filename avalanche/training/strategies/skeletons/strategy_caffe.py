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
from avalanche.training.utils.caffe_utils import create_solver, \
    compute_one_hot
from avalanche import ARTIFACTS_BP
from avalanche.evaluation import EvalProtocol

import caffe
import os
import numpy as np


class StrategyCaffe(Strategy):

    def __init__(self, model, optimizer_proto=None, mb_size=256,
                 train_ep=2, multi_head=False, use_cuda=True, preproc=None):

        # inits
        if use_cuda:
            caffe.set_mode_gpu()
            caffe.set_device(0)

        # preprocessing function
        self.preproc = preproc

        # proto filenames / locations
        self.model_fn = ARTIFACTS_BP + 'net.prototxt'
        self.optimizer_fn = ARTIFACTS_BP + 'optimizer.prototxt'

        # save model as prototxt file
        self.model_proto = model.get_proto()
        with open(self.model_fn, 'w') as f:
            f.writelines(self.model_proto)

        # save optimizer as prototxt file
        if optimizer_proto is None:
            # create solver proto if not provided
            self.solver_proto = create_solver(
                self.model_fn, 0.01, momentum=0.9, weight_decay=0.005
            )
        else:
            self.solver_proto = optimizer_proto
        with open(self.optimizer_fn, 'w') as f:
            f.writelines(self.solver_proto)

        # loading the solver and the model
        self.optimizer = caffe.get_solver(self.optimizer_fn)
        self.model = self.optimizer.net

        # other parameters
        self.mb_size = mb_size
        self.train_ep = train_ep
        self.multi_head = multi_head
        self.use_cuda = use_cuda

        # to be updated during training / test
        self.cur_ep = None
        self.cur_train_t = None
        self.cur_test_t = None

        super(StrategyCaffe, self).__init__()

    def train(self, x, y, t):

        self.cur_ep = 0
        self.cur_train_t = t

        if len(x.shape) < 4:
            # the channel is missing: needed for caffe
            x = np.expand_dims(x, axis=1)

        if self.preproc:
            x = self.preproc(x)

        (train_x, train_y), it_x_ep = pad_data(
            [x, y], self.mb_size
        )

        shuffle_in_unison(
            [train_x, train_y], 0, in_place=True
        )

        num_class = self.model.blobs['out'].data.shape[1]
        one_hot_y = compute_one_hot(train_y, num_class)

        correct_cnt, ave_loss = 0, 0
        acc = None

        for ep in range(self.train_ep):
            self.before_epoch()

            for it in range(it_x_ep):
                self.before_iteration()

                start = it * self.mb_size
                end = (it + 1) * self.mb_size

                self.model.blobs['data'].data[...] = train_x[start:end]
                self.model.blobs['label'].data[...] = train_y[start:end]
                if 'target' in self.model.blobs.keys():
                    self.model.blobs['target'].data[...] = one_hot_y[start:end]

                self.optimizer.step(1)

                # self.model.forward()
                logits = self.model.blobs['softmax'].data
                pred_label = np.argmax(logits, axis=1)

                for i, lab in enumerate(pred_label):
                    if int(lab) == train_y[start:end][i]:
                        correct_cnt += 1

                loss = self.model.blobs['loss'].data
                ave_loss += loss

                # TODO: expone these methods with custom caffe version
                # self.optimizer.backward()
                # self.before_weights_update()
                # self.optimizer.update()

                acc = correct_cnt / \
                      ((it + 1) * self.mb_size + ep * x.shape[0])
                ave_loss /= ((it + 1) * self.mb_size + ep * x.shape[0])

                if it % 100 == 0:
                    print(
                        '==>>> ep: {}, it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                        .format(ep, it, ave_loss, acc)
                    )

                self.after_iter_ended()

            self.after_epoch_ended()
            self.cur_ep += 1

        return ave_loss, acc

    def test(self, test_set, eval_protocol=EvalProtocol(metrics=[ACC])):

        self.before_test()

        res = {}
        ave_loss = 0
        acc = None
        accs = None

        for (x, y), t in test_set:

            if len(x.shape) < 4:
                # the channel is missing: needed for caffe
                x = np.expand_dims(x, axis=1)

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

                self.model.blobs['data'].data[...] = test_x[start:end]
                self.model.blobs['label'].data[...] = test_y[start:end]

                self.model.forward()
                logits = self.model.blobs['softmax'].data

                pred_label = np.argmax(logits, axis=1)

                loss = self.model.blobs['loss'].data
                ave_loss += loss

                y_hat.append(pred_label)
                true_y.append(test_y[start:end])

            results = eval_protocol.get_results(
                true_y, y_hat, self.cur_train_t, self.cur_test_t
            )
            acc, accs = results[ACC]

            ave_loss /= test_y.shape[0]

            print("Task {:}: Avg Loss {:.4f}; Avg Acc {:.4f}"
                  .format(t, ave_loss, acc))
            res[t].append(results)

            self.after_task_test()

        self.after_test()

        return res

    def before_epoch(self):
        super(StrategyCaffe, self).before_epoch()

    def before_iteration(self):
        super(StrategyCaffe, self).before_iteration()

    def before_weights_update(self):
        super(StrategyCaffe, self).before_weights_update()

    def after_iter_ended(self):
        super(StrategyCaffe, self).after_iter_ended()

    def after_epoch_ended(self):
        super(StrategyCaffe, self).after_epoch_ended()

    def before_test(self):
        super(StrategyCaffe, self).before_test()

    def after_test(self):
        super(StrategyCaffe, self).after_test()

    def before_task_test(self):
        super(StrategyCaffe, self).before_task_test()

    def after_task_test(self):
        super(StrategyCaffe, self).after_task_test()


