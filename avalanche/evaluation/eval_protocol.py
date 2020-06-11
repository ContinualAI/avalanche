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

""" Common metrics for CL. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from .metrics import ACC, CF, RAMU, CM, CPUUsage, GPUUsage, DiskUsage, TimeUsage
import numpy as np
from .tensorboard import TensorboardLogging


class EvalProtocol(object):

    def __init__(self, metrics=[ACC()], tb_logdir="../logs/test"):

        self.metrics = []
        for metric in metrics:
            self.metrics.append(metric)
        self.tb_logging = TensorboardLogging(tb_logdir=tb_logdir)

        # to be updated
        self.cur_acc = {}
        self.global_step = 0
        self.cur_classes = None
        self.prev_acc_x_class = {}
        self.cur_t = None

    def get_results(self, true_y, y_hat, train_t, test_t):
        """ Compute results based on accuracy """

        results = {}
        for metric in self.metrics:
            if isinstance(metric, ACC):
                results[ACC] = metric.compute(true_y, y_hat)
            elif isinstance(metric, CF):
                results[CF] = metric.compute(true_y, y_hat, train_t, test_t)
            elif isinstance(metric, RAMU):
                results[RAMU] = metric.compute(train_t)
            elif isinstance(metric, CM):
                results[CM] = metric.compute(true_y, y_hat)
            elif isinstance(metric, CPUUsage):
                results[CPUUsage] = metric.compute(train_t)
            elif isinstance(metric, GPUUsage):
                results[GPUUsage] = metric.compute(train_t)
            elif isinstance(metric, DiskUsage):
                results[DiskUsage] = metric.compute(train_t)
            elif isinstance(metric, TimeUsage):
                results[TimeUsage] = metric.compute(train_t)
            else:
                raise ValueError("Unknown metric")

        self.global_step += 1

        return results

    def update_tb_test(self, res, step):
        """ Function to update tensorboard """

        if self.tb_logging.writer is not None:

            acc_scalars = {}
            loss_scalars = {}
            cf_scalars = {}
            class_scalars = {}
            out_class_diff = []
            in_class_diff = []

            for t, (ave_loss_i, acc_i, accs_i, _) in res.items():
                acc_scalars["task_"+str(t).zfill(3)] = acc_i
                loss_scalars["task_"+str(t).zfill(3)] = ave_loss_i
                for c in range(len(accs_i)):
                    class_scalars["task_"+str(t).zfill(3)+"_class_" + str(
                        c).zfill(3)] = accs_i[c]

                    if t not in self.prev_acc_x_class:
                        self.prev_acc_x_class[t] = accs_i
                    else:
                        acc_diffs = accs_i - self.prev_acc_x_class[t]
                        if t == self.cur_t:
                            # filter only current classes
                            acc_diffs_in = []
                            acc_diffs_out = []
                            for i, acc in enumerate(acc_diffs):
                                if i in self.cur_classes:
                                    acc_diffs_in.append(acc)
                                else:
                                    acc_diffs_out.append(acc)
                            in_class_diff.append(acc_diffs_in)
                            if acc_diffs_out:
                                for diff in acc_diffs_out:
                                    out_class_diff.append(diff)
                        else:
                            for diff in acc_diffs:
                                out_class_diff.append(diff)
                        self.prev_acc_x_class[t] = accs_i

            in_out_scalars = {
                "in_class": np.average(in_class_diff),
                "out_class": np.average(out_class_diff)
            }

            for metric in self.metrics:
                if isinstance(metric, CF):
                    for t, (_, _, _, m_res) in res.items():
                        cf_scalars["task_" + str(t).zfill(3)] = m_res[CF]
                    self.tb_logging.writer.add_scalars(
                        'Accuracy/CF', cf_scalars, step
                    )
                elif isinstance(metric, RAMU):
                    self.tb_logging.writer.add_scalar(
                        'Efficiency/RAM', res[0][3][RAMU], step
                    )
                elif isinstance(metric, CM):
                    for i, (t, (_, _, _, m_res)) in enumerate(res.items()):
                        if i == 0:
                            cm_imgs = np.expand_dims(m_res[CM], axis=0)
                        else:
                            cm_imgs = np.concatenate((cm_imgs,
                                                      np.expand_dims(m_res[CM],
                                                                     axis=0)))
                    self.tb_logging.writer.add_images(
                        "Confusion_matrices", cm_imgs, step)

            self.tb_logging.writer.add_scalars(
                'Accuracy/Test', acc_scalars, step
            )
            self.tb_logging.writer.add_scalars(
                'Loss/Test', loss_scalars, step
            )
            self.tb_logging.writer.add_scalars(
                'Accuracy/Test_x_class', class_scalars, step
            )
            self.tb_logging.writer.add_scalar(
                'Loss/Avg_Test_Loss', np.average(list(loss_scalars.values())),
                step
            )
            self.tb_logging.writer.add_scalar(
                'Accuracy/Avg_Test_Acc', np.average(list(acc_scalars.values())),
                step
            )
            self.tb_logging.writer.add_scalars(
                'Accuracy/Test_acc_diff', in_out_scalars, step
            )

            self.tb_logging.writer.flush()

    def update_tb_train(self, loss, acc, step, encountered_class, t):
        """ Function to update tensorboard """

        self.cur_classes = encountered_class
        self.cur_t = t

        if self.tb_logging.writer is not None:
            self.tb_logging.writer.add_scalar(
                'Accuracy/Train', acc, step
            )
            self.tb_logging.writer.add_scalar(
                'Loss/Train', loss, step
            )

        self.tb_logging.writer.flush()
