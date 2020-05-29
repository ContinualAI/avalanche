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

""" Continual Data Loader for the MNIST continual learning benchmark. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import time

# other imports
import logging
from avalanche.benchmarks.datasets_envs import MNIST
from avalanche.benchmarks.utils import remove_some_labels


class CMNIST(object):
    """ Continuous MNIST benchmark data loader. """

    def __init__(self, bp=None, num_batch=5, mode='perm', task_sep=True,
                 eval_protocol=None):

        """" Initialize Object. mode={perm|split|rot}. """

        # checks
        if num_batch > 5 and mode == "split":
            raise Exception("Split mode cannot have more than 5 tasks!")

        self.bp = bp
        self.num_batch = num_batch
        self.iter = 0
        self.task_sep = task_sep
        self.eval_protocol = eval_protocol

        # Getting root logger
        self.log = logging.getLogger('mylogger')

        if self.bp is None:
            self.mnist = MNIST()
        else:
            self.mnist = MNIST(data_loc=bp)

        self.train_set, self.test_set = self.mnist.get_data()

        # to be filled
        self.all_train_sets = []
        self.all_test_sets = []
        self.tasks_id = []

        print("preparing CL benchmark...")
        for i in range(self.num_batch):

            if mode == "perm":

                train_x, test_x = self.mnist.permute_mnist(seed=i)
                train_y, test_y = self.train_set[1], self.test_set[1]
            elif mode == "rot":

                train_x, test_x = self.mnist.rotate_mnist(rotation=i*5)
                train_y, test_y = self.train_set[1], self.test_set[1]
            elif mode == "split":

                train_x, train_y = remove_some_labels(
                    self.train_set, [j for j in range(10)
                                     if j not in [i * 2, i * 2 + 1]]
                )
                test_x, test_y = remove_some_labels(
                    self.test_set, [j for j in range(10)
                                    if j not in [i * 2, i * 2 + 1]]
                )
            else:
                raise NotImplementedError()

            self.all_train_sets.append([train_x, train_y])
            self.all_test_sets.append([test_x, test_y])
            if self.task_sep:
                self.tasks_id.append(i)
            else:
                self.tasks_id.append(0)

    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """

        if self.iter == self.num_batch:
            raise StopIteration
        tic = time.time()
        train_set = self.all_train_sets[self.iter]
        print("Dataset loaded",time.time()-tic)

        # get ready for next iter
        self.iter += 1

        return train_set[0], train_set[1], self.tasks_id[self.iter-1]

    def get_full_testset(self):
        """
        Return the test set (the same for each inc. batch).
        """
        return list(zip(self.all_test_sets, self.tasks_id))

    def get_growing_testset(self):
        """
        Return the growing test set (test set of tasks encountered so far.
        """

        # up to the current train/test set
        # remember that self.iter has been already incremented at this point
        return list(zip(
            self.all_test_sets[:self.iter], self.tasks_id[:self.iter])
        )

    next = __next__  # python2.x compatibility.


if __name__ == "__main__":

    # Create the dataset object
    cmnist = CMNIST(mode="split", num_batch=5)

    test_full = cmnist.get_full_testset()

    # loop over the training incremental batches
    for i, (x, y, t) in enumerate(cmnist):
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.

        print("task {}: xsize: {}, ysize: {}".format(t, x.shape, y.shape))
