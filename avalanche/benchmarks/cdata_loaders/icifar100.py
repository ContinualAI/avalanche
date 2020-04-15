#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-11-2017                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Data Loader for the CIFAR10-100 continual learning benchmark. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# other imports
import logging
import numpy as np

from avalanche.benchmarks.datasets_envs.cifar import  remove_some_labels, \
    read_data_from_pickled

class ICifar100(object):
    """ iCifar100 (from iCaRL paper) benchmark loader. """

    def __init__(self,
                 root_cifar100='/home/admin/data/cifar100/cifar-100-python/',
                 fixed_test_set=True,
                 num_batch=10,
                 cumulative=False,
                 run=0,
                 seed=0,
                 task_sep=False):
        """" Initialize Object """

        self.num_batch = num_batch
        self.classxbatch = 10
        self.tot_num_labels = 100
        self.iter = 0
        self.fixed_test_set = fixed_test_set
        self.cumulative = cumulative
        self.run = run
        self.seed = seed
        self.task_sep = task_sep

        # Getting root logger
        self.log = logging.getLogger('mylogger')

        # load cifar100 images
        self.train_set, self.test_set = read_data_from_pickled(
            root_cifar100, 50000, 10000, 32
        )

        # compute which labels select for each batch given run
        labels = list(range(self.tot_num_labels))
        np.random.seed(self.seed)
        if self.run != 0:
            for i in range(self.run):
                np.random.shuffle(labels)

        self.batch2labels = {
            i: labels[i * self.classxbatch:(i + 1) * self.classxbatch]
            for i in range(10)
        }
        self.log.debug('Labels order: ' + str(labels) + '\n')

        # to be filled
        self.all_test_sets = []
        self.tasks_id = []

        print("preparing CL benchmark...")
        for i in range(self.num_batch):

            all_labels = range(self.tot_num_labels)
            te_curr_labels = self.batch2labels[i]
            te_labs2remove = [j for j in all_labels if j not in te_curr_labels]
            test_x, test_y = remove_some_labels(self.test_set, te_labs2remove)
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

        tr_curr_labels = self.batch2labels[self.iter]
        te_curr_labels = []
        for i in range(self.iter+1):
            te_curr_labels += self.batch2labels[i]

        all_labels = range(self.tot_num_labels)

        if self.cumulative:
            # we remove only never seen before labels
            not_remove = []
            for i in range(self.iter + 1):
                not_remove += self.batch2labels[i]
            tr_labs2remove = [i for i in all_labels if i not in not_remove]
        else:
            # we remove only labels not belonging to the current batch
            tr_labs2remove = [i for i in all_labels if i not in tr_curr_labels]

        train_x, train_y = remove_some_labels(self.train_set, tr_labs2remove)


        # get ready for next iter
        self.iter += 1

        if self.task_sep:
            t = self.iter-1
        else:
            t = 0

        return train_x, train_y, t

    def get_growing_testset(self):
        """
        Return the growing test set (test set of tasks encountered so far.
        """

        # up to the current train/test set
        # remember that self.iter has been already incremented at this point
        return list(zip(
            self.all_test_sets[:self.iter], self.tasks_id[:self.iter])
        )

    def get_full_testset(self):
        """
        Return the test set (the same for each inc. batch).
        """
        return list(zip(self.all_test_sets, self.tasks_id))

    next = __next__  # python2.x compatibility.


if __name__ == "__main__":

    # Create the dataset object
    dataset = ICifar100()

    # Get the fixed test set
    test_set = dataset.get_full_testset()

    # loop over the training incremental batches
    for train_batch in dataset:
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y, t = train_batch

        print("task: {}, x: {}, y: {}".format(t, train_x.shape, train_y.shape))

    for test_batch in test_set:
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        (test_x, test_y), t = test_batch

        print("test task: {}, x: {}, y: {}".format(t, test_x.shape,
                                                test_y.shape))
