#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 23-07-2019                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Data Loader for the CORe50 Dataset """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# other imports
import numpy as np
import pickle as pkl
import os
import logging
from hashlib import md5
from PIL import Image


class CORE50(object):
    """ CORe50 Data Loader calss

    Args:
        root (string): Root directory of the dataset where ``core50_128x128``,
            ``paths.pkl``, ``LUP.pkl``, ``labels.pkl``, ``core50_imgs.npz``
            live. For example ``~/data/core50``.
        preload (string, optional): If True data is pre-loaded with look-up
            tables. RAM usage may be high.
        scenario (string, optional): One of the three scenarios of the CORe50
            benchmark ``ni``, ``nc``, ``nic``, `nicv2_79`,``nicv2_196`` and
             ``nicv2_391``.
        train (bool, optional): If True, creates the dataset from the training
            set, otherwise creates from test set.
        cumul (bool, optional): If True the cumulative scenario is assumed, the
            incremental scenario otherwise. Practically speaking ``cumul=True``
            means that for batch=i also batch=0,...i-1 will be added to the
            available training data.
        run (int, optional): One of the 10 runs (from 0 to 9) in which the
            training batch order is changed as in the official benchmark.
        start_batch (int, optional): One of the training incremental batches
            from 0 to max-batch - 1. Remember that for the ``ni``, ``nc`` and
            ``nic`` we have respectively 8, 9 and 79 incremental batches. If
            ``train=False`` this parameter will be ignored.
    """

    nbatch = {
        'ni': 8,
        'nc': 9,
        'nic': 79,
        'nicv2_79': 79,
        'nicv2_196': 196,
        'nicv2_391': 391
    }

    def __init__(self, root='', preload=False, scenario='ni', cumul=False,
                 run=0, start_batch=0, task_sep=False):
        """" Initialize Object """

        self.root = os.path.expanduser(root)
        self.preload = preload
        self.scenario = scenario
        self.cumul = cumul
        self.run = run
        self.batch = start_batch
        self.task_sep = task_sep

        if preload:
            print("Loading data...")
            bin_path = os.path.join(root, 'core50_imgs.bin')
            if os.path.exists(bin_path):
                with open(bin_path, 'rb') as f:
                    self.x = np.fromfile(f, dtype=np.uint8) \
                        .reshape(164866, 128, 128, 3)

            else:
                with open(os.path.join(root, 'core50_imgs.npz'), 'rb') as f:
                    npzfile = np.load(f)
                    self.x = npzfile['x']
                    print("Writing bin for fast reloading...")
                    self.x.tofile(bin_path)

        print("Loading paths...")
        with open(os.path.join(root, 'paths.pkl'), 'rb') as f:
            self.paths = pkl.load(f)

        print("Loading LUP...")
        with open(os.path.join(root, 'LUP.pkl'), 'rb') as f:
            self.LUP = pkl.load(f)

        print("Loading labels...")
        with open(os.path.join(root, 'labels.pkl'), 'rb') as f:
            self.labels = pkl.load(f)

        # to be changed
        self.tasks_id = []

        print("preparing CL benchmark...")
        for i in range(self.nbatch[scenario]):
            if self.task_sep:
                self.tasks_id.append(i)
            else:
                self.tasks_id.append(0)

    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """

        scen = self.scenario
        run = self.run
        batch = self.batch

        if self.batch == self.nbatch[scen]:
            raise StopIteration

        # Getting the right indexis
        if self.cumul:
            train_idx_list = []
            for i in range(self.batch + 1):
                train_idx_list += self.LUP[scen][run][i]
        else:
            train_idx_list = self.LUP[scen][run][batch]

        # loading data
        if self.preload:
            train_x = np.take(self.x, train_idx_list, axis=0)\
                      .astype(np.float32)
        else:
            print("Loading data...")
            # Getting the actual paths
            train_paths = []
            for idx in train_idx_list:
                train_paths.append(os.path.join(self.root, self.paths[idx]))
            # loading imgs
            train_x = self.get_batch_from_paths(train_paths).astype(np.float32)

        # In either case we have already loaded the y
        if self.cumul:
            train_y = []
            for i in range(self.batch + 1):
                train_y += self.labels[scen][run][i]
        else:
            train_y = self.labels[scen][run][batch]

        train_y = np.asarray(train_y, dtype=np.float32)

        # Update state for next iter
        self.batch += 1

        return train_x, train_y, self.tasks_id[self.batch-1]

    next = __next__  # python2.x compatibility.

    def get_full_testset(self):
        """
        Return the test set (the same for each inc. batch).
        """

        if self.task_sep:
            raise NotImplemented

        scen = self.scenario
        run = self.run

        test_idx_list = self.LUP[scen][run][-1]

        if self.preload:
            test_x = np.take(self.x, test_idx_list, axis=0).astype(np.float32)
        else:
            # test paths
            test_paths = []
            for idx in test_idx_list:
                test_paths.append(os.path.join(self.root, self.paths[idx]))

            # test imgs
            test_x = self.get_batch_from_paths(test_paths).astype(np.float32)

        test_y = self.labels[scen][run][-1]
        test_y = np.asarray(test_y, dtype=np.float32)

        return [[(test_x, test_y), self.tasks_id[self.batch-1]]]

    def get_growing_testset(self):
        """
        Return the growing test set (test set of tasks encountered so far.
        """
        raise NotImplemented

    @staticmethod
    def get_batch_from_paths(paths, compress=False, snap_dir='',
                             on_the_fly=True, verbose=False):
        """ Given a number of abs. paths it returns the numpy array
        of all the images. """

        # Getting root logger
        log = logging.getLogger('mylogger')

        # If we do not process data on the fly we check if the same train
        # filelist has been already processed and saved. If so, we load it
        # directly. In either case we end up returning x and y, as the full
        # training set and respective labels.
        num_imgs = len(paths)
        hexdigest = md5(''.join(paths).encode('utf-8')).hexdigest()
        log.debug("Paths Hex: " + str(hexdigest))
        loaded = False
        x = None
        file_path = None

        if compress:
            file_path = snap_dir + hexdigest + ".npz"
            if os.path.exists(file_path) and not on_the_fly:
                loaded = True
                with open(file_path, 'rb') as f:
                    npzfile = np.load(f)
                    x, y = npzfile['x']
        else:
            x_file_path = snap_dir + hexdigest + "_x.bin"
            if os.path.exists(x_file_path) and not on_the_fly:
                loaded = True
                with open(x_file_path, 'rb') as f:
                    x = np.fromfile(f, dtype=np.uint8) \
                        .reshape(num_imgs, 128, 128, 3)

        # Here we actually load the images.
        if not loaded:
            # Pre-allocate numpy arrays
            x = np.zeros((num_imgs, 128, 128, 3), dtype=np.uint8)

            for i, path in enumerate(paths):
                if verbose:
                    print("\r" + path + " processed: " + str(i + 1), end='')
                x[i] = np.array(Image.open(path))

            if verbose:
                print()

            if not on_the_fly:
                # Then we save x
                if compress:
                    with open(file_path, 'wb') as g:
                        np.savez_compressed(g, x=x)
                else:
                    x.tofile(snap_dir + hexdigest + "_x.bin")

        assert (x is not None), 'Problems loading data. x is None!'

        return x


if __name__ == "__main__":

    # Create the dataset object for example with the "NIC_v2 - 79 benchmark"
    # and assuming the core50 location in ~/core50/128x128/
    dataset = CORE50(root='/home/admin/Ior50N/128/', scenario="nicv2_79")

    # Get the fixed test set
    full_testset = dataset.get_full_testset()

    # loop over the training incremental batches
    for i, (x, y, t) in enumerate(dataset):

        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y: {0}"
              .format(x.shape, y.shape))

        # use the data
        pass

