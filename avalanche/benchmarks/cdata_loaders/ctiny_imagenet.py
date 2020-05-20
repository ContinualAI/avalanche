#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-05-2020                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

import logging
import os
import random
import csv
import numpy as np
from PIL import Image
from avalanche.benchmarks.datasets_envs import TinyImageNet_data


class CTinyImageNet(object):
    """
    Tiny ImageNet dataset.
    Tasks can be defined by the user providing class names for each task.
    Otherwise, num_tasks of 2 classes each will be used.
    """

    def __init__(self, root=None, num_tasks=5, num_classes_per_task=2,
                classes_per_task=None, normalize=True):
        """
        Args:
            :param string root: path in which to download data. None to use
                    default path. Default None.

            :param int num_tasks: Number of tasks to train on. By default,
                    each task trains on pairs of classes. To specify a custom 
                    set of classes, use classes_per_task parameter. Default 5.
                    If classes_per_task is not None, num_tasks will be ignored.

            :param int num_classes_per_task: Number of classes for each task.
                    Ignored if classes_per_task is not None.

            :param list classes_per_task: A list of tuples. Each tuple contains
                    classes names from the original dataset to be used
                    for training and test. If None, random classes will be
                    used. Default None.

            :param bool normalize: True to normalize pixels in [0,1]. False to
                    keep pixels in [0, 255]. Default True.
        """

        # download dataset and set current data folder
        if root is None:
            self.tiny_data = TinyImageNet_data()
        else:
            self.tiny_data = TinyImageNet_data(root)

        self.root = self.tiny_data.data_folder

        self.normalize = normalize

        # compute classes_per_task (list of tuples containing class ids)
        # compute num_tasks, that is lenght of classes_per_task
        if classes_per_task is not None:

            try:
                assert(type(classes_per_task) == list)
                assert(all(
                    [(type(el) == tuple or type(el) == list) 
                    for el in classes_per_task ])
                )
            except AssertionError:
                logging.error("classes_per_task must be list of tuples.")
            
            self.num_tasks = len(classes_per_task)
            
            try:
                self.classes_per_task = map(
                    lambda u: tuple([self.tiny_data.label2id[el] for el in u]),
                    classes_per_task
                )
            except KeyError:
                logging.error("Class names provided are not valid.")

        else:
            self.num_tasks = num_tasks
            classes = list(range(200))
            random.shuffle(classes)
            self.classes_per_task = [ 
                tuple(classes[i:i+num_classes_per_task]) \
                for i in range(0, self.num_tasks*num_classes_per_task,\
                    num_classes_per_task)
                ]


        self.tasks_id = list(range(self.num_tasks))
        self.iter = 0

        self.all_train_sets, self.all_test_sets = [], []

        print("preparing CL benchmark...")
        # prepare datasets for each task
        for task_id, classes in enumerate(self.classes_per_task):

            train_set, test_set = self.load_classes(classes)
            self.all_train_sets.append(train_set)
            self.all_test_sets.append(test_set)

        print("Benchmark ready!")

    def load_classes(self, classes):
        """
        Load images and targets corresponding to a set of classes
        Args:
            classes: tuple containing classes ids
        
        :return: train_set, test_set: (train_X,train_y), (test_X,test_y)
        """
        
        train_set = [[], []]
        test_set = [[], []]

        for class_id in classes:

            class_name = self.tiny_data.id2label[class_id]

            train_X = self.get_train_images(class_name)

            # (batch_size)
            train_Y = np.array(class_id).repeat(train_X.shape[0])

            train_set[0].append(train_X)
            train_set[1].append(train_Y)
        
            # test set
            test_X = self.get_test_images(class_name)
            test_Y = np.array(class_id).repeat(test_X.shape[0])

            test_set[0].append(test_X)
            test_set[1].append(test_Y)

        train_set[0] = np.concatenate(train_set[0], axis=0)
        train_set[1] = np.concatenate(train_set[1], axis=0)
        test_set[0] = np.concatenate(test_set[0], axis=0)
        test_set[1] = np.concatenate(test_set[1], axis=0)

        return train_set, test_set

    def get_train_images(self, class_name):
        train_img_folder = os.path.join(self.tiny_data.data_folder, \
                'train', class_name, 'images')

        img_paths = [os.path.join(train_img_folder, f) 
                for f in os.listdir(train_img_folder) 
                if os.path.isfile(os.path.join(train_img_folder, f))]

        # (batch_size, W, H, n_channels)
        train_X = self.load_images_from_paths(img_paths)

        return train_X

    def get_test_images(self, class_name):

        val_img_folder = os.path.join(self.tiny_data.data_folder, 'val', \
            'images')

        valid_names = []
        # filter validation images by class using appropriate file
        with open(os.path.join(
            self.tiny_data.data_folder, 'val', 'val_annotations.txt'), 'r') \
            as f:

            reader = csv.reader(f, dialect='excel-tab')
            for l in reader:
                if l[1] == class_name:
                    valid_names.append(l[0])

        img_paths = [os.path.join(val_img_folder, f) for f in valid_names]
        
        # (batch_size, W, H, n_channels)
        test_X = self.load_images_from_paths(img_paths)
            
        return test_X

    def load_images_from_paths(self, paths):

        X = [np.array(Image.open(fname).convert('RGB'))
                for fname in paths]

        # batch, channel, H, W
        X = np.moveaxis(np.stack(X), 3, 1)

        if self.normalize:
            X = X / 255

        return X


    def __iter__(self):
        return self

    def __next__(self):

        if self.iter == self.num_tasks:
            raise StopIteration

        train_set = self.all_train_sets[self.iter]

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