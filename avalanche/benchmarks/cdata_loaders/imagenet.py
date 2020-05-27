#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-05-2020                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" Data Loader for the ImageNet continual learning benchmark. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# other imports
import logging
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from avalanche.benchmarks.datasets_envs import ImageNet



class CImageNet(object):
    """ ImageNet Data Loader class

    Args:
        root (string): The path to load ImageNet dataset.
        num_initial (int): The number of classes used for initial step.
        num_batch (int): The number of learning steps. The initial step is
        not counted. If num_initial > 0, num_batch will automatically plus 1.
        transform (torchvision.transforms): The transform to transfer PIL
        images to tensors.

    """


    def __init__(self, root='../data', num_initial = 500, num_batch=100,
                 sample_train=100, sample_test=10, transform=None):
        """" Initialize Object """

        imagenet = ImageNet(data_folder=root, download=False,
                        sample_train=sample_train, sample_test=sample_test)
        imagenet_data = imagenet.get_data()
        self.train_set, self.test_set = imagenet_data[0], imagenet_data[1]
        num_classes = len(imagenet.get_classes())
        classes_shuffled = np.random.permutation(num_classes).tolist()
        self.tasks = [classes_shuffled[:num_initial]] if num_initial>0 else []
        classes_shuffled = classes_shuffled[num_initial:] if num_initial>0 \
            else classes_shuffled

        self.num_batch = num_batch + 1 if num_initial >0 else num_batch
        self.tasks += [classes_shuffled[ib::self.num_batch]
                      for ib in range(self.num_batch)]
        self.transform = transform
        self.iter = 0



    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """

        if self.iter == self.num_batch:
            raise StopIteration

        train_set = []
        for lab in self.tasks[self.iter]:
            train_set += self.train_set[lab]

        images, labels = self.get_images(train_set)
        self.iter += 1
        
        return images, labels, self.iter-1

    next = __next__  # python2.x compatibility.


    def get_images(self, dataset):
        """
        Return images, labels according to the given dataset.
        :param dataset: the dataset to load
        :return: images, labels
        """

        images = []
        labels = []
        for fname, lab in dataset:
            try:
                img = Image.open(fname)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = self.transform(img)
                img = torch.unsqueeze(img, 0)
            except Exception:
                print('Image loading error occurs: %s'%fname)
                continue
            images.append(img)
            labels.append(lab)
        images = torch.cat(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return images, labels
        

    def get_growing_testset(self):
        """
        Return the growing test set (test set of tasks encountered so far.
        """

        # up to the current train/test set
        # remember that self.iter has been already incremented at this point
        # hence, self.iter is 1 when load the growing testset for
        # first learning set

        test_set = []
        for it in range(self.iter):
            for lab in self.tasks[it]:
                test_set += self.test_set[lab]

        images, labels = self.get_images(test_set)

        return images, labels, self.iter


    def get_full_testset(self):
        """
        Return the test set (the same for each inc. batch).
        """
        test_set = []
        for it in range(self.num_batch):
            for lab in self.tasks[it]:
                test_set += self.test_set[lab]

        images, labels = self.get_images(test_set)

        return [images, labels, self.iter]




if __name__ == "__main__":

    # Create the dataset object
    transform = transforms.Compose([transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize(mean=
                [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    imagenet_loader = CImageNet(root='/ssddata/ilsvrc-data/',
            num_initial = 500, num_batch=100, sample_train=100,
                sample_test=10, transform=transform)


    # Get the fixed test set
    full_testset = imagenet_loader.get_full_testset()

    # loop over the training incremental batches
    for i, (x, y, t) in enumerate(imagenet_loader):

        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y: {1}"
              .format(x.shape, y.shape))

        # use the data
        pass
