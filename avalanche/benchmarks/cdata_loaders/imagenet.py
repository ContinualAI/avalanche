#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. ContinualAI. All rights reserved.                        #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-020                                                             #
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



class imagenet(object):
    """ Cifar10/100 split benchmark loader. """

    def __init__(self,
                 root_imagenet='.'):
        """" Initialize Object """

    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """

        return None

    def get_growing_testset(self):
        """
        Return the growing test set (test set of tasks encountered so far.
        """

        # up to the current train/test set
        # remember that self.iter has been already incremented at this point
        return None

    def get_full_testset(self):
        """
        Return the test set (the same for each inc. batch).
        """
        return None

    next = __next__  # python2.x compatibility.



if __name__ == "__main__":

    # Create the dataset object
    dataset = imagenet()
