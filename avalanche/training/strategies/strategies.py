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

"""
Strategies generator: It's the class in charge of creating the framework
dependent strategy object.
"""
# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from avalanche.constants import DLBackEnd
import sys


def check_if_imported(string):

    import_list = sys.modules.keys()
    for import_name in import_list:
        if string in import_name:
            return True
    else:
        return False


def Naive(kwargs, dl_backend=None):

    if dl_backend is None:
        # automatically detect framework used base on the model
        print()

        if check_if_imported('torch'):
            print("PyTorch detected, using this framework.")
            from avalanche.training.strategies.naive.naive_pytorch \
                import NaivePytorch
            return NaivePytorch(**kwargs)

        elif check_if_imported('tensorflow'):
            print("Tensorflow detected, using this framework.")
            from avalanche.training.strategies.naive.naive_tensorflow \
                import NaiveTensorflow
            return NaiveTensorflow(**kwargs)

        elif check_if_imported('caffe'):
            print("Caffe detected, using this framework.")
            from avalanche.training.strategies.naive.naive_caffe \
                import NaiveCaffe
            return NaiveCaffe(**kwargs)

        else:
            print("Automatic detection of this framework is not supported.")
            raise NotImplemented

    elif dl_backend is DLBackEnd.PYTORCH:
        from avalanche.training.strategies.naive.naive_pytorch \
            import NaivePytorch
        return NaivePytorch(**kwargs)

    elif dl_backend is DLBackEnd.TENSORFLOW:
        from avalanche.training.strategies.naive.naive_tensorflow \
            import NaiveTensorflow
        return NaiveTensorflow(**kwargs)

    elif dl_backend is DLBackEnd.CAFFE:
        from avalanche.training.strategies.naive.naive_caffe \
            import NaiveCaffe
        return NaiveCaffe(**kwargs)

    else:
        raise NotImplemented





