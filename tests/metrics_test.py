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


""" Test metrics """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from avalanche.evaluation.metrics import ACC, CF, RAMU, DiskUsage, CM


if __name__ == '__main__':

    metrics = {
        'acc': ACC(),
        'cf': CF(),
        'ramu': RAMU(),
        'disk': DiskUsage(),
        'disk_io': DiskUsage(disk_io = True),
        'cm': CM()
    }

    n_tasks = 3

    for t in range(n_tasks):

        y = np.random.randint(low=0, high=10, size=(20,1))
        y_hat = np.random.randint(low=0, high=10, size=(20, 1))

        for name, metric in metrics.items():
            if name in ['acc', 'cm']:
                metric.compute(y, y_hat)
            elif name in ['disk', 'disk_io', 'ramu']:
                metric.compute(t)
            elif name in ['cf']:
                metric.compute(y, y_hat, t, t)
