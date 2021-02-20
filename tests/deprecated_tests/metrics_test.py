################################################################################
# Copyright (c) 2019. ContinualAI. All rights reserved.                        #
# Copyrights licensed under the MIT License.                                   #
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

import torch
from avalanche.evaluation.metrics import ACC, CF, RAMU, DiskUsage, CM, \
        TimeUsage, GPUUsage, CPUUsage


if __name__ == '__main__':

    metrics = {
        'acc': ACC(),
        'cf': CF(),
        'ramu': RAMU(),
        'disk': DiskUsage(),
        'disk_io': DiskUsage(disk_io=True),
        'cm': CM(),
        'time': TimeUsage(),
        'gpu': GPUUsage(gpu_id=0),
        'cpu': CPUUsage()
    }

    n_tasks = 3

    for t in range(n_tasks):

        tensors = ( 
            torch.randint(low=0, high=10, size=(20, 1)),
            torch.randint(low=0, high=10, size=(20, 1))
        )

        tensors_list = (
            [torch.randint(low=0, high=10, size=(20, 1)) for _ in range(3)],
            [torch.randint(low=0, high=10, size=(20, 1)) for _ in range(3)]
        )

        for y, y_hat in [tensors, tensors_list]:
            for name, metric in metrics.items():
                if name in ['acc', 'cm']:
                    metric.compute(y, y_hat)
                elif name in ['disk', 'disk_io', 'ramu', 'time', 'cpu', 'gpu']:
                    metric.compute(t)
                elif name in ['cf']:
                    metric.compute(y, y_hat, t, t)
