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

""" benchmark tests """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from avalanche.benchmarks import CMNIST, CORE50, ICIFAR10

if __name__ == "__main__":

    ##################
    ## MNIST TEST   ##
    ##################

    mode = ['perm', 'split', 'rot']

    for m in mode:
        cmnist = CMNIST(mode=m)
        cmnist.get_full_testset()
        cmnist.get_growing_testset()

        for x,y,t in cmnist:
            assert( type(x) == np.ndarray )
            assert( type(y) == np.ndarray )
            assert( type(t) == int )
            break
    ##################
    ## ICIFAR10 TEST ##
    ##################

    icifar10 = ICIFAR10()

    icifar10.get_full_testset()

    for x, y, t in icifar10:
        assert (type(x) == np.ndarray)
        assert (type(y) == np.ndarray)
        assert (type(t) == int)
        break
        
    ##################
    ## CORE 50 TEST ##
    ##################

    scenarios = [
        'ni',
        'nc',
        'nic',
        'nicv2_79',
        'nicv2_196',
        'nicv2_391'
    ]

    # test core50 downloader on default path
    cdata = CORE50()
    del cdata
    # now core50 must be able to find data without downloading again
    cdata = CORE50()
    assert(not cdata.core_data.download)

    # check that all scenarios are working
    for sc in scenarios:
        cdata = CORE50(scenario=sc)

        cdata.get_full_testset(reduced=True)
        cdata.get_full_testset(reduced=False)

        # this is not implemented, yet!
        #cdata.get_growing_testset(reduced=True)

        for x,y,t in cdata:
            assert( type(x) == np.ndarray )
            assert( type(y) == np.ndarray )
            assert( type(t) == int )
            break
