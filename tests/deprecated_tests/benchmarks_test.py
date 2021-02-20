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

""" benchmark tests """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from avalanche.benchmarks import CMNIST, CORE50, ICIFAR10
from avalanche.benchmarks import CMNIST, CORE50, CImageNet, CTinyImageNet
import torchvision.transforms as transforms

if __name__ == "__main__":

    ##################
    ## MNIST TEST   ##
    ##################

    mode = ['perm', 'split', 'rot']

    for m in mode:
        cmnist = CMNIST(mode=m, num_batch=5)
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
            
    ##################
    ## ImageNet TEST #
    ##################

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    imagenet_loader = CImageNet(root='/ssddata/ilsvrc-data/', num_initial=500,
                                num_batch=100, sample_train=100, sample_test=10,
                                transform=transform)
    # Get the fixed test set
    full_testset = imagenet_loader.get_full_testset()

    # loop over the training incremental batches
    for i, (x, y, t) in enumerate(imagenet_loader):
        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y: {1}"
              .format(x.shape, y.shape))

        # use the data
        pass

    ######################
    ## TinyImageNet TEST #
    ######################

    ctiny = CTinyImageNet()
    ctiny2 = CTinyImageNet(classes_per_task=[
        ['n02788148', 'n02909870', 'n03706229'],
        ['n06596364', 'n01768244', 'n02410509'],
        ['n04487081', 'n03250847', 'n03255030']
    ])

    ctiny.get_full_testset()
    ctiny.get_growing_testset()
    ctiny2.get_full_testset()
    ctiny2.get_growing_testset()

    for x,y,t in ctiny:
        assert( type(x) == np.ndarray )
        assert( type(y) == np.ndarray )
        assert( type(t) == int )
        break

    for x,y,t in ctiny2:
        assert( type(x) == np.ndarray )
        assert( type(y) == np.ndarray )
        assert( type(t) == int )
        break
