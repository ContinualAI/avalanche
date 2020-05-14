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


import os
import sys


if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve



class ImageNet(object):
    """ ImageNet dataset class

    Args:
        data_folder (string): The path to load ImageNet dataset.
        download (bool): Whether to download ImageNet dataset.

    """

    def __init__(self, data_folder='../data', download=False):


        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)

        self.train_path = os.path.join(data_folder, 'train')
        self.test_path = os.path.join(data_folder, 'val')

        try:
            if len(os.listdir(self.train_path)) != 1000 or len(os.listdir(self.test_path)) != 1000:
                msg = ("The ImageNet dataset should have 1000 train and val classes.")
                raise RuntimeError(msg)

            self.classes = os.listdir(self.train_path)
            sorted(self.classes)

            self.train_set = []
            self.test_set = []

            for lab, cls in enumerate(self.classes):
                files = os.listdir(os.path.join(self.train_path, cls))
                files.sort()
                self.train_set.append([[os.path.join(self.train_path, cls, fname), lab] for fname in files])

                files = os.listdir(os.path.join(self.test_path, cls))
                files.sort()
                self.test_set.append([[os.path.join(self.test_path, cls, fname), lab] for fname in files])

        except Exception:
            msg = ("There exist errors during dataset processing.")
            raise RuntimeError(msg)



    def get_classes(self):

        return self.classes


    def get_data(self):

        return [self.train_set, self.test_set]



if __name__ == '__main__':

    imagenet = ImageNet()
    imagenet_data = imagenet.get_data()

