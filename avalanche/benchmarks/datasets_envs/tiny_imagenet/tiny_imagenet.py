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

import csv
import os
import sys
from zipfile import ZipFile


if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve


filename = ('tiny-imagenet-200.zip', \
    'http://cs231n.stanford.edu/tiny-imagenet-200.zip' )

class TinyImageNet_data(object):
    """
    Tiny ImageNet dataset loader
    """

    def __init__(self, data_folder='data'):
        """
        Args:
            :param string data_folder: folder in which to download dataset
        """

        if os.path.isabs(data_folder):
            self.data_folder = data_folder
        else:
            self.data_folder = os.path.join(
                os.path.dirname(__file__), \
                data_folder
                )
                
        self.data_folder = os.path.join(self.data_folder, 'tiny-imagenet-200')

        try:
            # Create target Directory for Tiny ImageNet data
            os.mkdir(self.data_folder)
            print("Directory ", self.data_folder, " Created ")
            self.download = True
            self.download_tinyImageNet()

        except OSError:
            self.download = False
            print("Directory ", self.data_folder, " already exists")


        self.label2id, self.id2label = self.labels2dict()


    def download_tinyImageNet(self):

        print("Downloading " + filename[1] + "...")
        urlretrieve(filename[1], os.path.join(self.data_folder, filename[0]))

        with ZipFile( os.path.join(self.data_folder, filename[0]), 'r') as zipf:
                print('Extracting Tiny ImageNet images...') 
                zipf.extractall(self.data_folder) 
                print('Done!')

        print("Download complete.")

    def labels2dict(self):
        """
        Return dictionaries to convert class names into progressive ids
        and viceversa
        """

        label2id = {}
        id2label = {}

        with open(os.path.join(self.data_folder, 'wnids.txt'), 'r') as f:

            reader = csv.reader(f)
            curr_idx = 0
            for l in reader:
                if l[0] not in label2id:
                    label2id[l[0]] = curr_idx
                    id2label[curr_idx] = l[0]
                    curr_idx += 1
        
        return label2id, id2label
