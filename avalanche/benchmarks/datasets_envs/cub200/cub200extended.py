#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-05-2020                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

import os
import sys
import shutil


if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve

filename = [
    ('cub200_images_and_annotation.tgz',
     'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'),
    ('segmentations.tgz',
     'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz'),
    ('README',
     'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/README.txt')
]


class CUB200EXTENDED_DATA(object):
    """
    CUB200 downloader.
    """

    def __init__(self, data_folder='data/cub200extended/'):
        """
        Args:
            data_folder (string): folder in which to download CUB200 extended dataset.
        """

        if os.path.isabs(data_folder):
            self.data_folder = data_folder
        else:
            self.data_folder = os.path.join(os.path.dirname("__file__"),
                                            data_folder)

        try:
            # Create target Directory for CUB200 data
            os.mkdir(self.data_folder)
            print(f"Directory {self.data_folder} created")
            self.download = True
            self.download_cub200extended()

        except OSError:
            self.download = False
            print(f"Directory {self.data_folder} already exists")

    def download_cub200extended(self):
        for name in filename:
            print(f"Downloading {name[1]}...")
            urlretrieve(name[1], os.path.join(self.data_folder, name[0]))

            if name[1].endswith('.tgz'):
                print(f'Extracting {name[0]}...')
                shutil.unpack_archive(os.path.join(self.data_folder, name[0]), self.data_folder)
                print('Done!')

        print("Download complete.")


if __name__ == '__main__':
    cub200 = CUB200EXTENDED_DATA()
