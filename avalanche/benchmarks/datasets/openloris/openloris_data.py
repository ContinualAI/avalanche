#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-11-2020                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

import os
import sys
from zipfile import ZipFile

from google_drive_downloader import GoogleDriveDownloader as gdd

if sys.version_info[0] >= 3:
    pass
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    pass

filename = [
    ('train.zip',
     '1ChoBAGcQ_wkclPXsel8CjJHC0tD7b4ga'),
    ('test.zip',
     '1J7_ljcwSZNXo6KwlhRZoG0kiEcRK7U6x'),
    ('LUP.pkl',
     '1Os8T30NZ3ZU8liHQPeVbo2nlOoPZuDSV'),
    ('Paths.pkl',
     '1KnuYLdlG3VQrhgbtIANLki81ah8Thezj'),
    ('Labels.pkl',
     '1GkmOxIAvmjSwo22UzmZTSlw8NSmU5Q9H')
]

#11jgiPB2Z9WRI3bW6VSN8fJZgwFl5mLsF


class OPENLORIS_DATA(object):
    """
    OpenlORIS downloader.
    """

    def __init__(self, data_folder='data/'):
        """
        Args:
            data_folder (string): folder in which to download openloris dataset.
        """

        if os.path.isabs(data_folder):
            self.data_folder = data_folder
        else:
            self.data_folder = os.path.join(os.path.dirname(__file__),
                                            data_folder)

        try:
            # Create target Directory for openloris data
            os.makedirs(self.data_folder)
            print("Directory ", self.data_folder, " Created ")
            self.download = True
            self.download_openloris()

        except OSError:
            self.download = False
            print("Directory ", self.data_folder, " already exists")

    def download_openloris(self):

        for name in filename:
            print("Downloading " + name[1] + "...")
            gdd.download_file_from_google_drive(file_id=name[1],
                                                dest_path=os.path.join(self.data_folder, name[0]))
            if name[1].endswith('.zip'):
                with ZipFile(
                        os.path.join(self.data_folder, name[0]), 'r') as zipf:
                    print('Extracting OpenLORIS images...')
                    zipf.extractall(self.data_folder)
                    print('Done!')

        print("Download complete.")