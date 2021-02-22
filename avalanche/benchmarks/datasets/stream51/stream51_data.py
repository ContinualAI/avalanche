################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-02-2021                                                             #
# Author: Tyler L. Hayes                                                       #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

import os
import logging
from google_drive_downloader import GoogleDriveDownloader as gdd


class STREAM51_DATA(object):
    """
    STREAM51 downloader.
    """

    def __init__(self, data_folder='data/'):
        """
        Args:
            data_folder (string): folder in which to download stream51 dataset.
        """

        self.log = logging.getLogger("avalanche")

        if os.path.isabs(data_folder):
            self.data_folder = data_folder
        else:
            self.data_folder = os.path.join(os.path.dirname(__file__),
                                            data_folder)

        try:
            # Create target Directory for STREAM51 data
            os.makedirs(self.data_folder)
            self.log.info("Directory ", self.data_folder, " Created ")
            self.download = True
            self.download_stream51()

        except OSError:
            self.download = False
            self.log.error("Directory ", self.data_folder, " already exists")

    def download_stream51(self):
        # DEPRECATED: the following function causes an error unless the data
        # is already archived on the local machine
        # several issues on github are raised about this problem:
        # https://github.com/ndrplz/google-drive-downloader
        # gdd.download_file_from_google_drive(file_id=
        #                                     '15huZ756N2cp1CCO4HxF-'
        #                                     'MVDsMx1LMoIn',
        #                                     dest_path=os.path.join(
        #                                         self.data_folder,
        #                                         'Stream-51.zip'),
        #                                     unzip=True)
        # self.log.info("Download complete.")
        raise NotImplementedError


__all__ = [
    'STREAM51_DATA'
]
