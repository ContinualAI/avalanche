################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-05-2020                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

import os
import sys
import logging
from zipfile import ZipFile 


if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve


filename = [
    ('core50_128x128.zip',
     'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'),
    ('paths.pkl', 'https://vlomonaco.github.io/core50/data/paths.pkl'),
    ('LUP.pkl', 'https://vlomonaco.github.io/core50/data/LUP.pkl'),
    ('labels.pkl', 'https://vlomonaco.github.io/core50/data/labels.pkl'),
    ('core50_imgs.npz',
     'http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz'),
    ('batches_filelists.zip',
     'https://vlomonaco.github.io/core50/data/batches_filelists.zip'),
    ('batches_filelists_NICv2.zip',  
     'https://vlomonaco.github.io/core50/data/batches_filelists_NICv2.zip')
]


class CORE50_DATA(object):
    """
    CORE50 downloader.
    """

    def __init__(self, data_folder='data/'):
        """
        Args:
            data_folder (string): folder in which to download core50 dataset. 
        """

        self.log = logging.getLogger("avalanche")

        if os.path.isabs(data_folder):
            self.data_folder = data_folder
        else:
            self.data_folder = os.path.join(os.path.dirname(__file__),
                                            data_folder)

        try:
            # Create target Directory for CORE50 data
            os.makedirs(self.data_folder)
            self.log.info("Directory ", self.data_folder, " Created ")
            self.download = True
            self.download_core50()

        except OSError:
            self.download = False
            self.log.error("Directory ", self.data_folder, " already exists")

    def download_core50(self):

        for name in filename:
            self.log.info("Downloading " + name[1] + "...")
            urlretrieve(name[1], os.path.join(self.data_folder, name[0]))

            if name[1].endswith('.zip'):
                with ZipFile(
                        os.path.join(self.data_folder, name[0]), 'r') as zipf:
                    self.log.info('Extracting CORe50 images...')
                    zipf.extractall(self.data_folder) 
                    self.log.info('Done!')

        self.log.info("Download complete.")


__all__ = [
    'CORE50_DATA'
]
