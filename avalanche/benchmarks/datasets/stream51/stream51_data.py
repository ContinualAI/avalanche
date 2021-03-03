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
import sys
import logging
from zipfile import ZipFile
import shutil

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve

name = ('Stream-51.zip', 'http://klab.cis.rit.edu/files/Stream-51.zip')


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
        self.log.info("Downloading " + name[1] + "...")
        urlretrieve(name[1], os.path.join(self.data_folder, name[0]))

        if name[1].endswith('.zip'):
            lfilename = os.path.join(self.data_folder, name[0])
            with ZipFile(lfilename, 'r') as zipf:
                for member in zipf.namelist():
                    filename = os.path.basename(member)
                    # skip directories
                    if not filename:
                        continue

                    # copy file (taken from zipfile's extract)
                    source = zipf.open(member)
                    if 'json' in filename:
                        target = open(os.path.join(self.data_folder, filename),
                                      "wb")
                    else:
                        dest_folder = os.path.join(
                            *(member.split(os.path.sep)[1:-1]))
                        dest_folder = os.path.join(self.data_folder,
                                                   dest_folder)
                        if not os.path.exists(dest_folder):
                            os.makedirs(dest_folder)
                        target = open(os.path.join(dest_folder, filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)

                self.log.info('Done!')
            os.remove(lfilename)

        self.log.info("Download complete.")


__all__ = [
    'STREAM51_DATA'
]
