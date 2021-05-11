################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-11-2020                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" OpenLoris raw data handling module. It can support automatic download. """

import os
import sys
import gdown
from os.path import expanduser
from torchvision.datasets.utils import extract_archive

if sys.version_info[0] >= 3:
    pass
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    pass


class OPENLORIS_DATA(object):
    """
    OpenlORIS downloader.
    """

    filename = [
        ('train.zip',
         '11jgiPB2Z9WRI3bW6VSN8fJZgwFl5mLsF'),
        ('valid.zip',
         '1ChoBAGcQ_wkclPXsel8CjJHC0tD7b4ga'),
        ('test.zip',
         '1J7_ljcwSZNXo6KwlhRZoG0kiEcRK7U6x'),
        ('LUP.pkl',
         '1Os8T30NZ3ZU8liHQPeVbo2nlOoPZuDSV'),
        ('Paths.pkl',
         '1KnuYLdlG3VQrhgbtIANLki81ah8Thezj'),
        ('Labels.pkl',
         '1GkmOxIAvmjSwo22UzmZTSlw8NSmU5Q9H'),
        ('batches_filelists.zip',
         '1r0gbo5_Qlzrdet1GPIrJpVSGRgFU7NEp')
    ]

    def __init__(self, root=expanduser("~") + "/.avalanche/data/openloris/"):
        """
        Args:
            root (string): folder in which to download openloris dataset.
        """
        # we create the dir if it does not exists
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def download(self):
        """ Download from google drive official repositories. """

        for name in self.filename:
            try:
                filepath = os.path.join(self.root, name[0])
                url = "https://drive.google.com/u/0/uc?id=" + name[1]
                gdown.download(url, filepath, quiet=False)
                gdown.cached_download(url, filepath)
            except Exception as e:
                print('[OpenLoris] Direct download may no longer be supported!')
                raise e

            extract_archive(filepath, to_path=self.root)


if __name__ == "__main__":
    """ Simple object creation and download test """

    data = OPENLORIS_DATA()


__all__ = [
    'OPENLORIS_DATA'
]
