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

""" CORe50 Data handling utilities """

import os
import sys
import logging
import glob
import pickle as pkl
from zipfile import ZipFile


if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve


data = [
    ('core50_128x128.zip',
     'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'),
    ('batches_filelists.zip',
     'https://vlomonaco.github.io/core50/data/batches_filelists.zip'),
    ('batches_filelists_NICv2.zip',  
     'https://vlomonaco.github.io/core50/data/batches_filelists_NICv2.zip'),
    ('paths.pkl', 'https://vlomonaco.github.io/core50/data/paths.pkl'),
    ('LUP.pkl', 'https://vlomonaco.github.io/core50/data/LUP.pkl'),
    ('labels.pkl', 'https://vlomonaco.github.io/core50/data/labels.pkl'),
    ('labels2names.pkl',
     'https://vlomonaco.github.io/core50/data/labels2names.pkl')
]

extra_data = [
    ('core50_imgs.npz',
     'http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz')
]

scen2dirs = {
    'ni': "batches_filelists/NI_inc/",
    'nc': "batches_filelists/NC_inc/",
    'nic': "batches_filelists/NIC_inc/",
    'nicv2_79': "NIC_v2_79/",
    'nicv2_196': "NIC_v2_196/",
    'nicv2_391': "NIC_v2_391/"
}

name2cat = {
    'plug_adapter': 0,
    'mobile_phone': 1,
    'scissor': 2,
    'light_bulb': 3,
    'can': 4,
    'glass': 5,
    'ball': 6,
    'marker': 7,
    'cup': 8,
    'remote_control': 9
}

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
            self.log.info("Directory %s created", self.data_folder)
            self.download = True
            self.download_core50()

        except OSError:
            self.download = False
            self.log.error("Directory %s already exists", self.data_folder)

        with open(os.path.join(data_folder, 'labels2names.pkl'), 'rb') as f:
            self.labels2names = pkl.load(f)

        if os.path.exists(os.path.join(data_folder, "NIC_v2_79_cat")):
            # It means category filelists has been already created
            pass
        else:
            self._create_cat_filelists()

    def download_core50(self, extra=False):
        """ Download and extract CORe50 data

            :param extra: download also additional CORe50 data not strictly
                required by the data loader.
        """

        if extra:
            data2download = data + extra_data
        else:
            data2download = data

        for name in data2download:
            self.log.info("Downloading " + name[1] + "...")
            urlretrieve(name[1], os.path.join(self.data_folder, name[0]))

            if name[1].endswith('.zip'):
                with ZipFile(
                        os.path.join(self.data_folder, name[0]), 'r') as zipf:
                    self.log.info('Extracting CORe50 images...')
                    zipf.extractall(self.data_folder) 
                    self.log.info('Done!')

        self.log.info("Download complete.")


    def _objlab2cat(self, label, scen, run):
        """ Mapping an object label into its corresponding category label
        based on the scenario. """

        if scen == "nc":
            return name2cat[self.labels2names['nc'][run][label][:-1]]
        else:
            return int(label) // 5

    def _create_cat_filelists(self):
        """ Generates corresponding filelists with category-wise labels. The
        default one are based on the object-level labels from 0 to 49."""

        for k, v  in scen2dirs.items():
            orig_root_path = os.path.join(self.data_folder, v)
            root_path = os.path.join(self.data_folder, v[:-1] + "_cat")
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            for run in range(10):
                cur_path = os.path.join(root_path, "run"+str(run))
                orig_cur_path = os.path.join(orig_root_path, "run"+str(run))
                if not os.path.exists(cur_path):
                    os.makedirs(cur_path)
                for file in glob.glob(os.path.join(orig_cur_path, "*.txt")):
                    o_filename = file
                    _, d_filename = os.path.split(o_filename)
                    orig_f = open(o_filename, "r")
                    dst_f = open(os.path.join(cur_path, d_filename), "w")
                    for line in orig_f:
                        path, label = line.split(" ")
                        new_label = self._objlab2cat(int(label), k, run)
                        dst_f.write(path + " " + str(new_label) + "\n")
                    orig_f.close()
                    dst_f.close()


__all__ = [
    'CORE50_DATA'
]
