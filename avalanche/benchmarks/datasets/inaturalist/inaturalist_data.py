################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-05-2021                                                             #
# Author: Matthias De Lange                                                    #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" INATURALIST2018 Data handling utilities
For more info see: https://github.com/visipedia/inat_comp/tree/master/2018
There are a total of 8,142 species in the dataset, with 437,513 training and
24,426 validation images. We only use the imbalanced training data.
Images are 800x600.
Un-tarring the images creates a directory structure like
train_val2018/<supercategory>/<category>/<image>.jpg. This may take a while.


JSON is in COCO format like:
    {
      "info" : info,
      "images" : [image],
      "categories" : [category],
      "annotations" : [annotation],
      "licenses" : [license]
    }

    info{
      "year" : int,
      "version" : str,
      "description" : str,
      "contributor" : str,
      "url" : str,
      "date_created" : datetime,
    }

    image{
      "id" : int,
      "width" : int,
      "height" : int,
      "file_name" : str,
      "license" : int,
      "rights_holder" : str
    }

    category{
      "id" : int,
      "name" : str,
      "supercategory" : str,
      "kingdom" : str,
      "phylum" : str,
      "class" : str,
      "order" : str,
      "family" : str,
      "genus" : str
    }

    annotation{
      "id" : int,
      "image_id" : int,
      "category_id" : int
    }

    license{
      "id" : int,
      "name" : str,
      "url" : str
    }
"""

import os
import sys
import logging
import tarfile

from tqdm import tqdm

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve

base_url = "https://ml-inat-competition-datasets.s3.amazonaws.com/2018"
train_data = [
    # 120G: Train+val data
    ("train_val2018.tar.gz", f"{base_url}/train_val2018.tar.gz"),
    # Training annotations
    ("train2018.json.tar.gz", f"{base_url}/train2018.json.tar.gz"),
    # Validation annotations
    ("val2018.json.tar.gz", f"{base_url}/val2018.json.tar.gz"),
]

test_data = [
    # 40G: Test data
    ("test2018.tar.gz", f"{base_url}/test2018.tar.gz"),
    # Test annotations
    ("test2018.json.tar.gz", f"{base_url}/test2018.json.tar.gz"),
]


class TqdmUpTo(tqdm):
    """
    Progress bar for urlretrieve-based downloads.
    https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


class INATURALIST_DATA(object):
    """
    INATURALIST downloader.
    """

    def __init__(self, data_folder="data/", trainval=True):
        """
        Args:
            data_folder (string): folder in which to download
            inaturalist dataset.
        """
        # Get train data (incl val data) or test data
        self.trainval = trainval
        self.log = logging.getLogger("avalanche")

        if os.path.isabs(data_folder):
            self.data_folder = data_folder
        else:
            self.data_folder = os.path.join(os.path.dirname(__file__), data_folder)

        try:
            # Create target Directory for INATURALIST data
            os.makedirs(self.data_folder, exist_ok=True)
            self.log.info("Directory %s created", self.data_folder)
            self.download = True
            self.download_inaturalist()

        except OSError:
            import traceback

            traceback.print_exc()
            self.download = False
            self.log.error("Directory %s already exists", self.data_folder)

    def download_inaturalist(self):
        """Download and extract inaturalist data

        :param extra: download also additional INATURALIST data not strictly
            required by the data loader.
        """

        data2download = train_data if self.trainval else test_data

        for name in data2download:
            self.log.info("Downloading " + name[1] + "...")
            save_name = os.path.join(self.data_folder, name[0])
            if not os.path.exists(save_name):
                with TqdmUpTo(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=name[0],
                ) as t:
                    urlretrieve(name[1], save_name, reporthook=t.update_to)
            else:
                self.log.info("Skipping download, exists: ", save_name)

            if name[0].endswith("tar.gz"):
                untar_save_name = os.path.join(
                    self.data_folder, ".".join(name[0].split(".")[:-2])
                )
                if not os.path.exists(untar_save_name):
                    with tarfile.open(
                        os.path.join(self.data_folder, name[0]), "r:gz"
                    ) as tar:
                        self.log.info("Extracting INATURALIST images...")
                        tar.extractall(self.data_folder)
                        self.log.info("Done!")
                else:
                    self.log.info("Skipping untarring, exists: ", save_name)
        self.log.info("Download complete.")


__all__ = ["INATURALIST_DATA"]
