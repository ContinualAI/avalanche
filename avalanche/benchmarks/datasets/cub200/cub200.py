################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-04-2021                                                             #
# Author: Lorenzo Pellegrini, Vincenzo Lomonaco                                #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

"""
CUB200 Pytorch Dataset: Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an
extended version of the CUB-200 dataset, with roughly double the number of
images per class and new part location annotations. For detailed information
about the dataset, please check the official website:
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gdown
import os
import dill
from collections import OrderedDict
from torchvision.datasets.folder import default_loader

from avalanche.benchmarks.datasets import (
    default_dataset_location,
    DownloadableDataset,
)
from avalanche.benchmarks.utils import PathsDataset
from avalanche.checkpointing import constructor_based_serialization


class CUB200(PathsDataset, DownloadableDataset):
    """Basic CUB200 PathsDataset to be used as a standard PyTorch Dataset.
    A classic continual learning benchmark built on top of this dataset
    can be found in 'benchmarks.classic', while for more custom benchmark
    design please use the 'benchmarks.generators'."""

    images_folder = "CUB_200_2011/images"
    official_url = (
        "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/" "CUB_200_2011.tgz"
    )
    gdrive_url = (
        "https://drive.google.com/u/0/uc?id=" "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
    )
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        train=True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download=True
    ):
        """

        :param root: root dir where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            'CUB_200_2011' will be used.
        :param train: train or test subset of the original dataset. Default
            to True.
        :param transform: eventual input data transformations to apply.
            Default to None.
        :param target_transform: eventual target data transformations to apply.
            Default to None.
        :param loader: method to load the data from disk. Default to
            torchvision default_loader.
        :param download: default set to True. If the data is already
            downloaded it will skip the download.
        """

        if root is None:
            root = default_dataset_location("CUB_200_2011")

        self.train: bool = train

        # Needed for disambiguating the type,
        # which is not the same in the base classes
        self.root: Path = Path(root)
        self._images: List[Tuple[str, int]]

        DownloadableDataset.__init__(self, root, download=download, verbose=True)
        self._load_dataset()

        PathsDataset.__init__(
            self,
            os.path.join(root, CUB200.images_folder),
            self._images,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
        )

    def _download_dataset(self) -> None:
        try:
            self._download_and_extract_archive(
                CUB200.official_url, CUB200.filename, checksum=CUB200.tgz_md5
            )
        except Exception:
            if self.verbose:
                print(
                    "[CUB200] Direct download may no longer be possible, "
                    "will try GDrive."
                )

        filepath = self.root / self.filename
        gdown.download(self.gdrive_url, str(filepath), quiet=False)
        gdown.cached_download(self.gdrive_url, str(filepath), md5=self.tgz_md5)

        self._extract_archive(filepath)

    def _download_error_message(self) -> str:
        return (
            "[CUB200] Error downloading the dataset. Consider downloading "
            "it manually at: " + CUB200.official_url + " and placing it "
            "in: " + str(self.root)
        )

    def _load_metadata(self):
        """Main method to load the CUB200 metadata"""

        cub_dir = self.root / "CUB_200_2011"
        images_list: Dict[int, List] = OrderedDict()

        with open(str(cub_dir / "train_test_split.txt")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=" ")
            for row in csv_reader:
                img_id = int(row[0])
                is_train_instance = int(row[1]) == 1
                if is_train_instance == self.train:
                    images_list[img_id] = []

        with open(str(cub_dir / "images.txt")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=" ")
            for row in csv_reader:
                img_id = int(row[0])
                if img_id in images_list:
                    images_list[img_id].append(row[1])

        with open(str(cub_dir / "image_class_labels.txt")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=" ")
            for row in csv_reader:
                img_id = int(row[0])
                if img_id in images_list:
                    # CUB starts counting classes from 1 ...
                    images_list[img_id].append(int(row[1]) - 1)

        with open(str(cub_dir / "bounding_boxes.txt")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=" ")
            for row in csv_reader:
                img_id = int(row[0])
                if img_id in images_list:
                    box_cub = [int(float(x)) for x in row[1:]]
                    box_avl = [box_cub[1], box_cub[0], box_cub[3], box_cub[2]]
                    # PathsDataset accepts (top, left, height, width)
                    images_list[img_id].append(box_avl)

        images_tuples = []
        for _, img_tuple in images_list.items():
            images_tuples.append(tuple(img_tuple))
        self._images = images_tuples  # type: ignore

        # Integrity check
        for row_check in self._images:
            filepath = self.root / CUB200.images_folder / row_check[0]
            if not filepath.is_file():
                if self.verbose:
                    print("[CUB200] Error checking integrity of:", filepath)
                return False

        return True


@dill.register(CUB200)
def checkpoint_CUB200(pickler, obj: CUB200):
    constructor_based_serialization(
        pickler,
        obj,
        CUB200,
        deduplicate=True,
        kwargs=dict(
            root=obj.root,
            train=obj.train,
            transform=obj.transform,
            target_transform=obj.target_transform,
            loader=obj.loader,
        ),
    )


if __name__ == "__main__":
    """Simple test that will start if you run this script directly"""

    import matplotlib.pyplot as plt

    dataset = CUB200(train=False, download=True)
    print("test data len:", len(dataset))
    img, _ = dataset[14]
    plt.imshow(img)
    plt.show()

    dataset = CUB200(train=True)
    print("train data len:", len(dataset))
    img, _ = dataset[700]
    plt.imshow(img)
    plt.show()


__all__ = ["CUB200"]
