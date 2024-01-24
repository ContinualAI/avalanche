################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 10-10-2020                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" CORe50 Pytorch Dataset """

import glob
import os
import pickle as pkl
import dill
from pathlib import Path
from typing import List, Optional, Tuple, Union
from warnings import warn

from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets.core50 import core50_data
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.datasets.downloadable_dataset import (
    DownloadableDataset,
)
from avalanche.checkpointing import constructor_based_serialization


class CORe50Dataset(DownloadableDataset):
    """CORe50 Pytorch Dataset"""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        train=True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download=True,
        mini=False,
        object_level=True,
    ):
        """Creates an instance of the CORe50 dataset.

        :param root: root for the datasets data. Defaults to None, which
            means that the default location for 'core50' will be used.
        :param train: train or test split.
        :param transform: eventual transformations to be applied.
        :param target_transform: eventual transformation to be applied to the
            targets.
        :param loader: the procedure to load the instance from the storage.
        :param download: boolean to automatically download data. Default to
            True.
        :param mini: boolean to use the 32x32 version instead of the 128x128.
            Default to False.
        :param object_level: if the classification is objects based or
            category based: 50 or 10 way classification problem. Default to True
            (50-way object classification problem)
        """

        if root is None:
            root = default_dataset_location("core50")

        super(CORe50Dataset, self).__init__(root, download=download, verbose=True)

        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.object_level = object_level
        self.mini = mini

        # any scenario and run is good here since we want just to load the
        # train images and targets with no particular order
        self._scen = "ni"
        self._run = 0
        self._nbatch = 8

        # Download the dataset and initialize metadata
        self._load_dataset()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target
                class.
        """

        target = self.targets[index]
        if self.mini:
            bp = "core50_32x32"
        else:
            bp = "core50_128x128"

        img = self.loader(str(self.root / bp / self.paths[index]))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)

    def _download_dataset(self) -> None:
        data2download: List[Tuple[str, str, str]] = core50_data.data

        if self.mini:
            data2download = list(data2download)
            data2download[0] = core50_data.extra_data[1]

        for name in data2download:
            if self.verbose:
                print("Downloading " + name[1] + "...")
            file = self._download_file(name[1], name[0], name[2])
            if name[1].endswith(".zip"):
                if self.verbose:
                    print(f"Extracting {name[0]}...")
                extract_root = self._extract_archive(file)
                if self.verbose:
                    print("Extraction completed!")

    def _load_metadata(self) -> bool:
        if self.mini:
            bp = "core50_32x32"
        else:
            bp = "core50_128x128"

        if not (self.root / bp).exists():
            return False

        if not (self.root / "batches_filelists").exists():
            return False

        with open(self.root / "paths.pkl", "rb") as f:
            self.train_test_paths = pkl.load(f)

        if self.verbose:
            print("Loading labels...")
        with open(self.root / "labels.pkl", "rb") as f:
            self.all_targets = pkl.load(f)
            self.train_test_targets = []
            for i in range(self._nbatch + 1):
                self.train_test_targets += self.all_targets[self._scen][self._run][i]

        if self.verbose:
            print("Loading LUP...")
        with open(self.root / "LUP.pkl", "rb") as f:
            self.LUP = pkl.load(f)

        if self.verbose:
            print("Loading labels names...")
        with open(self.root / "labels2names.pkl", "rb") as f:
            self.labels2names = pkl.load(f)

        self.idx_list = []
        if self.train:
            for i in range(self._nbatch):
                self.idx_list += self.LUP[self._scen][self._run][i]
        else:
            self.idx_list = self.LUP[self._scen][self._run][-1]

        self.paths = []
        self.targets = []

        for idx in self.idx_list:
            self.paths.append(self.train_test_paths[idx])
            div = 1
            if not self.object_level:
                div = 5
            self.targets.append(self.train_test_targets[idx] // div)

        with open(self.root / "labels2names.pkl", "rb") as f:
            self.labels2names = pkl.load(f)

        if not (self.root / "NIC_v2_79_cat").exists():
            self._create_cat_filelists()

        return True

    def _download_error_message(self) -> str:
        all_urls = [name_url[1] for name_url in core50_data.data]

        base_msg = (
            "[CORe50] Error downloading the dataset!\n"
            "You should download data manually using the following links:\n"
        )

        for url in all_urls:
            base_msg += url
            base_msg += "\n"

        base_msg += "and place these files in " + str(self.root)

        return base_msg

    def _create_cat_filelists(self):
        """Generates corresponding filelists with category-wise labels. The
        default one are based on the object-level labels from 0 to 49."""

        for k, v in core50_data.scen2dirs.items():
            orig_root_path = os.path.join(self.root, v)
            root_path = os.path.join(self.root, v[:-1] + "_cat")
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            for run in range(10):
                cur_path = os.path.join(root_path, "run" + str(run))
                orig_cur_path = os.path.join(orig_root_path, "run" + str(run))
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

    def _objlab2cat(self, label, scen, run):
        """Mapping an object label into its corresponding category label
        based on the scenario."""

        if scen == "nc":
            return core50_data.name2cat[self.labels2names["nc"][run][label][:-1]]
        else:
            return int(label) // 5


def CORe50(*args, **kwargs):
    warn(
        "Dataset CORe50 has been renamed CORe50Dataset to prevent confusion "
        "with the CORe50 classic benchmark",
        DeprecationWarning,
        2,
    )
    return CORe50Dataset(*args, **kwargs)


@dill.register(CORe50Dataset)
def checkpoint_CORe50Dataset(pickler, obj: CORe50Dataset):
    constructor_based_serialization(
        pickler,
        obj,
        CORe50Dataset,
        deduplicate=True,
        kwargs=dict(
            root=obj.root,
            train=obj.train,
            transform=obj.transform,
            target_transform=obj.target_transform,
            loader=obj.loader,
            mini=obj.mini,
            object_level=obj.object_level,
        ),
    )


if __name__ == "__main__":
    # this litte example script can be used to visualize the first image
    # leaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    train_data = CORe50Dataset(transform=ToTensor())
    test_data = CORe50Dataset(train=False, transform=ToTensor())
    print("train size: ", len(train_data))
    print("Test size: ", len(test_data))
    print(train_data.labels2names)
    dataloader = DataLoader(train_data, batch_size=1)

    for batch_data in dataloader:
        x, y = batch_data
        plt.imshow(transforms.ToPILImage()(torch.squeeze(x)))
        plt.show()
        print(x.size())
        print(len(y))
        break


__all__ = ["CORe50Dataset", "CORe50"]
