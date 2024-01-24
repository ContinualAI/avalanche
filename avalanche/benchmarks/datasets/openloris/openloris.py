################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-11-2020                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" OpenLoris Pytorch Dataset """

import pickle as pkl
from pathlib import Path
from typing import Optional, Union
import dill

from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import (
    DownloadableDataset,
    default_dataset_location,
)
from avalanche.benchmarks.datasets.openloris import openloris_data
from avalanche.checkpointing import constructor_based_serialization


class OpenLORIS(DownloadableDataset):
    """OpenLORIS Pytorch Dataset"""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        train=True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download=True,
    ):
        """
        Creates an instance of the OpenLORIS dataset.

        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            'openloris' will be used.
        :param train: If True, the training set will be returned. If False,
            the test set will be returned.
        :param transform: The transformations to apply to the X values.
        :param target_transform: The transformations to apply to the Y values.
        :param loader: The image loader to use.
        :param download: If True, the dataset will be downloaded if needed.
        """

        if root is None:
            root = default_dataset_location("openloris")

        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        super(OpenLORIS, self).__init__(root, download=download, verbose=True)
        self._load_dataset()

    def _download_dataset(self) -> None:
        data2download = openloris_data.avl_vps_data

        for name in data2download:
            if self.verbose:
                print("Downloading " + name[1] + "...")
            file = self._download_file(name[1], name[0], name[2])
            if name[1].endswith(".zip"):
                if self.verbose:
                    print(f"Extracting {name[0]}...")
                self._extract_archive(file)
                if self.verbose:
                    print("Extraction completed!")

    def _load_metadata(self) -> bool:
        if not self._check_integrity():
            return False

        # any scenario and factor is good here since we want just to load the
        # train images and targets with no particular order
        scen = "domain"
        factor = [_ for _ in range(4)]
        ntask = 9

        print("Loading paths...")
        with open(str(self.root / "Paths.pkl"), "rb") as f:
            self.train_test_paths = pkl.load(f)

        print("Loading labels...")
        with open(str(self.root / "Labels.pkl"), "rb") as f:
            self.all_targets = pkl.load(f)
            self.train_test_targets = []
            for fact in factor:
                for i in range(ntask + 1):
                    self.train_test_targets += self.all_targets[scen][fact][i]

        print("Loading LUP...")
        with open(str(self.root / "LUP.pkl"), "rb") as f:
            self.LUP = pkl.load(f)

        self.idx_list = []
        if self.train:
            for fact in factor:
                for i in range(ntask):
                    self.idx_list += self.LUP[scen][fact][i]
        else:
            for fact in factor:
                self.idx_list += self.LUP[scen][fact][-1]

        self.paths = []
        self.targets = []

        for idx in self.idx_list:
            self.paths.append(self.train_test_paths[idx])
            self.targets.append(self.train_test_targets[idx])

        return True

    def _download_error_message(self) -> str:
        base_url = openloris_data.base_gdrive_url
        all_urls = [base_url + name_url[1] for name_url in openloris_data.avl_vps_data]

        base_msg = (
            "[OpenLoris] Direct download may no longer be supported!\n"
            "You should download data manually using the following links:\n"
        )

        for url in all_urls:
            base_msg += url
            base_msg += "\n"

        base_msg += "and place these files in " + str(self.root)

        return base_msg

    def _check_integrity(self):
        """Checks if the data is already available and intact"""

        for name, url, md5 in openloris_data.avl_vps_data:
            filepath = self.root / name
            if not filepath.is_file():
                if self.verbose:
                    print(
                        "[OpenLORIS] Error checking integrity of:",
                        str(filepath),
                    )
                return False
        return True

    def __getitem__(self, index):
        target = self.targets[index]
        img = self.loader(str(self.root / self.paths[index]))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


@dill.register(OpenLORIS)
def checkpoint_OpenLORIS(pickler, obj: OpenLORIS):
    constructor_based_serialization(
        pickler,
        obj,
        OpenLORIS,
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
    # this little example script can be used to visualize the first image
    # loaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    train_data = OpenLORIS(download=True, transform=ToTensor())
    test_data = OpenLORIS(train=False, transform=ToTensor())
    print("train size: ", len(train_data))
    print("Test size: ", len(test_data))
    dataloader = DataLoader(train_data, batch_size=1)

    for batch_data in dataloader:
        x, y = batch_data
        plt.imshow(transforms.ToPILImage()(torch.squeeze(x)))
        plt.show()
        print(x.size())
        print(len(y))
        break

__all__ = ["OpenLORIS"]
