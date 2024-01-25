################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-05-2020                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" Tiny-Imagenet Pytorch Dataset """

import csv
from pathlib import Path
import dill
from typing import List, Optional, Tuple, Union

from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import (
    SimpleDownloadableDataset,
    default_dataset_location,
)
from avalanche.checkpointing import constructor_based_serialization


class TinyImagenet(SimpleDownloadableDataset):
    """Tiny Imagenet Pytorch Dataset"""

    filename = (
        "tiny-imagenet-200.zip",
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    )
    md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        train: bool = True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download=True
    ):
        """
        Creates an instance of the Tiny Imagenet dataset.

        :param root: folder in which to download dataset. Defaults to None,
            which means that the default location for 'tinyimagenet' will be
            used.
        :param train: True for training set, False for test set.
        :param transform: Pytorch transformation function for x.
        :param target_transform: Pytorch transformation function for y.
        :param loader: the procedure to load the instance from the storage.
        :param bool download: If True, the dataset will be  downloaded if
            needed.
        """

        if root is None:
            root = default_dataset_location("tinyimagenet")

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.loader = loader

        super(TinyImagenet, self).__init__(
            root, self.filename[1], self.md5, download=download, verbose=True
        )

        self._load_dataset()

    def _load_metadata(self) -> bool:
        self.data_folder = self.root / "tiny-imagenet-200"

        self.label2id, self.id2label = TinyImagenet.labels2dict(self.data_folder)
        self.data, self.targets = self.load_data()
        return True

    @staticmethod
    def labels2dict(data_folder: Path):
        """
        Returns dictionaries to convert class names into progressive ids
        and viceversa.

        :param data_folder: The root path of tiny imagenet
        :returns: label2id, id2label: two Python dictionaries.
        """

        label2id = {}
        id2label = {}

        with open(str(data_folder / "wnids.txt"), "r") as f:
            reader = csv.reader(f)
            curr_idx = 0
            for ll in reader:
                if ll[0] not in label2id:
                    label2id[ll[0]] = curr_idx
                    id2label[curr_idx] = ll[0]
                    curr_idx += 1

        return label2id, id2label

    def load_data(self):
        """
        Load all images paths and targets.

        :return: train_set, test_set: (train_X_paths, train_y).
        """

        data: Tuple[List[Path], List[int]] = ([], [])

        classes = list(range(200))
        for class_id in classes:
            class_name = self.id2label[class_id]

            if self.train:
                X = self.get_train_images_paths(class_name)
                Y = [class_id] * len(X)
            else:
                # test set
                X = self.get_test_images_paths(class_name)
                Y = [class_id] * len(X)

            data[0].extend(X)
            data[1].extend(Y)

        return data

    def get_train_images_paths(self, class_name) -> List[Path]:
        """
        Gets the training set image paths.

        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """
        train_img_folder: Path = self.data_folder / "train" / class_name / "images"

        img_paths = [f for f in train_img_folder.iterdir() if f.is_file()]

        return img_paths

    def get_test_images_paths(self, class_name) -> List[Path]:
        """
        Gets the test set image paths

        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """

        val_img_folder: Path = self.data_folder / "val" / "images"
        annotations_file: Path = self.data_folder / "val" / "val_annotations.txt"

        valid_names = []

        # filter validation images by class using appropriate file
        with open(str(annotations_file), "r") as f:
            reader = csv.reader(f, dialect="excel-tab")
            for ll in reader:
                if ll[1] == class_name:
                    valid_names.append(ll[0])

        img_paths = [val_img_folder / f for f in valid_names]

        return img_paths

    def __len__(self):
        """Returns the length of the set"""
        return len(self.data)

    def __getitem__(self, index):
        """Returns the index-th x, y pattern of the set"""

        path, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


@dill.register(TinyImagenet)
def checkpoint_TinyImagenet(pickler, obj: TinyImagenet):
    constructor_based_serialization(
        pickler,
        obj,
        TinyImagenet,
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

    train_data = TinyImagenet(transform=ToTensor())
    dataloader = DataLoader(train_data, batch_size=1)

    for batch_data in dataloader:
        x, y = batch_data
        plt.imshow(transforms.ToPILImage()(torch.squeeze(x)))
        plt.show()
        print(x.shape)
        print(y.shape)
        break


__all__ = ["TinyImagenet"]
