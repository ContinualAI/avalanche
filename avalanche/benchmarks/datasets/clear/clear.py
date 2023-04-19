################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 05-17-2022                                                             #
# Author: Zhiqiu Lin, Jia Shi                                                  #
# E-mail: zl279@cornell.edu, jiashi@andrew.cmu.edu                             #
# Website: https://clear-benchmark.github.io                                   #
################################################################################

""" CLEAR Pytorch Dataset """

from pathlib import Path
from typing import Union, List
import json
import os

import torch
from torchvision.datasets.folder import default_loader

from avalanche.benchmarks.datasets import (
    DownloadableDataset,
    default_dataset_location,
)
from avalanche.benchmarks.utils import default_flist_reader
from avalanche.benchmarks.datasets.clear import clear_data

_CLEAR_DATA_SPLITS = {"clear10", "clear100"}

CLEAR_FEATURE_TYPES = {
    "clear10": ["moco_b0"],
    "clear100": ["moco_b0"],
}

SPLIT_OPTIONS = ["all", "train", "test"]

SEED_LIST = [0, 1, 2, 3, 4]  # Available seeds for train:test split


def _load_json(json_location):
    with open(json_location, "r") as f:
        obj = json.load(f)
    return obj


class CLEARDataset(DownloadableDataset):
    """CLEAR Base Dataset for downloading / loading metadata"""

    def __init__(
        self,
        root: Union[str, Path] = None,
        *,
        data_name: str = "clear10",
        download: bool = True,
        verbose: bool = False,
    ):
        """
        Creates an instance of the CLEAR dataset.
        This base class simply download and unzip the CLEAR dataset.

        This serves as a base class for _CLEARImage/_CLEARFeature dataset.

        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            str(data_name) will be used.
        :param data_name: Data module name with the google drive url and md5
        :param download: If True, the dataset will be downloaded if needed.
        """

        if root is None:
            root = default_dataset_location(data_name)

        assert data_name in _CLEAR_DATA_SPLITS
        self.data_name = data_name
        self.module = clear_data

        super(CLEARDataset, self).__init__(
            root, download=download, verbose=True
        )
        self._load_dataset()

    def _download_dataset(self) -> None:
        target_module = getattr(self.module, self.data_name)

        for name, base_url in target_module:
            if self.verbose:
                print("Downloading " + name + "...")
            url = os.path.join(base_url, name)
            filepath = self.root / name
            os.system(f"wget -P {str(self.root)} {url}")
            self._extract_archive(filepath, remove_archive=True)

    def _load_metadata(self) -> bool:
        splits = ["train", "test"] if self.split == "all" else [self.split]
        for split in splits:
            train_folder_path = self.root / split
            if not train_folder_path.exists():
                print(f"{train_folder_path} does not exist. ")
                return False
        
            self.labeled_metadata = _load_json(
                train_folder_path / "labeled_metadata.json"
            )

            class_names_file = train_folder_path / "class_names.txt"
            self.class_names = class_names_file.read_text().split("\n")

            self.samples = []
            self._paths_and_targets = []
            for bucket, data in self.labeled_metadata.items():
                for class_idx, class_name in enumerate(self.class_names):
                    metadata_path = data[class_name]
                    metadata_path = train_folder_path / metadata_path
                    if not metadata_path.exists():
                        print(f"{metadata_path} does not exist. ")
                        return False
                    metadata = _load_json(metadata_path)
                    for v in metadata.values():
                        f_path = os.path.join(split, v["IMG_PATH"])
                        self.samples.append((f_path, class_idx))

            # Check whether all labeled images exist
            for img_path, _ in self.samples:
                path = self.root / img_path
                if not os.path.exists(path):
                    print(f"{path} does not exist. Files not properly extracted?")
                    return False
            return True

    def _download_error_message(self) -> str:
        all_urls = [
            os.path.join(item[1], item[0])
            for item in getattr(self.module, self.data_name)
        ]

        base_msg = (
            f"[{self.data_name}] Direct download may no longer be supported!\n"
            "You should download data manually using the following links:\n"
        )

        for url in all_urls:
            base_msg += url
            base_msg += "\n"

        base_msg += "and place these files in " + str(self.root)

        return base_msg

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        return str(self.root / img_path), target

    def __len__(self):
        return len(self.samples)


class _CLEARImage(CLEARDataset):
    """CLEAR Image Dataset (base class for CLEARImage)"""

    def __init__(
        self,
        root: Union[str, Path] = None,
        *,
        data_name: str = "clear10",
        download: bool = True,
        verbose: bool = True,
        split: str = "all",
        seed: int = None,
        transform=None,
        target_transform=None,
        loader=default_loader,
    ):
        """
        Creates an instance of the CLEAR dataset.
        This image dataset will contain samples from all buckets of CLEAR,
        so it is not intended for CL purposes. It simply download and
        unzip the CLEAR dataset.

        Paths and targets for each bucket for benchmark creation will be
        loaded in self._paths_and_targets ;
        can use self.get_paths_and_targets() with root appended to each path


        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            str(data_name) will be used.
        :param data_name: Data module name with the google drive url and md5
        :param download: If True, the dataset will be downloaded if needed.
        :param split: Choose from ['all', 'train', 'test'].
            If 'all', then return all data from all buckets.
            If 'train'/'test', then only return train/test data.
        :param seed: The random seed used for splitting the train:test into 7:3
            If split=='all', then seed must be None (since no split is done)
            otherwise, choose from [0, 1, 2, 3, 4]
        :param transform: The transformations to apply to the X values.
        :param target_transform: The transformations to apply to the Y values.
        :param loader: The image loader to use.
        """
        self.split = split
        assert self.split in SPLIT_OPTIONS, "Invalid split option"
        if self.split == "all":
            assert seed is None, "Specify a seed if not splitting train:test"
        else:
            assert seed in SEED_LIST
        self.seed = seed
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.class_names: List[str] = None
        """
        After _load_metadata(), the class names will be loaded in order
        aligned with target index.
        """

        super(_CLEARImage, self).__init__(
            root, data_name=data_name, download=download, verbose=True
        )

    def _load_metadata(self) -> bool:
        if not super(_CLEARImage, self)._load_metadata():
            print("CLEAR has not yet been downloaded")
            return False
        
        self.paths = []
        self.targets = []
        self._paths_and_targets = []
        splits = ["test", "train"] if self.split == "all" else [self.split]
        for split in splits:
            train_folder_path = self.root / split
            if not train_folder_path.exists():
                print(f"{train_folder_path} does not exist. ")
                return False
        
            self.labeled_metadata = _load_json(
                train_folder_path / "labeled_metadata.json"
            )

            samples = []
            for bucket, data in self.labeled_metadata.items():
                for class_idx, class_name in enumerate(self.class_names):
                    metadata_path = data[class_name]
                    metadata_path = train_folder_path / metadata_path
                    if not metadata_path.exists():
                        print(f"{metadata_path} does not exist. ")
                        return False
                    metadata = _load_json(metadata_path)
                    for v in metadata.values():
                        f_path = os.path.join(split, v["IMG_PATH"])
                        samples.append((f_path, class_idx))
                if self.split == "all" and split == "train":
                    _samples = self._paths_and_targets[int(bucket)]
                    _samples += samples
                    self._paths_and_targets[int(bucket)] = _samples
                else:
                    self._paths_and_targets.append(samples)
        for path_and_target_list in self._paths_and_targets:
            for img_path, target in path_and_target_list:
                self.paths.append(self.root / img_path)
                self.targets.append(target)
        return True

    def get_paths_and_targets(self, root_appended=True):
        """Return self._paths_and_targets with root appended or not"""
        if not root_appended:
            return self._paths_and_targets
        else:
            paths_and_targets = []
            for path_and_target_list in self._paths_and_targets:
                paths_and_targets.append([])
                for img_path, target in path_and_target_list:
                    paths_and_targets[-1].append((self.root / img_path, target))
            return paths_and_targets

    def __getitem__(self, index):
        img_path = self.paths[index]
        target = self.targets[index]

        img = self.loader(str(self.root / img_path))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


class _CLEARFeature(CLEARDataset):
    """CLEAR Feature Dataset (base class for CLEARFeature)"""

    def __init__(
        self,
        root: Union[str, Path] = None,
        *,
        data_name: str = "clear10",
        download: bool = True,
        verbose: bool = True,
        split: str = "all",
        seed: int = None,
        feature_type: str = "moco_b0",
        target_transform=None,
    ):
        """
        Creates an instance of the CLEAR dataset.
        This image dataset will contain samples from all buckets of CLEAR,
        so it is not intended for CL purposes. It simply download and
        unzip the CLEAR dataset.

        Tensors and targets for benchmark creation will be
        loaded in self.tensors_and_targets

        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            str(data_name) will be used.
        :param data_name: Data module name with the google drive url and md5
        :param download: If True, the dataset will be downloaded if needed.
        :param split: Choose from ['all', 'train', 'test'].
            If 'all', then return all data from all buckets.
            If 'train'/'test', then only return train/test data.
        :param seed: The random seed used for splitting the train:test into 7:3
            If split=='all', then seed must be None (since no split is done)
            otherwise, choose from [0, 1, 2, 3, 4]
        :param feature_type: The type of features.
            For CLEAR10, choose from [
                'moco_b0', # Moco V2 ResNet50 pretrained on bucket 0
            ]
        :param target_transform: The transformations to apply to the Y values.
        """
        self.split = split
        assert self.split in ["all", "train", "test"], "Invalid split option"

        if self.split == "all":
            assert seed is None, "Specify a seed if not splitting train:test"
        else:
            assert seed in SEED_LIST
        self.seed = seed

        self.feature_type = feature_type
        assert feature_type in CLEAR_FEATURE_TYPES[data_name]
        self.target_transform = target_transform

        super(_CLEARFeature, self).__init__(
            root, data_name=data_name, download=download, verbose=True
        )

    def _load_metadata(self) -> bool:
        if not super(_CLEARFeature, self)._load_metadata():
            print("CLEAR has not yet been downloaded")
            return False
        
        
        self.tensors_and_targets = []
        splits = ["test", "train"] if self.split == "all" else [self.split]
        for split in splits:
            folder_path = self.root / self.split
            feature_folder_path = folder_path / "features" / self.feature_type
            metadata = _load_json(feature_folder_path / "features.json")
            tensors = []
            targets = []
            for bucket, data in metadata.items():
                for class_idx, class_name in enumerate(self.class_names):
                    feature_path = data[class_name]
                    try:
                        features = torch.load(folder_path / feature_path)
                    except Exception as e:
                        print(f"Error loading {feature_path}")
                        return False
                    for _id, tensor in features.items():
                        tensors.append(tensor)
                        targets.append(class_idx)
                if self.split == "all" and split == "train":
                    _tensors, _targets = self.tensors_and_targets[int(bucket)]
                    _tensors += tensors
                    _targets += targets
                    self.tensors_and_targets[int(bucket)] = (_tensors, _targets)
                else:
                    self.tensors_and_targets.append((tensors, targets))

        self.tensors = []
        self.targets = []
        for tensors, targets in self.tensors_and_targets:
            for tensor, target in zip(tensors, targets):
                self.tensors.append(tensor)
                self.targets.append(target)

        return True

    def __getitem__(self, index):
        tensor = self.tensors[index]
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return tensor, target

    def __len__(self):
        return len(self.targets)


if __name__ == "__main__":

    # this little example script can be used to visualize the first image
    # loaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    data_name = "clear10"
    root = f"../avalanche_datasets/{data_name}"
    if not os.path.exists(root):
        Path(root).mkdir(parents=True)
    clear_dataset_all = _CLEARImage(
        root=root,
        data_name=data_name,
        download=True,
        split="all",
        seed=None,
        transform=transform,
    )
    clear_dataset_train = _CLEARImage(
        root=root,
        data_name=data_name,
        download=True,
        split="train",
        seed=0,
        transform=transform,
    )
    clear_dataset_test = _CLEARImage(
        root=root,
        data_name=data_name,
        download=True,
        split="test",
        seed=0,
        transform=transform,
    )
    print("clear10 size (all): ", len(clear_dataset_all))
    print("clear10 size (train): ", len(clear_dataset_train))
    print("clear10 size (test): ", len(clear_dataset_test))

    clear_dataset_all_feature = _CLEARFeature(
        root=root,
        data_name=data_name,
        download=True,
        feature_type="moco_b0",
        split="all",
        seed=None,
    )
    clear_dataset_train_feature = _CLEARFeature(
        root=root,
        data_name=data_name,
        download=True,
        feature_type="moco_b0",
        split="train",
        seed=0,
    )
    clear_dataset_test_feature = _CLEARFeature(
        root=root,
        data_name=data_name,
        download=True,
        feature_type="moco_b0",
        split="test",
        seed=0,
    )
    print("clear10 size (all features): ", len(clear_dataset_all_feature))
    print("clear10 size (train features): ", len(clear_dataset_train_feature))
    print("clear10 size (test features): ", len(clear_dataset_test_feature))
    print("Classes are: ")
    for i, name in enumerate(clear_dataset_test.class_names):
        print(f"{i} : {name}")
    dataloader = DataLoader(clear_dataset_test_feature, batch_size=1)

    for batch_data in dataloader:
        x, y = batch_data
        print(x.size())
        print(len(y))
        break

__all__ = [
    "CLEARDataset",
    "_CLEARFeature",
    "_CLEARImage",
    "SEED_LIST",
    "CLEAR_FEATURE_TYPES",
    "_CLEAR_DATA_SPLITS",
]
