################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-02-2021                                                             #
# Author: Tyler L. Hayes                                                       #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" Stream-51 Pytorch Dataset """

import os

import shutil
import json
import random
import dill
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, TypeVar, Union

from torchvision.datasets.folder import default_loader
from zipfile import ZipFile

from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import (
    DownloadableDataset,
    default_dataset_location,
)
from avalanche.benchmarks.datasets.stream51 import stream51_data
from avalanche.checkpointing import constructor_based_serialization


TSequence = TypeVar("TSequence", bound=Sequence)


class Stream51(DownloadableDataset):
    """Stream-51 Pytorch Dataset"""

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
        Creates an instance of the Stream-51 dataset.

        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            'stream51' will be used.
        :param train: If True, the training set will be returned. If False,
            the test set will be returned.
        :param transform: The transformations to apply to the X values.
        :param target_transform: The transformations to apply to the Y values.
        :param loader: The image loader to use.
        :param download: If True, the dataset will be downloaded if needed.
        """

        if root is None:
            root = default_dataset_location("stream51")

        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.bbox_crop = True
        self.ratio = 1.1
        self.samples: Sequence[Tuple[int, Any, str]] = []

        super(Stream51, self).__init__(root, download=download, verbose=True)

        self._load_dataset()

    def _download_dataset(self) -> None:
        self._download_file(
            stream51_data.name[1], stream51_data.name[0], stream51_data.name[2]
        )

        if self.verbose:
            print("[Stream-51] Extracting dataset...")

        if stream51_data.name[1].endswith(".zip"):
            lfilename = self.root / stream51_data.name[0]
            with ZipFile(str(lfilename), "r") as zipf:
                for member in zipf.namelist():
                    filename = os.path.basename(member)
                    # skip directories
                    if not filename:
                        continue

                    # copy file (taken from zipfile's extract)
                    source = zipf.open(member)
                    if "json" in filename:
                        target = open(str(self.root / filename), "wb")
                    else:
                        dest_folder = os.path.join(*(member.split(os.path.sep)[1:-1]))
                        dest_folder_path = self.root / dest_folder
                        dest_folder_path.mkdir(exist_ok=True, parents=True)

                        target = open(str(dest_folder_path / filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)

            # lfilename.unlink()

    def _load_metadata(self) -> bool:
        if self.train:
            data_list = json.load(open(str(self.root / "Stream-51_meta_train.json")))
        else:
            data_list = json.load(open(str(self.root / "Stream-51_meta_test.json")))

        self.samples = data_list
        self.targets = [s[0] for s in data_list]

        self.bbox_crop = True
        self.ratio = 1.1

        return True

    def _download_error_message(self) -> str:
        return (
            "[Stream-51] Error downloading the dataset. Consider "
            "downloading it manually at: "
            + stream51_data.name[1]
            + " and placing it in: "
            + str(self.root)
        )

    @staticmethod
    def _instance_ordering(data_list: Sequence[TSequence], seed) -> List[TSequence]:
        # organize data by video
        total_videos = 0
        new_data_list = []
        temp_video: List[TSequence] = []
        for x in data_list:
            if x[3] == 0:
                new_data_list.append(temp_video)
                total_videos += 1
                temp_video = [x]
            else:
                temp_video.append(x)
        new_data_list.append(temp_video)
        new_data_list = new_data_list[1:]
        # shuffle videos
        random.seed(seed)
        random.shuffle(new_data_list)
        # reorganize by clip
        data_list_result = []
        for v in new_data_list:
            for x in v:
                data_list_result.append(x)
        return data_list_result

    @staticmethod
    def _class_ordering(data_list, class_type, seed):
        # organize data by class
        new_data_list = []
        for class_id in range(data_list[-1][0] + 1):
            class_data_list = [x for x in data_list if x[0] == class_id]
            if class_type == "class_iid":
                # shuffle all class data
                random.seed(seed)
                random.shuffle(class_data_list)
            else:
                # shuffle clips within class
                class_data_list = Stream51._instance_ordering(class_data_list, seed)
            new_data_list.append(class_data_list)
        # shuffle classes
        random.seed(seed)
        random.shuffle(new_data_list)
        # reorganize by class
        data_list = []
        for v in new_data_list:
            for x in v:
                data_list.append(x)
        return data_list

    @staticmethod
    def make_dataset(data_list, ordering="class_instance", seed=666):
        """
        data_list
        for train: [class_id, clip_num, video_num, frame_num, bbox, file_loc]
        for test: [class_id, bbox, file_loc]
        """
        if not ordering or len(data_list[0]) == 3:  # cannot order the test set
            return data_list
        if ordering not in ["iid", "class_iid", "instance", "class_instance"]:
            raise ValueError(
                'dataset ordering must be one of: "iid", "class_iid", '
                '"instance", or "class_instance"'
            )
        if ordering == "iid":
            # shuffle all data
            random.seed(seed)
            random.shuffle(data_list)
            return data_list
        elif ordering == "instance":
            return Stream51._instance_ordering(data_list, seed)
        elif "class" in ordering:
            return Stream51._class_ordering(data_list, ordering, seed)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target
            class.
        """
        fpath, target = self.samples[index][-1], self.targets[index]
        sample = self.loader(str(self.root / fpath))
        if self.bbox_crop:
            bbox = self.samples[index][-2]
            cw = bbox[0] - bbox[1]
            ch = bbox[2] - bbox[3]
            center = [int(bbox[1] + cw / 2), int(bbox[3] + ch / 2)]
            bbox = [
                min([int(center[0] + (cw * self.ratio / 2)), sample.size[0]]),
                max([int(center[0] - (cw * self.ratio / 2)), 0]),
                min([int(center[1] + (ch * self.ratio / 2)), sample.size[1]]),
                max([int(center[1] - (ch * self.ratio / 2)), 0]),
            ]
            sample = sample.crop((bbox[1], bbox[3], bbox[0], bbox[2]))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )

        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp,
            self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp)),
        )
        return fmt_str


@dill.register(Stream51)
def checkpoint_Stream51(pickler, obj: Stream51):
    constructor_based_serialization(
        pickler,
        obj,
        Stream51,
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

    train_data = Stream51(transform=ToTensor())
    test_data = Stream51(transform=ToTensor(), train=False)
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

__all__ = ["Stream51"]
