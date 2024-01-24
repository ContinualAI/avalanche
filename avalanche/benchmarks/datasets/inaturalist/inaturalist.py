################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-05-2021                                                             #
# Author: Matthias De Lange                                                    #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

"""INATURALIST2018 Pytorch Dataset

Info: https://www.kaggle.com/c/inaturalist-2018/data
Download: https://github.com/visipedia/inat_comp/tree/master/2018
Based on survey in CL: https://ieeexplore.ieee.org/document/9349197

Images have a max dimension of 800px and have been converted to JPEG format
You can select supercategories to include. By default 10 Super categories are
selected from the 14 available, based on at least having 100 categories (leaving
out Chromista, Protozoa, Bacteria), and omitting a random super category from
the remainder (Actinopterygii).

Example filename from the JSON: "file_name":
"train_val2018/Insecta/1455/994fa5...f1e360d34aae943.jpg"
"""

from typing import Any, Dict, List, Set

import os
import logging
import dill
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from os.path import expanduser
import pprint

from avalanche.checkpointing import constructor_based_serialization

from .inaturalist_data import INATURALIST_DATA


def pil_loader(path):
    """Load an Image with PIL"""
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class INATURALIST2018(Dataset):
    """INATURALIST Pytorch Dataset

    For default selection of 10 supercategories:

    - Training Images in total: 428,830
    - Validation Images in total:  23,229
    - Shape of images: torch.Size([1, 3, 600, 800])
    - Class counts per supercategory (both train/val):

        - 'Amphibia': 144,
        - 'Animalia': 178,
        - 'Arachnida': 114,
        - 'Aves': 1258,
        - 'Fungi': 321,
        - 'Insecta': 2031,
        - 'Mammalia': 234,
        - 'Mollusca': 262,
        - 'Plantae': 2917,
        - 'Reptilia': 284}
    """

    splits = ["train", "val", "test"]

    def_supcats = [
        "Amphibia",
        "Animalia",
        "Arachnida",
        "Aves",
        "Fungi",
        "Insecta",
        "Mammalia",
        "Mollusca",
        "Plantae",
        "Reptilia",
    ]

    def __init__(
        self,
        root=expanduser("~") + "/.avalanche/data/inaturalist2018/",
        split="train",
        transform=ToTensor(),
        target_transform=None,
        loader=pil_loader,
        download=True,
        supcats=None,
    ):
        super().__init__()
        # conda install -c conda-forge pycocotools
        from pycocotools.coco import COCO as jsonparser

        assert split in self.splits
        self.split = split  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.loader = loader
        self.log = logging.getLogger("avalanche")

        # Supercategories to include (None = all)
        self.supcats = supcats if supcats is not None else self.def_supcats

        if download:
            download_trainval = self.split in ["train", "val"]
            self.inat_data = INATURALIST_DATA(
                data_folder=root, trainval=download_trainval
            )

        # load annotations
        ann_file = f"{split}2018.json"
        self.log.info(f"Loading annotations from: {ann_file}")
        self.ds = jsonparser(annotation_file=os.path.join(root, ann_file))

        self.img_ids, self.targets = [], []  # targets field is required!
        self.cats_per_supcat: Dict[str, Set[int]] = {}

        # Filter full dataset parsed
        for ann in self.ds.anns.values():
            img_id = ann["image_id"]
            cat_id = ann["category_id"]

            # img = self.ds.loadImgs(img_id)[0]["file_name"]  # Img Path
            cat = self.ds.loadCats(cat_id)[0]  # Get category
            target = cat["name"]  # Is subdirectory
            supcat = cat["supercategory"]  # Is parent directory

            if self.supcats is None or supcat in self.supcats:  # Made selection
                # Add category to supercategory
                if supcat not in self.cats_per_supcat:
                    self.cats_per_supcat[supcat] = set()
                self.cats_per_supcat[supcat].add(int(target))  # Need int

                # Add to list
                self.img_ids.append(img_id)
                self.targets.append(target)
                # self.suptargets.append(supcat)

        cnt_per_supcat = {k: len(v) for k, v in self.cats_per_supcat.items()}
        self.log.info("Classes per supercategories:")
        self.log.info(pprint.pformat(cnt_per_supcat, indent=2))
        self.log.info(f"Images in total: {self.__len__()}")

    def _load_image(self, img_id: int) -> Image.Image:
        path = self.ds.loadImgs(img_id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, img_id) -> List[Any]:
        return self.ds.loadAnns(self.ds.getAnnIds(img_id))

    def __getitem__(self, index):
        id = self.img_ids[index]
        img = self._load_image(id)
        # target = self._load_target(id)
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_ids)


@dill.register(INATURALIST2018)
def checkpoint_INATURALIST2018(pickler, obj: INATURALIST2018):
    constructor_based_serialization(
        pickler,
        obj,
        INATURALIST2018,
        deduplicate=True,
        kwargs=dict(
            root=obj.root,
            split=obj.split,
            transform=obj.transform,
            target_transform=obj.target_transform,
            loader=obj.loader,
            supcats=obj.supcats,
        ),
    )


if __name__ == "__main__":
    # this litte example script can be used to visualize the first image
    # leaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    train_data = INATURALIST2018()
    test_data = INATURALIST2018(split="val")
    print("train size: ", len(train_data))
    print("test size: ", len(test_data))

    dataloader = DataLoader(train_data, batch_size=1)

    for batch_data in dataloader:
        x, y = batch_data
        plt.imshow(transforms.ToPILImage()(torch.squeeze(x)))
        plt.show()
        print(x.size())
        print(len(y))
        break

__all__ = ["INATURALIST2018"]
