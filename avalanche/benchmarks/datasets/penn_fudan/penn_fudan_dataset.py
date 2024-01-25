# The dataset code has been adapted from:
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# from https://github.com/pytorch/tutorials
# which has been distributed under the following license:
################################################################################
# BSD 3-Clause License
#
# Copyright (c) 2017, Pytorch contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################

# For the Avalanche data loader adaptation:
################################################################################
# Copyright (c) 2022 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 21-03-2022                                                             #
# Author: Lorenzo Pellegrini                                                   #
#                                                                              #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################


from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import dill

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from avalanche.benchmarks.datasets import (
    SimpleDownloadableDataset,
    default_dataset_location,
)
from avalanche.benchmarks.datasets.penn_fudan.penn_fudan_data import (
    penn_fudan_data,
)
from avalanche.checkpointing import constructor_based_serialization


def default_mask_loader(mask_path):
    return Image.open(mask_path)


class PennFudanDataset(SimpleDownloadableDataset):
    """
    The Penn-Fudan Pedestrian detection and segmentation dataset

    Adapted from the "TorchVision Object Detection Finetuning Tutorial":
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        transform=None,
        loader=default_loader,
        mask_loader=default_mask_loader,
        download=True
    ):
        """
        Creates an instance of the Penn-Fudan dataset.

        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            "pennfudanped" will be used.
        :param transform: The transformation to apply to (img, annotations)
            values.
        :param loader: The image loader to use.
        :param mask_loader: The mask image loader to use.
        :param download: If True, the dataset will be downloaded if needed.
        """

        if root is None:
            root = default_dataset_location("pennfudanped")

        self.imgs: Sequence[Path] = None  # type: ignore
        self.masks: Sequence[Path] = None  # type: ignore
        self.targets: List[Dict] = None  # type: ignore
        self.transform = transform
        self.loader = loader
        self.mask_loader = mask_loader

        super().__init__(
            root,
            penn_fudan_data[0],
            penn_fudan_data[1],
            download=download,
            verbose=True,
        )

        self._load_dataset()

    def _load_metadata(self):
        # load all image files, sorting them to
        # ensure that they are aligned
        imgs = (self.root / "PennFudanPed" / "PNGImages").iterdir()
        masks = (self.root / "PennFudanPed" / "PedMasks").iterdir()

        self.imgs = list(sorted(imgs))
        self.masks = list(sorted(masks))

        self.targets = [self.make_targets(i) for i in range(len(self.imgs))]
        return Path(self.imgs[0]).exists() and Path(self.masks[0]).exists()

    def make_targets(self, idx):
        # load images and masks
        mask_path = self.masks[idx]

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = self.mask_loader(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin: np.integer = np.min(pos[1])
            xmax: np.integer = np.max(pos[1])
            ymin: np.integer = np.min(pos[0])
            ymax: np.integer = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes_as_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        del boxes
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes_as_tensor[:, 3] - boxes_as_tensor[:, 1]) * (
            boxes_as_tensor[:, 2] - boxes_as_tensor[:, 0]
        )
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes_as_tensor
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        return target

    def __getitem__(self, idx):
        target = self.targets[idx]
        img_path = self.imgs[idx]
        img = self.loader(img_path)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


@dill.register(PennFudanDataset)
def checkpoint_PennFudanDataset(pickler, obj: PennFudanDataset):
    constructor_based_serialization(
        pickler,
        obj,
        PennFudanDataset,
        deduplicate=True,
        kwargs=dict(
            root=obj.root,
            transform=obj.transform,
            loader=obj.loader,
            mask_loader=obj.mask_loader,
        ),
    )


if __name__ == "__main__":
    # this little example script can be used to visualize the first image
    # loaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    train_data = PennFudanDataset(
        transform=lambda im, ann: (transforms.ToTensor()(im), ann)
    )
    dataloader = DataLoader(train_data, batch_size=1)

    for batch_data in dataloader:
        x, y = batch_data
        plt.imshow(transforms.ToPILImage()(torch.squeeze(x)))
        plt.show()
        print(x.shape)
        print(y)
        break


__all__ = ["PennFudanDataset"]
