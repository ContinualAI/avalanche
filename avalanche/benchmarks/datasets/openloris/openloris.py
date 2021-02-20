################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-11-2020                                                             #
# Author: ContinualAI                                                          #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" Tiny-Imagenet Pytorch Dataset """

import os
import sys
import pickle as pkl
import logging
from os.path import expanduser

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .openloris_data import OPENLORIS_DATA


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class OpenLORIS(Dataset):
    """ OpenLORIS Pytorch Dataset """

    def __init__(self, root=expanduser("~") + "/.avalanche/data/openloris/",
                 train=True, transform=ToTensor(), target_transform=None,
                 loader=pil_loader, download=False):

        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.loader = loader
        self.log = logging.getLogger("avalanche")

        # any scenario and factor is good here since we want just to load the
        # train images and targets with no particular order
        scen = 'domain'
        factor = 0
        ntask = 9

        if download:
            # self.core_data = OPENLORIS_DATA(data_folder=root)
            self.log.error(
                  "Download is not supported for this Dataset."
                  "You need to download the following files "
                  "manually:\n"
                  "- train.zip: "
                  "https://drive.google.com/u/0/uc?id"
                  "=11jgiPB2Z9WRI3bW6VSN8fJZgwFl5mLsF&export=download\n"
                  "- validation.zip: https://drive.google.com/u/0/uc?id="
                  "1ChoBAGcQ_wkclPXsel8CjJHC0tD7b4ga&export=download\n"
                  "- test.zip: https://drive.google.com/u/0/uc?id="
                  "1J7_ljcwSZNXo6KwlhRZoG0kiEcRK7U6x&export=download\n"
                  "- LUP.pkl: https://drive.google.com/u/0/uc?id="
                  "1Os8T30NZ3ZU8liHQPeVbo2nlOoPZuDSV&export=download\n"
                  "- Paths.pkl: https://drive.google.com/u/0/uc?id="
                  "1KnuYLdlG3VQrhgbtIANLki81ah8Thezj&export=download\n"
                  "- Labels.pkl: https://drive.google.com/u/0/uc?id="
                  "1GkmOxIAvmjSwo22UzmZTSlw8NSmU5Q9H&export=download\n"
                  "- batches_filelists.zip: "
                  "https://drive.google.com/u/0/uc?id="
                  "1r0gbo5_Qlzrdet1GPIrJpVSGRgFU7NEp&export=download\n"                  
                  "For more details, check the official website: "
                  "https://lifelong-robotic-vision.github.io/dataset/object\n"
            )
            sys.exit(0)

        self.log.info("Loading paths...")
        with open(os.path.join(root, 'Paths.pkl'), 'rb') as f:
            self.train_test_paths = pkl.load(f)

        self.log.info("Loading labels...")
        with open(os.path.join(root, 'Labels.pkl'), 'rb') as f:
            self.all_targets = pkl.load(f)
            self.train_test_targets = []
            for i in range(ntask + 1):
                self.train_test_targets += self.all_targets[scen][factor][i]

        self.log.info("Loading LUP...")
        with open(os.path.join(root, 'LUP.pkl'), 'rb') as f:
            self.LUP = pkl.load(f)

        self.idx_list = []
        if train:
            for i in range(ntask + 1):
                self.idx_list += self.LUP[scen][factor][i]
        else:
            self.idx_list = self.LUP[scen][factor][-1]

        self.paths = []
        self.targets = []

        for idx in self.idx_list:
            self.paths.append(self.train_test_paths[idx])
            self.targets.append(self.train_test_targets[idx])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target
                class.
        """

        target = self.targets[index]
        img = self.loader(
            os.path.join(
                self.root, self.paths[index]
            )
        )
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


if __name__ == "__main__":

    # this litte example script can be used to visualize the first image
    # leaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    train_data = OpenLORIS(download=True)
    test_data = OpenLORIS(train=False)
    print("train size: ", len(train_data))
    print("Test size: ", len(test_data))
    dataloader = DataLoader(train_data, batch_size=1)

    for batch_data in dataloader:
        x, y = batch_data
        plt.imshow(
            transforms.ToPILImage()(torch.squeeze(x))
        )
        plt.show()
        print(x.size())
        print(len(y))
        break

__all__ = [
    'OpenLORIS'
]
