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

""" CUB200 Pytorch Dataset """

import os
import logging
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, \
    extract_archive
from torch.utils.data.dataset import Dataset


class CUB200(Dataset):

    filename = 'images.tgz'
    metadata = "lists.tgz"
    basefolder = "images"
    tgz_md5 = '2bbe304ef1aa3ddb6094aa8f53487cf2'

    def __init__(self, root, train=True,
                 transform=None, loader=default_loader, download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.train = train
        self.log = logging.getLogger("avalanche")

        if download:
            self.log.error(
                  "Download is not supported for this Dataset."
                  "You need to download 'images.tgz' and 'lists.tgz' manually "
                  "at: http://www.vision.caltech.edu/visipedia/CUB-200.html"
            )

        if not os.path.exists(os.path.join(self.root, self.filename[:-4])):
            extract_archive(os.path.join(self.root, self.filename))
        if not os.path.exists(os.path.join(self.root, self.metadata[:-4])):
            extract_archive(os.path.join(self.root, self.metadata))

        if not self._check_integrity():
            raise RuntimeError('Dataset corrupted')

    def _load_metadata(self):

        self.data_paths = []
        self.targets = []

        if self.train:
            fname = "train.txt"
        else:
            fname = "test.txt"

        with open(os.path.join(self.root, self.metadata[:-4], fname), "r") \
                as rf:
            for line in rf:
                label = int(line[:3])
                self.data_paths.append(line)
                self.targets.append(label)

    def _check_integrity(self):

        self._load_metadata()

        fpath = os.path.join(self.root, self.filename)
        if not check_integrity(fpath, self.tgz_md5):
            return False
        return True

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        sample = self.data_paths[idx]
        path = os.path.join(self.root, self.basefolder, sample)
        # Targets start at 1 by default, so shift to 0
        target = self.targets[idx] - 1
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


if __name__ == "__main__":

    dataset = CUB200(root="~/.avalanche/data/cub200/", train=False)
    print("test data len:", len(dataset))
    dataset = CUB200(root="~/.avalanche/data/cub200/", train=True)
    print("train data len:", len(dataset))


__all__ = [
    'CUB200'
]
