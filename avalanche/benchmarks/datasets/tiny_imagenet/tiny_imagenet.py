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
import os
import logging
import sys
from zipfile import ZipFile
from os.path import expanduser
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve

filename = ('tiny-imagenet-200.zip',
            'http://cs231n.stanford.edu/tiny-imagenet-200.zip')


class TinyImagenet(Dataset):
    """Tiny Imagenet Pytorch Dataset"""

    def __init__(self, data_folder=expanduser("~") +
                 "/.avalanche/data/tinyimagenet/",
                 train=True, transform=ToTensor(),
                 target_transform=None, download=True):
        """
        Args:
            :param string data_folder: folder in which to download dataset
            :param boolean train: True for train set, False for test set
            :param fun transform: Pytorch transformation founction for x
            :param fun target_transform: Pytorch transformation founction for y
            :param bool download: True for downloading the dataset
        """
        self.log = logging.getLogger("avalanche")
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        if os.path.isabs(data_folder):
            self.data_folder = data_folder
        else:
            self.data_folder = os.path.join(
                os.path.dirname(__file__),
                data_folder
            )

        if os.path.exists(self.data_folder):
            if download:
                self.log.info(
                    "Directory {} already exists".format(self.data_folder))
        else:
            # Create target Directory for Tiny ImageNet data
            os.makedirs(self.data_folder)
            self.log.info("Directory {} created".format(self.data_folder))
            self.download = download
            self.download_tinyImageNet()

        self.data_folder = os.path.join(self.data_folder, 'tiny-imagenet-200')

        self.label2id, self.id2label = self.labels2dict()
        self.data, self.targets = self.load_data(train=train)

    def download_tinyImageNet(self):
        """ Downloads the TintImagenet Dataset """

        self.log.info("Downloading {}...".format(filename[1]))
        urlretrieve(filename[1], os.path.join(self.data_folder, filename[0]))

        with ZipFile(os.path.join(self.data_folder, filename[0]), 'r') as zipf:
            self.log.info('Extracting Tiny ImageNet images...')
            zipf.extractall(self.data_folder)
            self.log.info('Done!')

        self.log.info("Download complete.")

    def labels2dict(self):
        """
        Returns dictionaries to convert class names into progressive ids
        and viceversa.
        :returns: label2id, id2label: two Python dictionaries.
        """

        label2id = {}
        id2label = {}

        with open(os.path.join(self.data_folder, 'wnids.txt'), 'r') as f:

            reader = csv.reader(f)
            curr_idx = 0
            for ll in reader:
                if ll[0] not in label2id:
                    label2id[ll[0]] = curr_idx
                    id2label[curr_idx] = ll[0]
                    curr_idx += 1

        return label2id, id2label

    def load_data(self, train=True):
        """
        Load all images paths and targets.

        :param bool train: True for loading the training set, False for the
            test set.
        :return: train_set, test_set: (train_X_paths, train_y).
        """

        data = [[], []]

        classes = list(range(200))
        for class_id in classes:
            class_name = self.id2label[class_id]

            if train:
                X = self.get_train_images_paths(class_name)
                Y = [class_id] * len(X)
            else:
                # test set
                X = self.get_test_images_paths(class_name)
                Y = [class_id] * len(X)

            data[0] += X
            data[1] += Y

        return data

    def get_train_images_paths(self, class_name):
        """ Gets the training set image paths

            :param class_name: names of the classes of the images to be
                collected.
            :returns img_paths: list of strings (paths)
        """
        train_img_folder = os.path.join(self.data_folder,
                                        'train', class_name, 'images')

        img_paths = [os.path.join(train_img_folder, f)
                     for f in os.listdir(train_img_folder)
                     if os.path.isfile(os.path.join(train_img_folder, f))]

        return img_paths

    def get_test_images_paths(self, class_name):
        """ Gets the test set image paths

            :param class_name: names of the classes of the images to be
                collected.
            :returns img_paths: list of strings (paths)
        """

        val_img_folder = os.path.join(self.data_folder, 'val', 'images')

        valid_names = []
        # filter validation images by class using appropriate file
        with open(
                os.path.join(self.data_folder, 'val', 'val_annotations.txt'),
                'r') as f:

            reader = csv.reader(f, dialect='excel-tab')
            for ll in reader:
                if ll[1] == class_name:
                    valid_names.append(ll[0])

        img_paths = [os.path.join(val_img_folder, f) for f in valid_names]

        return img_paths

    def __len__(self):
        """ Returns the lenght of the set """
        return len(self.data)

    def __getitem__(self, index):
        """ Returns the index-th x, y pattern of the set """

        path, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


if __name__ == "__main__":

    # this litte example script can be used to visualize the first image
    # leaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    train_data = TinyImagenet()
    dataloader = DataLoader(train_data, batch_size=1)

    for batch_data in dataloader:
        x, y = batch_data
        plt.imshow(
            transforms.ToPILImage()(torch.squeeze(x))
        )
        plt.show()
        print(x.shape)
        print(y.shape)
        break


__all__ = [
    'TinyImagenet'
]
