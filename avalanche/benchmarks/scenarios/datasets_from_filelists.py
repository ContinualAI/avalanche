################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 21-06-2020                                                             #
# Author(s): Lorenzo Pellegrini, Vincenzo Lomonaco                             #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" This module contains useful utility functions and classes to generate
pytorch datasets based on fileists (Caffe style) """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from pathlib import Path

import torch.utils.data as data

from PIL import Image
import os
import os.path


def default_loader(path):
    """
    Sets the default image loader for the Pytorch Dataset.

    :param path: relative or absolute path of the file to load.

    :returns: Returns the image as a RGB PIL image.
    """
    return Image.open(path).convert('RGB')


def default_flist_reader(flist, root):
    """
    This reader reads a filelist and return a list of paths.

    :param flist: path of the flislist to read. The flist format should be:
        impath label\nimpath label\n ...(same to caffe's filelist)
    :param root: path to the dataset root. Each file defined in the file list
        will be searched in <root>/<impath>.

    :returns: Returns a list of paths (the examples to be loaded).
    """

    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((os.path.join(root, impath), int(imlabel)))

    return imlist


class FilelistDataset(data.Dataset):
    """
    This class extends the basic Pytorch Dataset class to handle filelists as
    main data source.
    """

    def __init__(
            self, root, flist, transform=None, target_transform=None,
            flist_reader=default_flist_reader, loader=default_loader):
        """
        This reader reads a filelist and return a list of paths.

        :param root: root path where the data to load are stored.
        :param flist: path of the flislist to read. The flist format should be:
            impath label\nimpath label\n ...(same to caffe's filelist)
        :param transform: eventual transformation to add to the input data (x)
        :param transform: eventual transformation to add to the targets (y)
        :param root: root path where the data to load are stored.
        :param flist_reader: loader function to use (for the filelists) given
            path.
        :param loader: loader function to use (for the real data) given path.
        """

        root = str(root)  # Manages Path objects
        flist = str(flist)  # Manages Path objects

        self.root = root
        self.imgs = flist_reader(flist, root)
        self.targets = [img_data[1] for img_data in self.imgs]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Returns next element in the dataset given the current index.

        :param index: index of the data to get.
        :return: loaded item.
        """

        impath, target = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Returns the total number of elements in the dataset.

        :return: Total number of dataset items.
        """

        return len(self.imgs)


def datasets_from_filelists(root, train_filelists, test_filelists,
                            complete_test_set_only=False):
    """
    This reader reads a filelist and return a list of paths.

    :param root: root path where the data to load are stored.
    :param train_filelists: list of paths to train filelists. The flist format
        should be: impath label\nimpath label\n ...(same to caffe's filelist)
    :param test_filelists: list of paths to test filelists. It can be also a
        single path when the datasets is the same for each batch.
    :param complete_test_set_only: if True, test_filelists must contain
        the path to a single filelist that will serve as the complete test set.
        Alternatively, test_filelists can be the path (str) to the complete test
        set filelist. If False, train_filelists and test_filelists must contain
        the same amount of filelists paths. Defaults to False.

    :return: list of tuples (train dataset, test dataset) for each train
        filelist in the list.
    """

    if complete_test_set_only and not (isinstance(test_filelists, str) or
                                       isinstance(test_filelists, Path)):
        if len(test_filelists) > 1:
            raise ValueError(
                'When complete_test_set_only is True, test_filelists must be a '
                'str, Path or a list with a single element describing the path '
                'to the complete test set.')
        test_filelists = test_filelists[0]
    if not complete_test_set_only:
        if len(test_filelists) != len(train_filelists):
            raise ValueError(
                'When complete_test_set_only is False, test_filelists and '
                'train_filelists must contain the same number of elements.')

    train_inc_datasets = \
        [FilelistDataset(root, tr_flist) for tr_flist in train_filelists]
    test_inc_datasets = \
        [FilelistDataset(root, te_flist) for te_flist in test_filelists]

    return train_inc_datasets, test_inc_datasets
