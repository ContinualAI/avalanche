################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 21-06-2020                                                             #
# Author(s): Lorenzo Pellegrini, Vincenzo Lomonaco                             #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" This module contains useful utility functions and classes to generate
pytorch datasets based on filelists (Caffe style) """

from pathlib import Path
from typing import (
    Callable,
    Generic,
    List,
    Tuple,
    Sequence,
    Optional,
    TypeVar,
    Union,
)

import torch.utils.data as data

from PIL import Image
import os
import os.path
import dill

from torch import Tensor
from torchvision.transforms.functional import crop

from avalanche.checkpointing import constructor_based_serialization

from .transform_groups import XTransform, YTransform


def default_image_loader(path):
    """
    Sets the default image loader for the Pytorch Dataset.

    :param path: relative or absolute path of the file to load.

    :returns: Returns the image as a RGB PIL image.
    """
    return Image.open(path).convert("RGB")


def default_flist_reader(flist: Union[str, Path]) -> List[Tuple[str, int]]:
    """
    This reader reads a filelist and return a list of paths.

    :param flist: path of the flislist to read. The flist format should be:
        impath label, impath label,  ...(same to caffe's filelist)

    :returns: Returns a list of paths (the examples to be loaded).
    """

    imlist = []
    with open(flist, "r") as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


T = TypeVar("T", covariant=True)
TTargetsType = TypeVar("TTargetsType")

PathALikeT = Union[Path, str]
CoordsT = Union[int, float]
CropBoxT = Tuple[CoordsT, CoordsT, CoordsT, CoordsT]
FilesDefT = Union[
    Tuple[PathALikeT, TTargetsType], Tuple[PathALikeT, TTargetsType, Sequence[int]]
]


class PathsDataset(data.Dataset[Tuple[T, TTargetsType]], Generic[T, TTargetsType]):
    """
    This class extends the basic Pytorch Dataset class to handle list of paths
    as the main data source.
    """

    def __init__(
        self,
        root: Optional[PathALikeT],
        files: Sequence[FilesDefT[TTargetsType]],
        transform: XTransform = None,
        target_transform: YTransform = None,
        loader: Callable[[str], T] = default_image_loader,
    ):
        """
        Creates a File Dataset from a list of files and labels.

        :param root: root path where the data to load are stored. May be None.
        :param files: list of tuples. Each tuple must contain two elements: the
            full path to the pattern and its class label. Optionally, the tuple
            may contain a third element describing the bounding box to use for
            cropping (top, left, height, width).
        :param transform: eventual transformation to add to the input data (x)
        :param target_transform: eventual transformation to add to the targets
            (y)
        :param loader: loader function to use (for the real data) given path.
        """

        self.root: Optional[Path] = Path(root) if root is not None else None
        self.imgs = files
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

        img_description = self.imgs[index]
        impath = img_description[0]
        target = img_description[1]
        bbox = None
        if len(img_description) > 2:
            bbox = img_description[2]

        if self.root is not None:
            impath = self.root / impath
        img = self.loader(impath)

        # If a bounding box is provided, crop the image before passing it to
        # any user-defined transformation.
        if bbox is not None:
            if isinstance(bbox, Tensor):
                bbox = bbox.tolist()
            # crop accepts PIL images, too
            img = crop(img, *bbox)  # type: ignore

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


class FilelistDataset(PathsDataset[T, int]):
    """
    This class extends the basic Pytorch Dataset class to handle filelists as
    main data source.
    """

    def __init__(
        self,
        root,
        flist,
        transform=None,
        target_transform=None,
        flist_reader=default_flist_reader,
        loader=default_image_loader,
    ):
        """
        This reader reads a filelist and return a list of paths.

        :param root: root path where the data to load are stored. May be None.
        :param flist: path of the flislist to read. The flist format should be:
            impath label\nimpath label\n ...(same to caffe's filelist).
        :param transform: eventual transformation to add to the input data (x).
        :param target_transform: eventual transformation to add to the targets
            (y).
        :param flist_reader: loader function to use (for the filelists) given
            path.
        :param loader: loader function to use (for the real data) given path.
        """

        self.flist = str(flist)  # Manages Path objects
        files_and_labels = flist_reader(flist)
        super().__init__(
            root,
            files_and_labels,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
        )


@dill.register(FilelistDataset)
def checkpoint_FilelistDataset(pickler, obj: FilelistDataset):
    constructor_based_serialization(
        pickler,
        obj,
        FilelistDataset,
        deduplicate=True,
        kwargs=dict(
            root=obj.root,
            flist=obj.flist,
            transform=obj.transform,
            target_transform=obj.target_transform,
            loader=obj.loader,
        ),
    )


def datasets_from_filelists(
    root,
    train_filelists,
    test_filelists,
    complete_test_set_only=False,
    train_transform=None,
    train_target_transform=None,
    test_transform=None,
    test_target_transform=None,
):
    """
    This reader reads a list of Caffe-style filelists and returns the proper
    Dataset objects.

    A Caffe-style list is just a text file where, for each line, two elements
    are described: the path to the pattern (relative to the root parameter)
    and its class label. Those two elements are separated by a single white
    space.

    This method reads each file list and returns a separate
    dataset for each of them.

    Beware that the parameters must be **list of paths to Caffe-style
    filelists**. If you need to create a dataset given a list of
    **pattern paths**, use `datasets_from_paths` instead.

    :param root: root path where the data to load are stored. May be None.
    :param train_filelists: list of paths to train filelists. The flist format
        should be: impath label\\nimpath label\\n ...(same to Caffe's filelist).
    :param test_filelists: list of paths to test filelists. It can be also a
        single path when the datasets is the same for each batch.
    :param complete_test_set_only: if True, test_filelists must contain
        the path to a single filelist that will serve as the complete test set.
        Alternatively, test_filelists can be the path (str) to the complete test
        set filelist. If False, train_filelists and test_filelists must contain
        the same amount of filelists paths. Defaults to False.
    :param train_transform: The transformation to apply to training patterns.
        Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param test_transform: The transformation to apply to test patterns.
        Defaults to None.
    :param test_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.

    :return: list of tuples (train dataset, test dataset) for each train
        filelist in the list.
    """

    if complete_test_set_only:
        if not (isinstance(test_filelists, str) or isinstance(test_filelists, Path)):
            if len(test_filelists) > 1:
                raise ValueError(
                    "When complete_test_set_only is True, test_filelists must "
                    "be a str, Path or a list with a single element describing "
                    "the path to the complete test set."
                )
            else:
                test_filelists = test_filelists[0]
        else:
            test_filelists = [test_filelists]
    else:
        if len(test_filelists) != len(train_filelists):
            raise ValueError(
                "When complete_test_set_only is False, test_filelists and "
                "train_filelists must contain the same number of elements."
            )

    transform_groups = dict(
        train=(train_transform, train_target_transform),
        eval=(test_transform, test_target_transform),
    )

    # import here to prevent circular import issue
    from .utils import as_taskaware_classification_dataset

    train_inc_datasets = [
        as_taskaware_classification_dataset(
            FilelistDataset(root, tr_flist),
            transform_groups=transform_groups,
            initial_transform_group="train",
        )
        for tr_flist in train_filelists
    ]
    test_inc_datasets = [
        as_taskaware_classification_dataset(
            FilelistDataset(root, te_flist),
            transform_groups=transform_groups,
            initial_transform_group="eval",
        )
        for te_flist in test_filelists
    ]

    return train_inc_datasets, test_inc_datasets


def datasets_from_paths(
    train_list,
    test_list,
    complete_test_set_only=False,
    train_transform=None,
    train_target_transform=None,
    test_transform=None,
    test_target_transform=None,
):
    """
    This utility takes, for each dataset to generate, a list of tuples each
    containing two elements: the full path to the pattern and its class label.
    Optionally, the tuple may contain a third element describing the bounding
    box to use for cropping.

    This is equivalent to `datasets_from_filelists`, which description
    contains more details on the behaviour of this utility. The two utilities
    differ in which `datasets_from_filelists` accepts paths to Caffe-style
    filelists while this one is able to create the datasets from an in-memory
    list.

    Note: this utility may try to detect (and strip) the common root path of
    all patterns in order to save some RAM memory.

    :param train_list: list of lists. Each list must contain tuples of two
        elements: the full path to the pattern and its class label. Optionally,
        the tuple may contain a third element describing the bounding box to use
        for cropping (top, left, height, width).
    :param test_list: list of lists. Each list must contain tuples of two
        elements: the full path to the pattern and its class label. Optionally,
        the tuple may contain a third element describing the bounding box to use
        for cropping (top, left, height, width). It can be also a single list
        when the test dataset is the same for each experience.
    :param complete_test_set_only: if True, test_list must contain a single list
        that will serve as the complete test set. If False, train_list and
        test_list must describe the same amount of datasets. Defaults to False.
    :param train_transform: The transformation to apply to training patterns.
        Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param test_transform: The transformation to apply to test patterns.
        Defaults to None.
    :param test_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.

    :return: A list of tuples (train dataset, test dataset).
    """

    if complete_test_set_only:
        # Check if the single dataset was passed as [Tuple1, Tuple2, ...]
        # or as [[Tuple1, Tuple2, ...]]
        if not isinstance(test_list[0], tuple):
            if len(test_list) > 1:
                raise ValueError(
                    "When complete_test_set_only is True, test_list must "
                    "be a single list of tuples or a nested list containing "
                    "a single lis of tuples"
                )
            else:
                test_list = test_list[0]
        else:
            test_list = [test_list]
    else:
        if len(test_list) != len(train_list):
            raise ValueError(
                "When complete_test_set_only is False, test_list and "
                "train_list must contain the same number of elements."
            )

    transform_groups = dict(
        train=(train_transform, train_target_transform),
        eval=(test_transform, test_target_transform),
    )

    common_root = None

    # Detect common root
    try:
        all_paths = [
            pattern_tuple[0] for exp_list in train_list for pattern_tuple in exp_list
        ] + [pattern_tuple[0] for exp_list in test_list for pattern_tuple in exp_list]

        common_root = os.path.commonpath(all_paths)
    except ValueError:
        # commonpath may throw a ValueError in different situations!
        # See the official documentation for more details
        pass

    if common_root is not None and len(common_root) > 0 and common_root != "/":
        has_common_root = True
        common_root = str(common_root)
    else:
        has_common_root = False
        common_root = None

    if has_common_root:
        # print(f'Common root found: {common_root}!')
        # All paths have a common filesystem root
        # Remove it from all paths!
        single_path_case = False
        tr_list = list()
        te_list = list()

        for idx_exp_list in range(len(train_list)):
            if single_path_case:
                break
            st_list = list()
            for x in train_list[idx_exp_list]:
                rel = os.path.relpath(x[0], common_root)
                if len(rel) == 0 or rel == ".":
                    # May happen if the dataset has a single path
                    single_path_case = True
                    break
                st_list.append((rel, *x[1:]))
            tr_list.append(st_list)

        for idx_exp_list in range(len(test_list)):
            if single_path_case:
                break
            st_list = list()
            for x in test_list[idx_exp_list]:
                rel = os.path.relpath(x[0], common_root)
                if len(rel) == 0 or rel == ".":
                    # May happen if the dataset has a single path
                    single_path_case = True
                    break
                st_list.append((rel, *x[1:]))
            te_list.append(st_list)
        if not single_path_case:
            train_list = tr_list
            test_list = te_list
        else:
            has_common_root = False
            common_root = None

    from avalanche.benchmarks.utils import as_taskaware_classification_dataset

    train_inc_datasets = [
        as_taskaware_classification_dataset(
            PathsDataset(common_root, tr_flist),
            transform_groups=transform_groups,
            initial_transform_group="train",
        )
        for tr_flist in train_list
    ]
    test_inc_datasets = [
        as_taskaware_classification_dataset(
            PathsDataset(common_root, te_flist),
            transform_groups=transform_groups,
            initial_transform_group="eval",
        )
        for te_flist in test_list
    ]

    return train_inc_datasets, test_inc_datasets


def common_paths_root(
    exp_list: Sequence[FilesDefT],
) -> Tuple[Union[str, None], Sequence[FilesDefT]]:
    common_root = None

    # Detect common root
    try:
        all_paths = [pattern_tuple[0] for pattern_tuple in exp_list]

        common_root = os.path.commonpath(all_paths)
    except ValueError:
        # commonpath may throw a ValueError in different situations!
        # See the official documentation for more details
        pass

    if common_root is not None and len(common_root) > 0 and common_root != "/":
        has_common_root = True
        common_root = str(common_root)
    else:
        has_common_root = False
        common_root = None

    exp_list_result: Sequence[FilesDefT]

    if has_common_root:
        # print(f'Common root found: {common_root}!')
        # All paths have a common filesystem root
        # Remove it from all paths!
        single_path_case = False
        exp_tuples: List[FilesDefT] = list()

        for x in exp_list:
            if single_path_case:
                break

            rel = os.path.relpath(x[0], common_root)
            if len(rel) == 0 or rel == ".":
                # May happen if the dataset has a single path
                single_path_case = True
                break
            exp_tuples.append((rel, *x[1:]))  # type: ignore

        if not single_path_case:
            exp_list_result = exp_tuples
        else:
            exp_list_result = exp_list
            common_root = None
    else:
        exp_list_result = exp_list

    return common_root, exp_list_result


__all__ = [
    "default_image_loader",
    "default_flist_reader",
    "PathsDataset",
    "FilelistDataset",
    "datasets_from_filelists",
    "datasets_from_paths",
    "common_paths_root",
]
