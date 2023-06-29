################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-06-2021                                                             #
# Author: Timm Hess                                                            #
# E-mail: hess@ccc.cs.uni-frankfurt.de                                         #
# Website: continualai.org                                                     #
################################################################################

"""Endless-CL-Sim Dataset."""

from pathlib import Path
import glob
import os
from typing import List, Optional, Union
from warnings import warn
import sys
import json

import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.datasets.endless_cl_sim import endless_cl_sim_data
from avalanche.benchmarks.datasets.downloadable_dataset import (
    DownloadableDataset,
)


class ClassificationSubSequence(Dataset):
    """Image-Patch Classification Subsequence Dataset"""

    def __init__(
        self,
        file_paths,
        targets,
        patch_size=64,
        labelmap_path=None,
        transform=None,
        target_transform=None,
    ):
        """Dataset containing image-patches and targets for one subsequence of
        an endless continual learning simulator's sequence, that has been
        converted for image-patch classification.

        :param file_paths: List that contains the paths to all images files
            that are part of this subsequence.
        :param targets: List that contains the targets (`object category
            names` (str)) for each respective image.
        :param patch_size: Int defining the quadratic patch-size the
            image-patches are resized to.
        :param labelmap_path: Path to a `labelmap.json` file that specifies
            a mapping from `object category names` to labels.
        :param transform: Eventual transformations to be applied to the image
            data.
        :param target_transform: Eventual transformations to be applied to the
            target data.
        """
        self.file_paths = file_paths
        self.targets = targets
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform

        self.labelmap = self._load_labelmap(labelmap_path)

        return

    def _pil_loader(self, file_path):
        with open(file_path, "rb") as f:
            img = (
                Image.open(f)
                .convert("RGB")
                .resize((self.patch_size, self.patch_size), Image.NEAREST)
            )
        return img

    def _load_labelmap(self, path):
        # If path is None, load default labelmap
        if path is None:
            return endless_cl_sim_data.default_classification_labelmap

        # If path is valid, load labelmap from json file
        elif Path(path).exists():
            with open(path) as file:
                json_array = json.load(file)
                labelmap = json_array["SegmentationClasses"]
                return labelmap

        # Finally, raise value error
        raise ValueError(f"path: {path} does not exist!")

    def _convert_target(self, target):
        return self.labelmap[target]

    def __getitem__(self, index: int):
        img_path = self.file_paths[index]
        target = self._convert_target(self.targets[index])

        img = self._pil_loader(img_path)
        img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.file_paths)


class VideoSubSequence(Dataset):
    """Video Subsequence Dataset"""

    def __init__(
        self,
        file_paths,
        target_paths,
        segmentation_file,
        classmap_file=None,
        patch_size=(240, 135),
        transform=None,
        target_transform=None,
    ):
        """Dataset that contains the (image) data and semantic segmentation
        targets for one subsequence of a video sequence.

        :param file_paths: List containing the paths to all images files that
            are part of this subsequence.
        :param target_paths: List containing the paths to all target files
            corresponding to the `file_paths`.
        :param segmentation_file: Path to a `segmentation.json` file that
            specifies a mapping from label indices to object
            (or object category) names. Defaults to None, which loads a
            predefined default mapping.
        :param classmap_file: Path to a `classmap.json' file that specifies
            the mapping from object (or object category) names to a
            respective label. Defaults to None, which loads a predefined
            default mapping.
        :param patch_size: Size of the images and target data to be resized to.
            Defaults to (240, 135).
        :param transform: Eventual transformations to be applied to the image
            data.
        :param target_transform: Eventual transformations to be applied to the
            target data.
        """
        self.file_paths = file_paths
        self.targets = target_paths
        self.segmentation_file = segmentation_file
        self.classmap_file = classmap_file
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = transform

        # Init classmap
        self.classmap = self._load_classmap(classmap_file=self.classmap_file)

        # Init labelmap
        self.labelmap = self._load_labelmap(labelmap_file=self.segmentation_file)
        return

    def _pil_loader(self, file_path, is_target=False):
        with open(file_path, "rb") as f:
            convert_identifier = "RGB"
            if is_target:
                convert_identifier = "L"
            img = (
                Image.open(f)
                .convert(convert_identifier)
                .resize((self.patch_size[0], self.patch_size[1]), Image.NEAREST)
            )
        return img

    def _load_classmap(self, classmap_file):
        classmap = {}
        if classmap_file is None:
            classmap = endless_cl_sim_data.default_semseg_classmap_obj
        elif Path(classmap_file).exists():
            with open(classmap_file) as file:
                json_array = json.load(file)
                classmap = json_array["ClassMapping"]
        else:
            raise ValueError(f"classmap_file: {classmap_file} does not exist!")
        return classmap

    def _load_labelmap(self, labelmap_file):
        labelmap = {}
        if Path(labelmap_file).exists():
            with open(labelmap_file) as file:
                json_array = json.load(file)

                segMin = json_array[0]["ObjectClassMapping"]
                segMax = json_array[1]["ObjectClassMapping"]

                for key in segMin:
                    labelmap[key] = [segMin[key], segMax[key]]
        else:
            raise ValueError(f"labelmap_file: {labelmap_file} does not exist!")
        return labelmap

    def _get_label_name(self, label):
        for key in self.labelmap:
            min_val, max_val = self.labelmap[key]
            if min_val == max_val:
                if label == min_val:
                    return key
            else:
                if label >= min_val and label <= max_val:
                    return key
        raise ValueError(f"label: {label} could not be converted!")

    def _convert_target(self, target):
        """Converts segmentation target (instance-segmented) according to
        classmap.
        """
        # Get all unique labels in target
        target = target.copy()
        unique_labels = torch.unique(torch.tensor(target)).numpy()

        for unique_label in unique_labels:
            # Get respective obj class label
            label_name = self._get_label_name(unique_label)
            class_label = self.classmap[label_name]
            # Convert instance label to object class label
            target[target == unique_label] = class_label
        return target

    def __getitem__(self, index: int):
        img_path = self.file_paths[index]
        target_path = self.targets[index]

        # Load image
        img = self._pil_loader(img_path, is_target=False)
        img = self.transform(img)

        # Load target
        target = self._pil_loader(target_path, is_target=True)
        target = self._convert_target(np.asarray(target))

        return img, target

    def __len__(self) -> int:
        return len(self.file_paths)


class EndlessCLSimDataset(DownloadableDataset):
    """Endless Continual Leanring Simulator Dataset"""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        scenario=None,
        patch_size=64,
        transform=None,
        target_transform=None,
        download=True,
        semseg=False,
        labelmap_path=None,
    ):
        """Creates an instance of the Endless-Continual-Leanring-Simulator
        Dataset.

        This dataset is able to download and prepare datasets derived from the
        Endless-Continual-Learning Simulator, including settings of incremental
        classes, decrasing illumination, and shifting weather conditions, as
        described in the paper `A Procedural World Generation Framework for
        Systematic Evaluation of Continual Learning
        <https://arxiv.org/abs/2106.02585>`__.
        Also custom datasets are supported
        when following the same structure. Such can be obtained from the
        `Endless-CL-Simulator standalone application
        <https://zenodo.org/record/4899294>`__.

        Please note:
        1) The EndlessCLSimDataset does not provide examples directly, but
        SubsequenceDatasets (ClassificationSubSequence, VideoSubSequence). Each
        SubSequenceDataset will contain the samples for one respective sub
        sequence.

        2) For video sequences currently only one sequence per dataset is
        supported!

        :param root: root for the datasets data. Defaults to None, which means
            that the default location for 'endless-cl-sim' will be used.
        :param scenario: identifier for the dataset to be used.
            Predefined options are 'Classes', for incremental classes scenario,
            'Illumination', for the decreasing lighting scenario,
            and 'Weather', for the scenario of shifting weather conditions.
            To load a custom (non-predefined/downloadable) dataset, the
            identifier needs to be set to None. Defaults to None.
        :param patch_size: optional size of image data to be loaded.
            For classification the patch_size is of type `int`, because we only
            consider quadratic input sizes. If the `semseg` flag is set,
            the patch_size type is `tuple`, with `(width, height)`.
        :param transform: optional transformations to be applied to the image
            data.
        :param target_transform: optional transformations to be applied to the
            targets.
        :param download: boolean to automatically download data.
            Defaults to True.
        :param semseg: boolean to indicate the use of targets for a
            semantic segmentation task. Defaults to False.
        :param labelmap_path: path (str) to a labelmap.json file,
            that provides a dictionary mapping 'class-names'(str) to
            class-labels(int). The 'class-names' are derived from the
            sub-directory names for each subsequence.
        """

        if root is None:
            root = default_dataset_location("endless-cl-sim")

        if scenario is None and download:
            raise ValueError("No scenario defined to download!")

        super(EndlessCLSimDataset, self).__init__(root, download=download, verbose=True)

        self.scenario = scenario
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform
        self.semseg = semseg
        self.labelmap_path = labelmap_path

        self.train_sub_sequence_datasets: List[ClassificationSubSequence] = []
        self.test_sub_sequence_datasets: List[ClassificationSubSequence] = []

        if self.semseg and self.patch_size == 64:
            self.patch_size = (240, 135)

        if self.semseg:
            assert isinstance(
                self.patch_size, tuple
            ), "If semseg is False, patch_size needs to be of type `int`"
        else:
            assert isinstance(
                self.patch_size, int
            ), "If semseg is True, patch_size needs to be of type `tuple`"

        # Download the dataset and initialize metadata
        self._load_dataset()
        return

    def _get_scenario_data(self):
        """Get data about the scenario.

        :return:
            tuple ("DataName.zip", "download-url", "MD5-checksum") of a
            derived data to be used, as defined in endless_cl_sim_data.py
        """
        data = endless_cl_sim_data.data
        # Video data
        if self.semseg:
            if self.scenario == "Classes":
                return data[3]
            if self.scenario == "Illumination":
                return data[4]
            if self.scenario == "Weather":
                return data[5]
        # Image-patch (classification) data
        if self.scenario == "Classes":
            return data[0]
        if self.scenario == "Illumination":
            return data[1]
        if self.scenario == "Weather":
            return data[2]

        raise ValueError("Provided 'scenario' parameter is not valid!")

    def _prepare_classification_subsequence_datasets(self, path) -> bool:
        """Prepare subsequences.

        :param path: (str) Path to the root of the data to be loaded.
        :return: success (bool): Boolean wether the preparation was successfull.
        """
        # Get sequence dirs
        sequence_paths = glob.glob(path + os.path.sep + "*" + os.path.sep)

        # For every sequence (train, test)
        for sequence_path in sequence_paths:
            sub_sequence_paths = glob.glob(
                sequence_path + os.path.sep + "*" + os.path.sep
            )
            # Get sub-sequence dirs (0,1,....,n)
            for sub_sequence_path in sub_sequence_paths:
                image_paths = []
                targets = []

                # Get class dirs
                class_name_dirs = [
                    f.name
                    for f in os.scandir(sub_sequence_path + os.path.sep)
                    if f.is_dir()
                ]

                # Load file_paths and targets
                for class_name in class_name_dirs:
                    class_path = sub_sequence_path + class_name + os.path.sep
                    for file_name in os.listdir(class_path):
                        image_paths.append(class_path + file_name)
                        targets.append(class_name)

                # Create sub-sequence dataset
                subsequence_dataset = ClassificationSubSequence(
                    image_paths,
                    targets,
                    patch_size=self.patch_size,
                    labelmap_path=self.labelmap_path,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
                if "train" in (sequence_path.lower()):
                    self.train_sub_sequence_datasets.append(subsequence_dataset)
                elif "test" in (sequence_path.lower()):
                    self.test_sub_sequence_datasets.append(subsequence_dataset)
                else:
                    raise ValueError(
                        "Sequence path contains neighter 'train' nor \
                            'test' identifier!"
                    )

        # Check number of train and test subsequence datasets are equal
        if self.verbose:
            print(
                "Num train subsequences:",
                len(self.train_sub_sequence_datasets),
                "Num test subsequences:",
                len(self.test_sub_sequence_datasets),
            )
        assert len(self.train_sub_sequence_datasets) == len(
            self.test_sub_sequence_datasets
        )

        # Has run without errors
        if self.verbose:
            print("Successfully created subsequence datasets..")
        return True

    def _load_sequence_indices(self, sequence_file):
        sequence_indices = {}
        with open(sequence_file) as file:
            json_array = json.load(file)

            for i in range(len(json_array)):
                sequence_indices[i] = json_array[i]["Sequence"]["ImageCounter"]
        return sequence_indices

    def _prepare_video_subsequence_datasets(self, path) -> bool:
        """Prepare video subsequence datasets.

        :param path: (str) Path to the root of the data to be loaded.
        :return: success (bool) Boolean wether the preparation was successfull.
        """
        # Get sequence dirs
        sequence_paths = glob.glob(path + os.path.sep + "*" + os.path.sep)

        # For every sequence (train, test)
        for sequence_path in sequence_paths:
            # Get dir contents (data + files)
            data_contents = glob.glob(sequence_path + os.path.sep + "*")

            image_paths = []
            target_paths = []
            sequence_file = None
            segmentation_file = None

            # Get Color, Seg dirs
            for data_content in data_contents:
                # If directory
                if Path(data_content).is_dir():
                    dir_name = data_content.split(os.path.sep)[-1]
                    if "Color" == dir_name:
                        # Extend color path
                        color_path = data_content + os.path.sep + "0" + os.path.sep
                        # Get all files
                        for file_name in sorted(os.listdir(color_path)):
                            image_paths.append(color_path + file_name)
                    elif "Seg" == dir_name:
                        # Extend seg path
                        seg_path = data_content + os.path.sep + "0" + os.path.sep
                        # Get all files
                        for file_name in sorted(os.listdir(seg_path)):
                            target_paths.append(seg_path + file_name)

                # If file
                if Path(data_content).is_file():
                    if "Sequence.json" in data_content:
                        sequence_file = data_content
                    elif "Segmentation.json" in data_content:
                        segmentation_file = data_content

            # Final checks
            if not len(image_paths) == len(target_paths):
                print("Not equal number of images and targets!")
                return False
            if sequence_file is None:
                print("No Sequence.json found!")
                return False
            if segmentation_file is None:
                print("No Segmentation.json found!")
                return False

            if self.verbose:
                print("All metadata checks complete!")

            sequence_indices = self._load_sequence_indices(sequence_file=sequence_file)

            if self.verbose:
                print("Sequence file loaded..")

            for i in range(len(sequence_indices)):
                last_index = sequence_indices[i]
                if (i + 1) == len(sequence_indices):
                    next_index = len(image_paths)
                else:
                    next_index = sequence_indices[i + 1]

                image_subsequence_paths = image_paths[last_index:next_index]
                target_subsequence_paths = target_paths[last_index:next_index]

                assert len(image_subsequence_paths) == len(target_subsequence_paths)

                # Create subsequence dataset
                subsequence_dataset = VideoSubSequence(
                    image_subsequence_paths,
                    target_subsequence_paths,
                    segmentation_file,
                    patch_size=self.patch_size,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
                if "train" in (sequence_path.lower()):
                    self.train_sub_sequence_datasets.append(subsequence_dataset)
                elif "test" in (sequence_path.lower()):
                    self.test_sub_sequence_datasets.append(subsequence_dataset)
                else:
                    raise ValueError(
                        "Sequence path contains neighter 'train' nor \
                             'test' identifiers!"
                    )
        return True

    def __getitem__(self, index):
        """Index dataset.

        :param index: Index
        :return: tuple (TrainSubSeqquenceDataset, TestSubSequenceDataset),
            the i-th subsequence data, as requested by the provided index.
        """
        return (
            self.train_sub_sequence_datasets[index],
            self.test_sub_sequence_datasets[index],
        )

    def __len__(self):
        return len(self.train_sub_sequence_datasets)

    def _download_dataset(self) -> None:
        data_name = self._get_scenario_data()

        if self.verbose:
            print("Downloading " + data_name[1] + "...")
        file = self._download_file(data_name[1], data_name[0], data_name[2])
        if data_name[1].endswith(".zip"):
            if self.verbose:
                print(f"Extracting {data_name[0]}...")
            extract_subdir = data_name[0].split(".")[0]
            extract_root = self._extract_archive(file, extract_subdir)

            # see all extracted files and extract all .zip again
            extract_root_file_list = glob.glob(str(extract_root) + "/*")
            for file_name in extract_root_file_list:
                sub_file_name = file_name.split("/")[-1]
                extract_subsubdir = extract_subdir + "/" + sub_file_name.split(".")[0]
                if self.verbose:
                    print(f"Extracting: {sub_file_name} to {extract_subdir}")
                self._extract_archive(file_name, extract_subdir, remove_archive=True)
                if self.verbose:
                    print("Extraction complete!")

            if self.verbose:
                print("All extractions complete!")

    def _load_metadata(self) -> bool:
        # If a 'named'-scenario has been selected
        if self.scenario is not None:
            # Get data name
            scenario_data_name = self._get_scenario_data()
            scenario_data_name = scenario_data_name[0].split(".")[0]
            # Check matching directory exists in endless_cl_sim_data
            match_path = None
            for data_name in endless_cl_sim_data.data:
                name = data_name[0].split(".")[0]
                # Omit non selected directories
                if str(scenario_data_name) == str(name):
                    # Check there is such a directory
                    if (self.root / name).exists():
                        if match_path is not None:
                            raise ValueError(
                                "Two directories match the selected scenario!"
                            )
                        match_path = str(self.root / name)

            if match_path is None:
                return False

            if not self.semseg:
                is_subsequence_preparation_done = (
                    self._prepare_classification_subsequence_datasets(match_path)
                )
            else:
                is_subsequence_preparation_done = (
                    self._prepare_video_subsequence_datasets(match_path)
                )

            if is_subsequence_preparation_done and self.verbose:
                print("Data is loaded..")
            else:
                return False
            return True

        # If a 'generic'-endless-cl-sim-scenario has been selected
        if not self.semseg:
            is_subsequence_preparation_done = (
                self._prepare_classification_subsequence_datasets(str(self.root))
            )
        else:
            is_subsequence_preparation_done = self._prepare_video_subsequence_datasets(
                str(self.root)
            )

        if is_subsequence_preparation_done and self.verbose:
            print("Data is loaded...")
        else:
            return False

        # Finally
        return True

    def _download_error_message(self) -> str:
        scenario_data_name = self._get_scenario_data()
        all_urls = [name_url[1] for name_url in scenario_data_name]

        base_msg = (
            "[Endless-CL-Sim] Error downloading the dataset!\n"
            "You should download data manually using the following links:\n"
        )

        for url in all_urls:
            base_msg += url
            base_msg += "\n"

        base_msg += "and place these files in " + str(self.root)

        return base_msg


if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    _default_transform = transforms.Compose([transforms.ToTensor()])

    # data = EndlessCLSimDataset(scenario="Classes", root="/data/avalanche",
    #                           semseg=True, transform=_default_transform)
    data = EndlessCLSimDataset(
        scenario=None,
        download=False,
        root="/data/avalanche/IncrementalClasses_Video",
        semseg=True,
        transform=_default_transform,
    )

    print("num subsequence:", len(data.train_sub_sequence_datasets))

    sub_sequence_index = 0
    subsequence = data.train_sub_sequence_datasets[sub_sequence_index]

    print(
        f"num samples in subsequence {sub_sequence_index} \
            = {len(subsequence)}"
    )

    dataloader = DataLoader(subsequence, batch_size=1)

    for i, (img, target) in enumerate(dataloader):
        print(i)
        print(img.shape)
        img = torch.squeeze(img)
        img = transforms.ToPILImage()(img)
        print("img size:", img.size)
        print("targets:", np.unique(target))
        # plt.imshow(img)
        # plt.show()
        break
    print("Done...")


__all__ = ["EndlessCLSimDataset"]
