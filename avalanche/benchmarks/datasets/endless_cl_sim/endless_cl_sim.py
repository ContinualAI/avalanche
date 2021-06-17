################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-06-2021                                                             #
# Author: TimmHess                                                             #
# E-mail: hess@ccc.cs.uni-frankfurt.de                                         #
# Website: continualai.org                                                     #
################################################################################

""" Endless-CL-Sim Dataset """

from pathlib import Path
import glob
import os
from typing import Union
from warnings import warn
import sys

from PIL import Image

from torch.utils.data import Dataset

from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.datasets.endless_cl_sim import \
    endless_cl_sim_data
from avalanche.benchmarks.datasets.downloadable_dataset import \
    DownloadableDataset

class ClassificationSubSequence(Dataset):
    def __init__(self, file_paths, targets, patch_size=64, transform=None, target_transform=None):
        self.file_paths = file_paths
        self.targets = targets
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform
        return

    def _pil_loader(self, file_path):
        with open(file_path, "rb") as f:
            img = Image.open(f).convert("RGB").resize(
                (self.patch_size, self.patch_size), Image.NEAREST)
        return img

    def __getitem__(self, index):
        img_path = self.file_paths[index]
        target = self.targets[index]

        img = self._pil_loader(img_path)
        
        img = self.transform(img)

        if not self.target_transform is None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.file_paths)

class EndlessCLSimDataset(DownloadableDataset):
    """ Endless-CL-Sim Dataset """ 
    def __init__(
        self,
        root: Union[str, Path] = None,
        *,
        scenario=None, # "Classes", "Illumination", "Weather"
        transform=None, target_transform=None,
        download=True, semseg=False
    ):

        # TODO: define dataloader
        if root is None:
            root = default_dataset_location('endless-cl-sim')
        print("root path:", root)

        if scenario is None and download:
            raise ValueError("No scenario defined to download!")

        super(EndlessCLSimDataset, self).__init__(
            root, download=download, verbose=True)

        self.scenario = scenario
        self.transform = transform
        self.target_transform = target_transform
        self.semseg = semseg

        self.train_sub_sequence_datasets = []
        self.test_sub_sequence_datasets = []

        # Download the dataset and initialize metadata
        self._load_dataset()
        return

    def _get_scenario_data(self):
        # TODO: define return data-type
        data = endless_cl_sim_data.data
        if self.semseg:
            raise NotImplementedError
        
        if self.scenario == "Classes":
            return data[0]
        if self.scenario == "Illumination":
            return data[1]
        if self.scenario == "Weather":
            return data[2]

        raise ValueError("Provided 'scenario' parameter is not valid!")
        
    def _prepare_subsequence_datasets(self, path) -> bool:
        # Get sequence dirs
        sequence_paths = glob.glob(path + os.path.sep + "*" + os.path.sep)

        # For every sequence (train, test)
        for sequence_path in sequence_paths:
            sub_sequence_paths = glob.glob(sequence_path + os.path.sep + "*" + os.path.sep)
            # Get sub-sequence dirs (0,1,....,n)       
            for sub_sequence_path in sub_sequence_paths:
                image_paths = []
                targets = []

                # Get class dirs
                class_name_dirs = class_name_dirs = [f.name for f \
                    in os.scandir(sub_sequence_path + os.path.sep) if f.is_dir()]
                
                # Load file_paths and targets
                for class_name in class_name_dirs:
                    try:
                        label = endless_cl_sim_data.default_classification_labelmap[class_name]
                    except:
                        ValueError(f"{class_name} is not part of the provided labelmap!")
                    class_path = sub_sequence_path + class_name + os.path.sep
                    for file_name in os.listdir(class_path):
                        image_paths.append(class_path + file_name)
                        targets.append(label)
                
                # Create sub-sequence dataset
                subseqeunce_dataset = ClassificationSubSequence(image_paths, targets, 
                    transform=self.transform, target_transform=self.target_transform)
                if "train" in (sequence_path.lower()):
                    self.train_sub_sequence_datasets.append(subseqeunce_dataset)
                elif "test" in (sequence_path.lower()):
                    self.test_sub_sequence_datasets.append(subseqeunce_dataset)
                else:
                    raise ValueError("Sequence path contains neighter 'train' not 'test' identifier!")
        
        # Check number of train and test subsequence datasets are equal
        if self.verbose:
            print("Num train subsequences:", len(self.train_sub_sequence_datasets), \
                "Num test subsequences:", len(self.test_sub_sequence_datasets))
        assert(len(self.train_sub_sequence_datasets) == len(self.test_sub_sequence_datasets))
        
        # Has run without errors
        if self.verbose:
            print("Successfully created subsequence datasets..")
        return True

    def __getitem__(self, index):
        return self.train_sub_sequence_datasets[index], self.test_sub_sequence_datasets[index]

    def __len__(self):
        return len(self.train_sub_sequence_datasets)

    def _download_dataset(self)->None:
        data_name = self._get_scenario_data()

        if self.verbose:
            print("Downloading " + data_name[1] + "...")
        file = self._download_file(data_name[1], data_name[0], data_name[2])
        if data_name[1].endswith('.zip'):
            if self.verbose:
                print(f'Extracting {data_name[0]}...')
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
        if not self.scenario is None:
            # Get data name
            scenario_data_name = self._get_scenario_data()
            scenario_data_name = scenario_data_name[0].split(".")[0]
            # Get list of directories in root
            root_file_list = glob.glob(str(self.root))
            # Check matching directory exists in endless_cl_sim_data
            match_path = None
            for data_name in endless_cl_sim_data.data:
                name = data_name[0].split(".")[0]

                # Omit non selected directories
                #print(f"{scenario_data_name} == {name}")
                if str(scenario_data_name) == str(name):
                    # Check there is such a directory
                    if (self.root / name).exists():
                        if not match_path is None:
                            raise ValueError("Two directories match the selected scenario!")
                        match_path = str(self.root / name)
                
            if match_path is None:
                return False
                
            is_subsequence_preparation_done = self._prepare_subsequence_datasets(match_path)
            if is_subsequence_preparation_done and self.verbose:
                print("Data is loaded..")
            else:
                return False    
            return True

        # If a 'generic'-endless-cl-sim-scenario has been selected
        print("loading generic dataset")
        is_subsequence_preparation_done = self._prepare_subsequence_datasets(str(self.root))
        if is_subsequence_preparation_done and self.verbose:
            print("Data is loaded...")
            print()
        else:
            return False

        # Finally
        return True

    def _download_error_message(self) -> str:
        scenario_data_name = self._get_scenario_data()
        all_urls = [
            name_url[1] for name_url in scenario_data_name
        ]

        base_msg = \
            '[Endless-CL-Sim] Error downloading the dataset!\n' \
            'You should download data manually using the following links:\n'

        for url in all_urls:
            base_msg += url 
            base_msg += '\n'

        base_msg += 'and place these files in ' + str(self.root)

        return base_msg



if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch
    
    #train_data = EndlessCLSimDataset(scenario="Classes", root="/data/avalanche")
    data = EndlessCLSimDataset(scenario=None, download=False, root="/data/avalanche/IncrementalClasses_Classification",
            transform=transforms.ToTensor())
    
    print("num subseqeunces: ", len(data.train_sub_sequence_datasets))

    sub_sequence_index = 0
    subsequence = data.train_sub_sequence_datasets[sub_sequence_index]

    print(f"num samples in subseqeunce {sub_sequence_index} = {len(subsequence)}")

    dataloader = DataLoader(subsequence, batch_size=1)

    for i, (img, target) in enumerate(dataloader):
        print(i)
        print(img.shape)
        img = torch.squeeze(img)
        img = transforms.ToPILImage()(img)
        print(img.size)
        #plt.imshow(img)
        #plt.show()
        break
    print("Done...")


__all__ = [
    'EndlessCLSimDataset'
]