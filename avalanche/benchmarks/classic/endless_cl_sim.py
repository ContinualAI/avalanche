################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Timm Hess                                                         #
# E-mail: hess@ccc.cs.uni-frankfurt.de                                         #
# Website: www.continualai.org                                                 #
################################################################################


""" 
This module contains the high-level Endless-Continual-Learning-Simulator's 
scenario generator. It basically returns an interable scenario object 
``GenericCLScenario`` given a number of configuration parameters.
"""

from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset
from avalanche.benchmarks.datasets.endless_cl_sim.endless_cl_sim import EndlessCLSimDataset
from pathlib import Path
from typing import List, Union, Optional, Any

from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Compose

from avalanche.benchmarks.classic.classic_benchmarks_utils import \
    check_vision_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.datasets.endless_cl_sim import endless_cl_sim

_default_transform = Compose([
    ToTensor()
])

_scenario_names = ["Classes", "Illumination", "Weather"]

def EndlessCLSim(
        *,
        scenario: str = _scenario_names[0],
        patch_size: int = 64,
        sequence_order: Optional[List[int]] = None,
        task_order: Optional[List[int]] = None,
        train_transform: Optional[Any] = _default_transform,
        eval_transform: Optional[Any] = _default_transform,
        dataset_root: Union[str, Path] = None):

    """
    Creates a default CL scenario from the EndlessCLSimulator derived datasets. 
    """
    # Check scenario name is valid
    assert(scenario in _scenario_names), "The selected scenario is not "\
                                         "recognized: it should be "\
                                         "'Classes', 'Illumination', "\
                                         "or 'Weather'."

    # Assign default dataset root if None provided
    if dataset_root is None:
        dataset_root = default_dataset_location('endless-cl-sim')

    # Download and prepare the dataset
    endless_cl_sim_dataset = EndlessCLSimDataset(root=dataset_root, 
            scenario=scenario, transform=None, 
            download=True, semseg=False)
    
    if sequence_order is None:
        sequence_order = list(range(len(endless_cl_sim_dataset)))


    train_datasets = []
    eval_datasets = []
    for i in range(len(sequence_order)):
        train_data, eval_data = endless_cl_sim_dataset[sequence_order[i]]

        train_data.transform = train_transform
        eval_data.transform = eval_transform

        train_datasets.append(
            AvalancheDataset(
                dataset=train_data, task_labels=task_order[i]
            )
        )
        eval_datasets.append(
            AvalancheDataset(
                dataset=eval_data, task_labels=task_order[i]
            )
        )
    
    scenario_obj = dataset_benchmark(train_datasets, eval_datasets)

    return scenario_obj


__all__ = [
    'EndlessCLSim'
]


if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    from torchvision.transforms import ToPILImage, ToTensor
    import matplotlib.pyplot as plt

    scenario_obj = EndlessCLSim(scenario="Classes",
            sequence_order=[0,1,2,3],
            task_order=[0,1,2,3],
            dataset_root="/data/avalanche")
 
    #FIXME: check_vision_benchmark function is crashing -> this is not..
    #check_vision_benchmark(scenario_obj)
    print('The benchmark instance contains',
          len(scenario_obj.train_stream), 'training experiences.')

    for i, exp in enumerate(scenario_obj.train_stream):
        dataset, t = exp.dataset, exp.task_label
        print(dataset, t)
        print(len(dataset))
    
    dataloader = DataLoader(dataset, batch_size=300)
    print('Train experience', exp.current_experience)
    
    for batch in dataloader:
        x,y, *other = batch
        print('X tensor:', x.shape)
        print('Y tensor:', y.shape)
        if len(other) > 0:
            print('T tensor:', other[0].shape)

        img = ToPILImage()(x[0])
        plt.title('Experience: ' + str(exp.current_experience))
        plt.imshow(img)
        #plt.show()
        break

    print("Done..")