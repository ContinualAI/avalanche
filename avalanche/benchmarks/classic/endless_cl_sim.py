################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-06-2021                                                             #
# Author: Timm Hess                                                            #
# E-mail: hess@ccc.cs.uni-frankfurt.de                                         #
# Website: www.continualai.org                                                 #
################################################################################

"""
This module contains the high-level EndlessCLSim scenario 
generator. It returns an iterable scenario object 
``GenericCLScenario`` given a number of configuration parameters.
"""

from avalanche.benchmarks.datasets.endless_cl_sim.endless_cl_sim import (
    EndlessCLSimDataset,
)
from pathlib import Path
from typing import List, Union, Optional, Any

from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Compose

from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.scenarios.deprecated.generators import dataset_benchmark
from avalanche.benchmarks.utils import _make_taskaware_classification_dataset

_default_transform = Compose([ToTensor()])

_scenario_names = ["Classes", "Illumination", "Weather"]


def EndlessCLSim(
    *,
    scenario: str = _scenario_names[0],
    patch_size: int = 64,
    sequence_order: Optional[List[int]] = None,
    task_order: Optional[List[int]] = None,
    train_transform: Optional[Any] = _default_transform,
    eval_transform: Optional[Any] = _default_transform,
    dataset_root: Optional[Union[str, Path]] = None,
    semseg=False
):
    """Creates a CL scenario for the Endless-Continual-Learning Simulator's
    derived `datasets <https://zenodo.org/record/4899267>`__, or custom
    datasets created from
    the Endless-Continual-Learning-Simulator's `standalone application <
    https://zenodo.org/record/4899294>`__.
    Both are part of the publication of `A Procedural World Generation
    Framework for Systematic Evaluation of Continual Learning
    <https://arxiv.org/abs/2106.02585>`__.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    All generated scenarios make use of 'task labels'. We regard a full dataset
    as one learning 'sequence', aligned to the terminology in the above paper,
    with 'subsequences' being the iterative learning tasks. Each subsequence
    is realized as one `AvalancheDataset` with ordering inforaced by task
    labels.

    :param scenario: Available, predefined, learning scenarios are:
        'Classes': An learning scenario based on incremental availability of
        object class examples,
        'Illumination': A learning scenario based on iteratively decreasing
        scene illumination.
        'Weather': A learning scenario based on iteratively shifting weather
        conditions.
    :param patch_size: The dimension of the image-patches. Int in the case of
            image-patch classification, because the image-patches need to be
            quadratic. Tuple of integers for image segmentation tasks.
    :param sequence_order: List of intergers indexing the subsequences,
            enables reordering of the subsequences, especially subsequences can
            be omitted. Defaults to None, loading subsequences in their
            original order.
    :param task_order: List of intergers, assigning task labels to each
            respective subsequence.
    :param train_transform: The transformation to apply to the training data.
            Defaults to `_default_transform`, i.e. conversion ToTensor of
            torchvision.
    :param eval_transform: The transformation to apply to the eval data.
            Defaults to `_default_transform`, i.e. conversion ToTensor of
            torchvision.
    :param dataset_root: Absolute path indicating where to store the dataset.
            Defaults to None, which means the default location for
            'endless-cl-sim' will be used.
    :param semseg: boolean to indicate the use of targets for a semantic
            segmentation task. Defaults to False.

    :returns: A properly initialized :class:`EndlessCLSim` instance.
    """
    # Check scenario name is valid
    assert scenario in _scenario_names, (
        "The selected scenario is not "
        "recognized: it should be "
        "'Classes', 'Illumination', "
        "or 'Weather'."
    )

    # Assign default dataset root if None provided
    if dataset_root is None:
        dataset_root = default_dataset_location("endless-cl-sim")

    # Download and prepare the dataset
    endless_cl_sim_dataset = EndlessCLSimDataset(
        root=dataset_root,
        scenario=scenario,
        transform=None,
        download=True,
        semseg=semseg,
    )

    # Default sequence_order if None
    if sequence_order is None:
        sequence_order = list(range(len(endless_cl_sim_dataset)))

    # Default sequence_order if None
    if task_order is None:
        task_order = list(range(len(endless_cl_sim_dataset)))

    train_datasets = []
    eval_datasets = []
    for i in range(len(sequence_order)):
        train_data, eval_data = endless_cl_sim_dataset[sequence_order[i]]

        train_data.transform = train_transform
        eval_data.transform = eval_transform

        train_datasets.append(
            _make_taskaware_classification_dataset(
                dataset=train_data, task_labels=task_order[i]
            )
        )
        eval_datasets.append(
            _make_taskaware_classification_dataset(
                dataset=eval_data, task_labels=task_order[i]
            )
        )

    scenario_obj = dataset_benchmark(train_datasets, eval_datasets)

    return scenario_obj


__all__ = ["EndlessCLSim"]


if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    from torchvision.transforms import ToPILImage, ToTensor
    import matplotlib.pyplot as plt

    scenario_obj = EndlessCLSim(
        scenario="Classes",
        sequence_order=[0, 1, 2, 3],
        task_order=[0, 1, 2, 3],
        semseg=True,
        dataset_root="/data/avalanche",
    )

    # FIXME: check_vision_benchmark function is crashing -> this is not..
    # check_vision_benchmark(scenario_obj)
    print(
        "The benchmark instance contains",
        len(scenario_obj.train_stream),
        "training experiences.",
    )

    for i, exp in enumerate(scenario_obj.train_stream):
        dataset, t = exp.dataset, exp.task_label
        print(dataset, t)
        print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=300)
    print("Train experience", exp.current_experience)

    for batch in dataloader:
        x, y, *other = batch
        print("X tensor:", x.shape)
        print("Y tensor:", y.shape)
        if len(other) > 0:
            print("T tensor:", other[0].shape)

        img = ToPILImage()(x[0])
        plt.title("Experience: " + str(exp.current_experience))
        plt.imshow(img)
        # plt.show()
        break

    print("Done..")
