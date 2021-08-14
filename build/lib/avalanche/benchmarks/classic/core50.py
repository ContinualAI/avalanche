################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" This module contains the high-level CORe50 scenario generator. It
basically returns a iterable scenario object ``GenericCLScenario`` given a
number of configuration parameters."""
from pathlib import Path
from typing import Union

from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import get_default_dataset_location
from avalanche.benchmarks.scenarios.generic_benchmark_creation import \
    create_generic_benchmark_from_filelists
from avalanche.benchmarks.datasets.core50.core50 import CORe50Dataset

nbatch = {
    'ni': 8,
    'nc': 9,
    'nic': 79,
    'nicv2_79': 79,
    'nicv2_196': 196,
    'nicv2_391': 391
}

scen2dirs = {
    'ni': "batches_filelists/NI_inc/",
    'nc': "batches_filelists/NC_inc/",
    'nic': "batches_filelists/NIC_inc/",
    'nicv2_79': "NIC_v2_79/",
    'nicv2_196': "NIC_v2_196/",
    'nicv2_391': "NIC_v2_391/"
}


def CORe50(root: Union[str, Path] = get_default_dataset_location('core50'),
           scenario: str = "nicv2_391",
           run: int = 0,
           object_lvl: bool = True,
           mini: bool = False,
           train_transform=None,
           eval_transform=None):
    """
    Creates a CL scenario for CORe50.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    This generator can be used to obtain the NI, NC, NIC and NICv2-* scenarios.

    The scenario instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The task label "0" will be assigned to each experience.

    The scenario API is quite simple and is uniform across all scenario
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param root: Absolute path indicating where to store the dataset and related
        metadata. By default they will be stored in
        "~/.avalanche/datasets/core50/data/".
    :param scenario: CORe50 main scenario. It can be chosen between 'ni', 'nc',
        'nic', 'nicv2_79', 'nicv2_196' or 'nicv2_391.'
    :param run: number of run for the scenario. Each run defines a different
        ordering. Must be a number between 0 and 9.
    :param object_lvl: True for a 50-way classification at the object level.
        False if you want to use the categories as classes. Default to True.
    :param mini: True for processing reduced 32x32 images instead of the
        original 128x128. Default to False.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.

    :returns: a properly initialized :class:`GenericCLScenario` instance.
    """

    assert (0 <= run <= 9), "Pre-defined run of CORe50 are only 10. Indicate " \
                            "a number between 0 and 9."
    assert (scenario in nbatch.keys()), "The selected scenario is note " \
                                        "recognized: it should be 'ni', 'nc'," \
                                        "'nic', 'nicv2_79', 'nicv2_196' or " \
                                        "'nicv2_391'."

    # Download the dataset and initialize filelists
    core_data = CORe50Dataset(root=root, mini=mini)

    root = core_data.root
    if mini:
        bp = "core50_32x32"
    else:
        bp = "core50_128x128"
    root_img = root / bp

    if object_lvl:
        suffix = "/"
    else:
        suffix = "_cat/"
    filelists_bp = scen2dirs[scenario][:-1] + suffix + "run" + str(run)
    train_failists_paths = []
    for batch_id in range(nbatch[scenario]):
        train_failists_paths.append(
            root / filelists_bp / ("train_batch_" +
                                   str(batch_id).zfill(2) + "_filelist.txt"))

    scenario_obj = create_generic_benchmark_from_filelists(
        root_img, train_failists_paths,
        [root / filelists_bp / "test_filelist.txt"],
        task_labels=[0 for _ in range(nbatch[scenario])],
        complete_test_set_only=True,
        train_transform=train_transform,
        eval_transform=eval_transform)

    return scenario_obj


__all__ = [
    'CORe50'
]

if __name__ == "__main__":

    # this below can be taken as a usage example or a simple test script
    import sys
    from torch.utils.data.dataloader import DataLoader

    scenario = CORe50(scenario="nicv2_79",
                      train_transform=ToTensor(),
                      eval_transform=ToTensor(), mini=True)
    for i, batch in enumerate(scenario.train_stream):
        print(i, batch)
        dataset, t = batch.dataset, batch.task_label
        dl = DataLoader(dataset, batch_size=300)

        for mb in dl:
            x, y, t = mb
            print(x.shape)
            print(y.shape)
        sys.exit(0)
