################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-12-2020                                                             #
# Author(s): ContinualAI                                                       #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" This module contains the high-level OpenLORIS benchmark/factor generator.
It basically returns a iterable benchmark object ``GenericCLScenario`` given
a number of configuration parameters."""

from pathlib import Path
from typing import Union, Any, Optional, Literal

from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.datasets.openloris import (
    OpenLORIS as OpenLORISDataset,
)
from avalanche.benchmarks.scenarios.deprecated.generic_benchmark_creation import (
    create_generic_benchmark_from_filelists,
)


nbatch = {
    "clutter": 9,
    "illumination": 9,
    "occlusion": 9,
    "pixel": 9,
    "mixture-iros": 12,
}

fac2dirs = {
    "clutter": "batches_filelists/domain/clutter",
    "illumination": "batches_filelists/domain/illumination",
    "occlusion": "batches_filelists/domain/occlusion",
    "pixel": "batches_filelists/domain/pixel",
    "mixture-iros": "batches_filelists/domain/iros",
}


def OpenLORIS(
    *,
    factor: Literal[
        "clutter", "illumination", "occlusion", "pixel", "mixture-iros"
    ] = "clutter",
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    dataset_root: Optional[Union[str, Path]] = None
):
    """
    Creates a CL benchmark for OpenLORIS.

    If the dataset is not present in the computer, **this method will NOT be
    able automatically download** and store it.

    This generator can be used to obtain scenarios based on different "factors".
    Valid factors include 'clutter', 'illumination', 'occlusion', 'pixel', or
    'mixture-iros'.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The task label 0 will be assigned to each experience.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param factor: OpenLORIS main factors, indicating different environmental
        variations. It can be chosen between 'clutter', 'illumination',
        'occlusion', 'pixel', or 'mixture-iros'. The first three factors are
        included in the ICRA 2020 paper and the last factor (mixture-iros) is
        the benchmark setting for IROS 2019 Lifelong robotic vision competition.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        'openloris' will be used.

    :returns: a properly initialized :class:`GenericCLScenario` instance.
    """

    assert factor in nbatch.keys(), (
        "The selected factor is note "
        "recognized: it should be 'clutter',"
        "'illumination', 'occlusion', "
        "'pixel', or 'mixture-iros'."
    )

    # Dataset created just to download it
    dataset = OpenLORISDataset(dataset_root, download=True)

    # Use the root produced by the dataset implementation
    dataset_root = dataset.root

    filelists_bp = fac2dirs[factor] + "/"
    train_failists_paths = []
    for i in range(nbatch[factor]):
        train_failists_paths.append(
            dataset_root / filelists_bp / ("train_batch_" + str(i).zfill(2) + ".txt")
        )

    factor_obj = create_generic_benchmark_from_filelists(
        dataset_root,
        train_failists_paths,
        [dataset_root / filelists_bp / "test.txt"],
        task_labels=[0 for _ in range(nbatch[factor])],
        complete_test_set_only=True,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    return factor_obj


__all__ = ["OpenLORIS"]

if __name__ == "__main__":
    import sys

    # Untested!
    benchmark_instance = OpenLORIS()
    check_vision_benchmark(benchmark_instance)
    sys.exit(0)
