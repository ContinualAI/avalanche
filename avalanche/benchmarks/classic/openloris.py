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

""" This module contains the high-level OpenLORIS scenario/factor generator.
It basically returns a iterable scenario object ``GenericCLScenario`` given
a number of configuration parameters."""

from avalanche.benchmarks.datasets.openloris.openloris_data import \
    OPENLORIS_DATA
from avalanche.benchmarks.scenarios.generic_benchmark_creation import \
    create_generic_benchmark_from_filelists
from os.path import expanduser

nbatch = {
    'clutter': 9,
    'illumination': 9,
    'occlusion': 9,
    'pixel': 9,
    'mixture-iros': 12
}

fac2dirs = {
    'clutter': "batches_filelists/domain/clutter",
    'illumination': "batches_filelists/domain/illumination",
    'occlusion': "batches_filelists/domain/occlusion",
    'pixel': "batches_filelists/domain/pixel",
    'mixture-iros': "batches_filelists/domain/iros"
}


def OpenLORIS(root=expanduser("~") + "/.avalanche/data/openloris/",
              factor="clutter",
              train_transform=None,
              eval_transform=None):
    """
    Creates a CL scenario for OpenLORIS.

    If the dataset is not present in the computer, **this method will NOT be
    able automatically download** and store it.

    This generator can be used to obtain scenarios based on different "factors".
    Valid factors include 'clutter', 'illumination', 'occlusion', 'pixel', or
    'mixture-iros'.

    The scenario instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The task label "0" will be assigned to each experience.

    The scenario API is quite simple and is uniform across all scenario
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param root: Base path where OpenLORIS data is stored.
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

    :returns: a properly initialized :class:`GenericCLScenario` instance.
    """

    assert (factor in nbatch.keys()), "The selected factor is note " \
                                      "recognized: it should be 'clutter'," \
                                      "'illumination', 'occlusion', " \
                                      "'pixel', or 'mixture-iros'."
    if root is None:
        data = OPENLORIS_DATA()
    else:
        data = OPENLORIS_DATA(root)

    root = data.data_folder
    root_img = root

    filelists_bp = fac2dirs[factor] + "/"
    train_failists_paths = []
    for i in range(nbatch[factor]):
        train_failists_paths.append(
            root + filelists_bp + "train_batch_" +
            str(i).zfill(2) + ".txt")

    factor_obj = create_generic_benchmark_from_filelists(
        root_img, train_failists_paths,
        [root + filelists_bp + "test.txt"],
        task_labels=[0 for _ in range(nbatch[factor])],
        complete_test_set_only=True,
        train_transform=train_transform,
        eval_transform=eval_transform)

    return factor_obj


__all__ = [
    'OpenLORIS'
]

if __name__ == "__main__":

    # this below can be taken as a usage example or a simple test script
    import sys
    from torch.utils.data.dataloader import DataLoader

    factor = OpenLORIS(factor="clutter")
    for i, batch in enumerate(factor.train_stream):
        print(i, batch)
        dataset, t = batch.dataset, batch.task_label
        dl = DataLoader(dataset, batch_size=128)

        for mb in dl:
            x, y = mb
            print(x.shape)
            print(y.shape)
        sys.exit(0)
