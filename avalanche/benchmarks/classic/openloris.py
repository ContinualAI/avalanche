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
from avalanche.benchmarks.scenarios.generic_scenario_creation import \
    create_generic_scenario_from_filelists
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
              factor="clutter"):
    """ OpenLORIS continual scenario generator

    :param root: Path indicating where to store the dataset and related
        metadata. By default they will be stored in
        avalanche/datasets/openloris/data/.
    :param factor: OpenLORIS main factors, indicating different environmental
        variations. It can be chosen between 'clutter', 'illumination',
        'occlusion', 'pixel', or 'mixture-iros'. The first three factors are
        included in the ICRA 2020 paper and the last factor (mixture-iros) is
        the benchmark setting for IROS 2019 Lifelong robotic vision competition.

    :returns: it returns a :class:`GenericCLScenario` instance that can be
        iterated.
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

    factor_obj = create_generic_scenario_from_filelists(
        root_img, train_failists_paths,
        root + filelists_bp + "test.txt",
        [0 for _ in range(nbatch[factor])],
        complete_test_set_only=True,
        train_transform=transforms.ToTensor(),
        test_transform=transforms.ToTensor())

    return factor_obj


__all__ = [
    'OpenLORIS'
]

if __name__ == "__main__":

    # this below can be taken as a usage example or a simple test script
    import sys
    from torch.utils.data.dataloader import DataLoader
    from torchvision import transforms

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
