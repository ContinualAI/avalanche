################################################################################
# Copyright (c) 2020 ContinualAI                                               #
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

from avalanche.benchmarks.datasets.core50.core50_data import CORE50_DATA
from avalanche.benchmarks.scenarios.generic_scenario_creation import \
    create_generic_scenario_from_filelists
from os.path import expanduser

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


def CORe50(root=expanduser("~") + "/.avalanche/data/core50/",
           scenario="nicv2_391", run=0):
    """ CORe50 continual scenario generator

    :param root: Path indicating where to store the dataset and related
        metadata. By default they will be stored in
        avalanche/datasets/core50/data/.
    :param scenario: CORe50 main scanario. I can be chosen between 'ni', 'nc',
        'nic', 'nicv2_79', 'nicv2_196' or 'nicv2_391.'
    :param run: number of run for the scenario. Batch ordering change based
        on this parameter (a number between 0 and 9).

    :returns: it returns a :class:`GenericCLScenario` instance that can be
        iterated.
    """

    assert (0 <= run <= 9), "Pre-defined run of CORe50 are only 10. Indicate " \
                            "a number between 0 and 9."
    assert (scenario in nbatch.keys()), "The selected scenario is note " \
                                        "recognized: it should be 'ni', 'nc'," \
                                        "'nic', 'nicv2_79', 'nicv2_196' or " \
                                        "'nicv2_391'."
    if root is None:
        core_data = CORE50_DATA()
    else:
        core_data = CORE50_DATA(root)

    root = core_data.data_folder
    root_img = root + "core50_128x128/"

    filelists_bp = scen2dirs[scenario] + "run" + str(run) + "/"
    train_failists_paths = []
    for i in range(nbatch[scenario]):
        train_failists_paths.append(
            root + filelists_bp + "train_batch_" +
            str(i).zfill(2) + "_filelist.txt")

    scenario_obj = create_generic_scenario_from_filelists(
        root_img, train_failists_paths,
        root + filelists_bp + "test_filelist.txt",
        [0 for _ in range(nbatch[scenario])],
        complete_test_set_only=True,
        train_transform=transforms.ToTensor(),
        test_transform=transforms.ToTensor())

    return scenario_obj


__all__ = [
    'CORe50'
]

if __name__ == "__main__":

    # this below can be taken as a usage example or a simple test script
    import sys
    from torch.utils.data.dataloader import DataLoader
    from torchvision import transforms

    scenario = CORe50(scenario="nicv2_79")
    for i, batch in enumerate(scenario.train_stream):
        print(i, batch)
        dataset, t = batch.dataset, batch.task_label
        dl = DataLoader(dataset, batch_size=300)

        for mb in dl:
            x, y = mb
            print(x.shape)
            print(y.shape)
        sys.exit(0)
