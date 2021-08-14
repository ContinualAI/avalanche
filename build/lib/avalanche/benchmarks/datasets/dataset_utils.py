################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 25-05-2021                                                             #
# Author: Lorenzo Pellegrini                                                   #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from pathlib import Path


def get_default_dataset_location(dataset_name: str) -> Path:
    """
    Return the default location of a dataset.

    This currently returns "~/.avalanche/data/<dataset_name>" but in the future
    an environment variable bay be introduced to change the root path.

    :param dataset_name: The name of the dataset. Consider using a string that
        can be used to name a directory in most filesystems!
    :return: The default path for the dataset.
    """
    return Path.home() / f".avalanche/data/{dataset_name}"


__all__ = [
    'get_default_dataset_location'
]
