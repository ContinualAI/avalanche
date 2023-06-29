################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-09-2021                                                             #
# Author: Lorenzo Pellegrini, Antonio Carta                                    #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from pathlib import Path

import json
import os


AVALANCHE_CONFIG_ROOT = os.path.expanduser("~/.avalanche")
AVALANCHE_CONFIG_FILENAME = os.path.expanduser("~/.avalanche/config.json")


def set_dataset_root(root_dir: str):
    """Set the default dataset root directory.

    :param root_dir: The new root directory.
    """
    AVALANCHE_CONFIG["dataset_location"] = root_dir


def default_dataset_location(dataset_name: str) -> Path:
    """Returns the default download location for Avalanche datasets.

    The default value is "~/.avalanche/data/<dataset_name>", but it may be
    changed via the `dataset_location` value in the configuration file
    in `~/.avalanche/config.json`.

    :param dataset_name: The name of the dataset. Must be a string that
        can be used to name a directory in most filesystems!
    :return: The default path for the dataset.
    """
    base_dir = os.path.expanduser(AVALANCHE_CONFIG["dataset_location"])
    return Path(f"{base_dir}/{dataset_name}")


def load_config_file():
    with open(AVALANCHE_CONFIG_FILENAME, "r") as f:
        return json.load(f)


def maybe_init_config_file():
    """Initialize Avalanche user's config file, if it does not exists yet.

    The file is located in `~/.avalanche/config.json`
    """
    if os.path.exists(AVALANCHE_CONFIG_FILENAME):
        return
    os.makedirs(AVALANCHE_CONFIG_ROOT, exist_ok=True)
    default_config = {"dataset_location": os.path.expanduser("~/.avalanche/data")}

    with open(AVALANCHE_CONFIG_FILENAME, "w") as f:
        json.dump(default_config, f, indent=4)


maybe_init_config_file()
AVALANCHE_CONFIG = load_config_file()


__all__ = ["default_dataset_location"]
