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

""" This module conveniently wraps Pytorch Datasets building utils for using a
clean and comprehensive Avalanche API."""

from torchvision.datasets import ImageFolder as torchImageFolder  # type: ignore
from torchvision.datasets import DatasetFolder as torchDatasetFolder


def ImageFolder(*args, **kwargs):
    return torchImageFolder(*args, **kwargs)


def DatasetFolder(*args, **kwargs):
    return torchDatasetFolder(*args, **kwargs)


if __name__ == "__main__":
    mnist = DatasetFolder(".", download=True)


__all__ = ["ImageFolder", "DatasetFolder"]
