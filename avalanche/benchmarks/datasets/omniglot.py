################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 13-02-2021                                                             #
# Author(s): Jary Pomponi                                                      #
################################################################################

from os.path import join
from typing import Optional, Callable

from torchvision.datasets import Omniglot as OmniglotTorch


class Omniglot(OmniglotTorch):
    """
    Custom class used to adapt Omniglot (from Torchvision) and make it
    compatible with the Avalanche API.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            join(root, self.folder),
            download=download,
            transform=transform,
            target_transform=target_transform,
            background=train,
        )

        self.targets = [x[1] for x in self._flat_character_images]

    @property
    def data(self):
        return [x for x, _ in self]


__all__ = ["Omniglot"]
