################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-04-2022                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""Reinforcement Learning scenario definitions."""
from typing import Union, Sequence

try:
    from gym import Env
except ImportError:
    # empty class to make sure everything below works without changes
    class Env:
        pass


RLStreamDataOrigin = Union[Env, Sequence[Env]]
