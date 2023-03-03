"""
The :py:mod:`training` module provides a generic continual learning training
class (:py:class:`BaseTemplate`) and implementations of the most common
CL strategies. These are provided either as standalone strategies in
:py:mod:`training.supervised` or as plugins (:py:mod:`training.plugins`) that
can be easily combined with your own strategy.
"""
from .regularization import *
from .supervised import *
from .storage_policy import *
from .losses import *
