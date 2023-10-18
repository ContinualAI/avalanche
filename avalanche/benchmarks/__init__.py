"""
The :py:mod:`benchmarks` module provides a set of utilities that can be used for
handling and generating your continual learning data stream. In the
:py:mod:`datasets` module, basic PyTorch Datasets are provided. In the
:py:mod:`classic` module instead, classic benchmarks (already proposed in the
CL literature) generated from the datasets are provided. Finally,
in :py:mod:`generators` basic utilities to generate new benchmarks on-the-fly
are made available.
"""

from .scenarios import *
from .scenarios.deprecated.generators import *
from .classic import *
