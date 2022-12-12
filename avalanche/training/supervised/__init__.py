"""High-level strategies.

This module contains
"""
from .joint_training import *
from .ar1 import AR1
from .cumulative import Cumulative
from .strategy_wrappers import *
from .strategy_wrappers_online import *
from .deep_slda import *
from .icarl import ICaRL
from .er_ace import ER_ACE, OnlineER_ACE
