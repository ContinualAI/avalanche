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
from .er_ace import ER_ACE
from .er_aml import ER_AML
from .der import DER
from .l2p import LearningToPrompt
from .supervised_contrastive_replay import SCR
from .expert_gate import ExpertGateStrategy
from .mer import MER
from .feature_replay import *
