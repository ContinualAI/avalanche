"""
Strategies define the basic train/eval loops. A strategy can be used as is or
customized by adding plugins, which may be used to combine multiple basic
strategies (e.g., EWC + replay).
"""
from .joint_training import *
from .ar1 import AR1
from .cumulative import Cumulative
from .strategy_wrappers import *
from .deep_slda import *
from .icarl import ICaRL
