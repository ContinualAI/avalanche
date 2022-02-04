"""Templates define the train/eval loops logic.

Some CL scenarios need different loops, such as supervised CL and
reinforcement learning CL.
Templates define a "template" of the algorithm, which consists of:
- which attributes are used by the strategy
- training loop logic
- supported callbacks

Templates are the backbone that supports the plugin systems.
"""
from .base import BaseTemplate
from .base_sgd import BaseSGDTemplate
from .online_supervised import SupervisedOnlineTemplate
from .supervised import SupervisedTemplate
