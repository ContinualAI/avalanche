from typing import Sequence, Optional, Union

import torch

from avalanche.benchmarks import Experience
from avalanche.training.plugins import EvaluationPlugin, StrategyPlugin
from avalanche.training.plugins.clock import Clock
from avalanche.training.plugins.evaluation import default_logger


class BaseStrategy:
    """Base class for continual learning skeletons.

    **Training loop**
    The training loop is organized as follows::

        train
            train_exp  # for each experience

    **Evaluation loop**
    The evaluation loop is organized as follows::

        eval
            eval_exp  # for each experience

    """
    def __init__(
            self):
        """Init."""
        pass
