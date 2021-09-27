import copy
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

import torch
import tqdm
from torch import Tensor
from torch.utils.data import TensorDataset

from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.storage_policy import ClassBalancedStoragePolicy

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class GDumbPlugin(StrategyPlugin):
    """ GDumb plugin.

    At each experience the model is trained  from scratch using a buffer of
    samples collected from all the previous learning experiences.
    The buffer is updated at the start of each experience to add new classes or
    new examples of already encountered classes.
    In multitask scenarios, mem_size is the memory size for each task.
    This plugin can be combined with a Naive strategy to obtain the
    standard GDumb strategy.
    https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf
    """

    def __init__(self, mem_size: int = 200):
        super().__init__()
        self.mem_size = mem_size

        # model initialization
        self.buffer = {}
        self.storage_policy = ClassBalancedStoragePolicy(
            ext_mem=self.buffer,
            mem_size=self.mem_size,
            adaptive_size=True
        )
        self.init_model = None

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy', **kwargs):
        """ Reset model. """
        if self.init_model is None:
            self.init_model = copy.deepcopy(strategy.model)
        else:
            strategy.model = copy.deepcopy(self.init_model)

    def after_train_dataset_adaptation(self, strategy: "BaseStrategy",
                                       **kwargs):
        self.storage_policy(strategy, **kwargs)
        cat_data = AvalancheConcatDataset(self.buffer.values())
        strategy.adapted_dataset = cat_data
