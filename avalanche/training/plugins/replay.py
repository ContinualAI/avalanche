from typing import Optional, TYPE_CHECKING

from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import \
    ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.storage_policy import StoragePolicy, ExperienceBalancedStoragePolicy

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class ReplayPlugin(StrategyPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks. 
    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced 
    such that there are the same number of examples for each experience.    
    
    The `after_training_exp` callback is implemented in order to add new 
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored 
    in the external memory.
    :param storage_policy: The policy that controls how to add new exemplars
                           in memory
    """

    def __init__(self, mem_size: int = 200,
                 storage_policy: Optional["StoragePolicy"] = None):
        super().__init__()
        self.mem_size = mem_size

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            self.ext_mem = storage_policy.ext_mem  # Keep ref
            assert storage_policy.mem_size == self.mem_size

        else:  # Default
            self.ext_mem = {}  # a Dict<task_id, Dataset>
            self.storage_policy = ExperienceBalancedStoragePolicy(
                ext_mem=self.ext_mem,
                mem_size=self.mem_size,
                adaptive_size=True)

    def before_training_exp(self, strategy: "BaseStrategy",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.ext_mem) == 0:
            return
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            AvalancheConcatDataset(self.ext_mem.values()),
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        self.storage_policy(strategy, **kwargs)


