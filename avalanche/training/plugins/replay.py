from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Dict, TYPE_CHECKING

from torch.utils.data import random_split, ConcatDataset

from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils.data_loader import \
    MultiTaskJoinedBatchDataLoader


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
    """

    def __init__(self, mem_size=200):
        super().__init__()

        self.mem_size = mem_size
        self.ext_mem = {}  # a Dict<task_id, Dataset>
        self.storage_policy = RandomStoragePolicy(replay_mem=self.ext_mem,
                                                  experience_balanced=True)

    def before_training_exp(self, strategy, num_workers=0, shuffle=True,
                            **kwargs):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.ext_mem) == 0:
            return
        strategy.current_dataloader = MultiTaskJoinedBatchDataLoader(
            strategy.adapted_dataset,
            AvalancheConcatDataset(self.ext_mem.values()),
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy, **kwargs):
        """ After training we update the external memory with the patterns of
         the current training batch/task, adding the patterns from the new
         experience and removing those from past experiences to comply the limit
         of the total number of patterns in memory """
        self.storage_policy(strategy)


class StoragePolicy(ABC):
    """ Any policy to store exemplars in a replay memory."""

    @abstractmethod
    def __call__(self, replay_mem: Dict, **kwargs):
        """Store exemplars in the replay memory"""
        pass


class ExperienceBalancedStoragePolicy(StoragePolicy):

    def __init__(self,
                 replay_mem: Dict, mem_size: int,
                 class_balanced=False,
                 ):
        self.replay_mem = replay_mem
        """
        The replay memory as a Dict. Keys are the conditionals (experience/class), values are the samples stored.
        """

        assert not (experience_balanced and class_balanced), \
            "Can't be both experience and class balanced."
        self.experience_balanced = experience_balanced
        """
        Each experience gets the same amount of space in the replay memory.
        """

        self.class_balanced = class_balanced
        """
        Each class label gets the same amount of space in the replay memory.
        """

        self.allocate_size = allocate_size
        """
        None = adaptive:    Memory capacity is divided over all keys in the replay memory.
                            This exploits all the memory capacity over the experiences/classes (keys).
        int = fixed:        All keys in the replay memory get a fixed memory capacity. 
                            Useful when experiences/classes known beforehand.
        """
        self.adaptive_size = self.allocate_size < 0
        """
        Memory capacity is divided over all keys in the replay memory.
        """

        self.fixed_size = self.allocate_size > 0
        """
        All keys in the replay memory get a fixed memory capacity. 
        Useful when experiences/classes known beforehand.
        """

    def __call__(self, strategy, **kwargs):
        """
        :param strategy:
        :param kwargs:
        :return:
        """
        curr_task_id = strategy.experience.task_label
        curr_data = strategy.experience.dataset

        # Additional set of the current batch to be concatenated
        #  to the external memory.
        rm_add = None

        # how many patterns to save for next iter
        single_task_mem_size = min(self.mem_size, len(curr_data))
        h = single_task_mem_size // (strategy.training_exp_counter + 1)

        remaining_example = single_task_mem_size % (
                strategy.training_exp_counter + 1)
        # We recover it using the random_split method and getting rid of the
        # second split.
        rm_add, _ = random_split(
            curr_data, [h, len(curr_data) - h]
        )
        # replace patterns randomly in memory
        ext_mem = self.ext_mem
        if curr_task_id not in ext_mem:
            ext_mem[curr_task_id] = rm_add
        else:
            rem_len = len(ext_mem[curr_task_id]) - len(rm_add)
            _, saved_part = random_split(ext_mem[curr_task_id],
                                         [len(rm_add), rem_len]
                                         )
            ext_mem[curr_task_id] = AvalancheConcatDataset(
                [saved_part, rm_add])

        # remove exceeding patterns, the amount of pattern kept is such that the
        # sum is equal to mem_size and that the number of patterns between the
        # tasks is balanced
        for task_id in ext_mem.keys():
            current_mem_size = h if remaining_example <= 0 else h + 1
            remaining_example -= 1

            if (current_mem_size < len(ext_mem[task_id]) and
                    task_id != curr_task_id):
                rem_len = len(ext_mem[task_id]) - current_mem_size
                _, saved_part = random_split(
                    ext_mem[task_id],
                    [rem_len, current_mem_size])
                ext_mem[task_id] = saved_part
        self.ext_mem = ext_mem

class ClassBalancedStoragePolicy(StoragePolicy):
    pass