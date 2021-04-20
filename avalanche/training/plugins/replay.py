from abc import ABC, abstractmethod
from typing import Dict
import random
from torch.utils.data import random_split

from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
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

    def __init__(self, mem_size=200, storage_policy=None):
        super().__init__()

        self.mem_size = mem_size
        self.ext_mem = {}  # a Dict<task_id, Dataset>

        if storage_policy is not None:
            self.storage_policy = storage_policy
        else:  # Default
            self.storage_policy = ExperienceBalancedStoragePolicy(
                replay_mem=self.ext_mem,
                mem_size=self.mem_size,
                adaptive_size=True)

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
        self.storage_policy(strategy)


class StoragePolicy(ABC):
    """ A policy to store exemplars in a replay memory."""

    @abstractmethod
    def __call__(self, replay_mem: Dict, **kwargs):
        """Store exemplars in the replay memory"""
        pass


class ExperienceBalancedStoragePolicy(StoragePolicy):

    def __init__(self, replay_mem: Dict, mem_size: int, adaptive_size=True,
                 num_experiences=-1):
        """
        Stores samples for replay, equally divided over experiences.
        Because it is conditioned on the experience, it should be called in
        the 'after_training_exp' phase.

        The number of experiences can be fixed up front or adaptive, based on
        the 'adaptive_size' attribute. When adaptive, the memory is equally
        divided over all the unique observed experiences so far.

        :param replay_mem: The replay memory dictionary to store samples.
        :param mem_size: max number of total input samples in the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        self.ext_mem = replay_mem
        self.mem_size = mem_size
        self.adaptive_size = adaptive_size
        self.num_experiences = num_experiences

        if not self.adaptive_size:
            assert self.num_experiences > 0, \
                """When fixed exp mem size, num_experiences should be > 0."""

    def __call__(self, strategy, **kwargs):
        curr_task_id = strategy.experience.task_label
        curr_data = strategy.experience.dataset

        # how many experiences to divide the memory over
        if self.adaptive_size:  # Observed number of experiences
            exp_cnt = (strategy.training_exp_counter + 1)
        else:  # Fixed number of experiences
            exp_cnt = self.num_experiences

        exp_mem_size = self.mem_size // exp_cnt

        # Add current experience data to memory
        if curr_task_id not in self.ext_mem:
            self.ext_mem[curr_task_id] = AvalancheDataset(curr_data)
        else:  # Merge data with previous seen data
            self.ext_mem[curr_task_id] = AvalancheConcatDataset(
                [curr_data, self.ext_mem[curr_task_id]])

        # Get data from current experience
        # We recover it using the random_split method and getting rid of the
        # second split.
        # rm_add_size = min(exp_mem_size, len(curr_data))
        # self.ext_mem[curr_task_id], _ = random_split(
        #     self.ext_mem[curr_task_id],
        #     [rm_add_size, len(self.ext_mem[curr_task_id]) - rm_add_size]
        # )

        # Find number of remaining samples
        samples_per_exp = {exp: len(m) for exp, m in self.ext_mem.items()}
        remaining_from_exps = {exp: exp_mem_size - size for exp, size in
                               samples_per_exp.items()
                               if exp_mem_size - size > 0}
        remaining_from_div = self.mem_size % (strategy.training_exp_counter + 1)
        remaining_examples = sum(remaining_from_exps.values()) + \
                             remaining_from_div

        # Divide the remaining samples randomly over the experiences
        cutoff_per_exp = {exp: min(exp_mem_size, len(m))
                          for exp, m in self.ext_mem.items()}
        while len(remaining_from_exps) > 0 and remaining_examples > 0:
            exps_samples_left = list(remaining_from_exps.values())
            exp = exps_samples_left[random.randrange(len(exps_samples_left))]
            cutoff_per_exp[exp] += 1
            remaining_examples -= 1
            remaining_from_exps[exp] -= 1
            if remaining_from_exps[exp] <= 0:
                del remaining_from_exps[exp]

        # Allocate to experiences
        for exp, cutoff in cutoff_per_exp.items():
            self.ext_mem[exp], _ = random_split(
                self.ext_mem[exp], [cutoff, len(self.ext_mem[exp]) - cutoff])


class ClassBalancedStoragePolicy(StoragePolicy):
    def __init__(self, replay_mem: Dict, mem_size: int, adaptive_size=True,
                 total_num_classes=-1):
        """

        :param replay_mem: The replay memory dictionary where samples will be stored.
        :param mem_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number of classes to divide capacity over.

        The :mem_size: attribute controls the total number of patterns to be stored
        in the external memory.
        """
        self.replay_mem = replay_mem
        self.mem_size = mem_size
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes

        if not self.adaptive_size:
            assert self.total_num_classes > 0, """When no adaptive size, total_num_classes should be > 0."""

    def __call__(self, strategy, **kwargs):
        curr_task_id = strategy.experience.task_label
        curr_data = strategy.experience.dataset

        # Additional set of the current batch to be concatenated
        #  to the external memory.
        rm_add = None

        # how many patterns to save for next iter
        if self.adaptive_size:
            seen_classes = strategy.experience.classes_seen_so_far
            h = self.mem_size // seen_classes  # Observed number of classes
        else:
            assert len(
                self.replay_mem) <= self.total_num_classes, "Fixed total_num_classes cannot be exceeded!"
            h = self.mem_size // self.total_num_classes  # Fixed number of classes

        remaining_example = self.mem_size % (
                strategy.training_exp_counter + 1)
        # We recover it using the random_split method and getting rid of the
        # second split.
        curr_task_h = min(self.mem_size, len(curr_data))
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
