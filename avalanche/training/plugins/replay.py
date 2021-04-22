from abc import ABC, abstractmethod
from typing import Dict
import random
from torch.utils.data import random_split

from avalanche.benchmarks.utils import AvalancheConcatDataset, \
    AvalancheDataset, AvalancheSubset
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
        strategy.dataloader = MultiTaskJoinedBatchDataLoader(
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
    def __call__(self, data_source, **kwargs):
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

        # Distribute remaining samples using counts
        cutoff_per_exp = self.divide_remaining_samples(
            exp_mem_size, strategy.training_exp_counter + 1)

        # Use counts to remove samples from memory
        self.cutoff_memory(cutoff_per_exp)

    def divide_remaining_samples(self, exp_mem_size, div_cnt):
        # Find number of remaining samples
        samples_per_exp = {exp: len(mem) for exp, mem in self.ext_mem.items()}
        rem_from_exps = {exp: exp_mem_size - memsize for exp, memsize in
                         samples_per_exp.items() if exp_mem_size - memsize > 0}
        rem_from_div = self.mem_size % (div_cnt)
        free_mem = sum(rem_from_exps.values()) + rem_from_div

        # Divide the remaining samples randomly over the experiences
        cutoff_per_exp = {exp: min(exp_mem_size, len(m))
                          for exp, m in self.ext_mem.items()}

        # Find remaining data samples to divide
        rem_samples_exp = {exp: memsize - exp_mem_size for exp, memsize in
                           samples_per_exp.items()
                           if memsize - exp_mem_size > 0}

        while len(rem_samples_exp) > 0 and free_mem > 0:
            exp = random.choice(list(rem_samples_exp.keys()))
            cutoff_per_exp[exp] += 1
            free_mem -= 1
            rem_samples_exp[exp] -= 1
            if rem_samples_exp[exp] <= 0:
                del rem_samples_exp[exp]

        return cutoff_per_exp

    def cutoff_memory(self, cutoff_per_exp):
        # Allocate to experiences
        for exp, cutoff in cutoff_per_exp.items():
            self.ext_mem[exp], _ = random_split(
                self.ext_mem[exp], [cutoff, len(self.ext_mem[exp]) - cutoff])


class ClassBalancedStoragePolicy(ExperienceBalancedStoragePolicy):
    def __init__(self, replay_mem: Dict, mem_size: int, adaptive_size=True,
                 total_num_classes=-1):
        """
        Stores samples for replay, equally divided over classes.
        It should be called in the 'after_training_exp' phase (see
        ExperienceBalancedStoragePolicy).

        The number of classes can be fixed up front or adaptive, based on
        the 'adaptive_size' attribute. When adaptive, the memory is equally
        divided over all the unique observed classes so far.

        :param replay_mem: The replay memory dictionary to store samples in.
        :param mem_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        self.ext_mem = replay_mem
        self.mem_size = mem_size
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

        if not self.adaptive_size:
            assert self.total_num_classes > 0, \
                """When fixed exp mem size, total_num_classes should be > 0."""

    def __call__(self, strategy, **kwargs):
        new_data = strategy.experience.dataset

        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = AvalancheSubset(new_data, indices=c_idxs)

        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # how many experiences to divide the memory over
        div_cnt = len(self.seen_classes) if self.adaptive_size \
            else self.total_num_classes
        class_mem_size = self.mem_size // div_cnt

        # Add current classes data to memory
        for c, c_mem in cl_datasets.items():
            if c not in self.ext_mem:
                self.ext_mem[c] = c_mem
            else:  # Merge data with previous seen data
                self.ext_mem[c] = AvalancheConcatDataset(
                    [c_mem, self.ext_mem[c]])

        # Distribute remaining samples using counts
        cutoff_per_exp = self.divide_remaining_samples(class_mem_size, div_cnt)

        # Use counts to remove samples from memory
        self.cutoff_memory(cutoff_per_exp)
