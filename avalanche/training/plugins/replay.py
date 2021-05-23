import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
from numpy import inf
from torch import Tensor, cat
from torch.nn import Module
from torch.utils.data import random_split, DataLoader

from avalanche.benchmarks.utils import AvalancheConcatDataset, \
    AvalancheDataset, AvalancheSubset
from avalanche.benchmarks.utils.data_loader import \
    ReplayDataLoader
from avalanche.models import FeatureExtractorBackbone
from avalanche.training.plugins.strategy_plugin import StrategyPlugin


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


class StoragePolicy(ABC):
    """
    A policy to store exemplars in a replay memory.

    :param ext_mem: The replay memory dictionary to store samples.
    :param mem_size: max number of total input samples in the replay memory.
    """
    def __init__(self, ext_mem: Dict[int, AvalancheDataset], mem_size: int):
        self.ext_mem = ext_mem
        self.mem_size = mem_size

    @abstractmethod
    def __call__(self, data_source: AvalancheDataset, **kwargs):
        """Store exemplars in the replay memory"""


class ExperienceBalancedStoragePolicy(StoragePolicy):
    def __init__(self, ext_mem: Dict, mem_size: int, adaptive_size: bool = True,
                 num_experiences=-1):
        """
        Stores samples for replay, equally divided over experiences.
        Because it is conditioned on the experience, it should be called in
        the 'after_training_exp' phase.

        The number of experiences can be fixed up front or adaptive, based on
        the 'adaptive_size' attribute. When adaptive, the memory is equally
        divided over all the unique observed experiences so far.

        :param ext_mem: The replay memory dictionary to store samples.
        :param mem_size: max number of total input samples in the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(ext_mem, mem_size)
        self.adaptive_size = adaptive_size
        self.num_experiences = num_experiences

        if not self.adaptive_size:
            assert self.num_experiences > 0, \
                """When fixed exp mem size, num_experiences should be > 0."""

    def subsample_single(self, data: AvalancheDataset, new_size: int):
        """ Subsample `data` to match length `new_size`. """
        removed_els = len(data) - new_size
        if removed_els > 0:
            data, _ = random_split(data, [new_size, removed_els])
        return data

    def subsample_all_groups(self, new_size: int):
        """ Subsample all groups equally to match total buffer size
        `new_size`. """
        groups = list(self.ext_mem.keys())
        if len(groups) == 0:
            return  # buffer is empty.

        num_groups = len(groups) if self.adaptive_size else self.num_experiences
        group_size = new_size // num_groups
        last_group_size = group_size + (new_size % num_groups)

        for g in groups[:-1]:
            self.ext_mem[g] = self.subsample_single(self.ext_mem[g], group_size)
        # last group may be bigger
        last = self.ext_mem[groups[-1]]
        self.ext_mem[groups[-1]] = self.subsample_single(last, last_group_size)

    def __call__(self, strategy: "BaseStrategy", **kwargs):
        num_exps = strategy.training_exp_counter + 1
        num_exps = num_exps if self.adaptive_size else self.num_experiences
        curr_data = strategy.experience.dataset

        # new group may be bigger because of the remainder.
        group_size = self.mem_size // num_exps
        new_group_size = group_size + (self.mem_size % num_exps)

        self.subsample_all_groups(group_size * (num_exps - 1))
        curr_data = self.subsample_single(curr_data, new_group_size)
        self.ext_mem[strategy.training_exp_counter + 1] = curr_data

        # buffer size should always equal self.mem_size
        len_tot = sum(len(el) for el in self.ext_mem.values())
        assert len_tot == self.mem_size


class ClassBalancedStoragePolicy(StoragePolicy):
    def __init__(self, ext_mem: Dict, mem_size: int, adaptive_size: bool = True,
                 total_num_classes: int = -1, selection_strategy:
                 Optional["ClassExemplarsSelectionStrategy"] = None):
        """
        Stores samples for replay, equally divided over classes.
        It should be called in the 'after_training_exp' phase (see
        ExperienceBalancedStoragePolicy).
        The number of classes can be fixed up front or adaptive, based on
        the 'adaptive_size' attribute. When adaptive, the memory is equally
        divided over all the unique observed classes so far.
        :param ext_mem: The replay memory dictionary to store samples in.
        :param mem_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        super().__init__(ext_mem, mem_size)
        self.selection_strategy = selection_strategy or \
            RandomExemplarsSelectionStrategy()
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

        if not self.adaptive_size:
            assert self.total_num_classes > 0, \
                """When fixed exp mem size, total_num_classes should be > 0."""

    def __call__(self, strategy: "BaseStrategy", **kwargs):
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
            if c in self.ext_mem:  # Merge data with previous seen data
                c_mem = AvalancheConcatDataset((c_mem, self.ext_mem[c]))
            sorted_indices = self.selection_strategy.make_sorted_indices(
                strategy, c_mem)
            self.ext_mem[c] = AvalancheSubset(c_mem, sorted_indices)

        # Distribute remaining samples using counts
        cutoff_per_exp = self.divide_remaining_samples(class_mem_size, div_cnt)

        # Use counts to remove samples from memory
        self.cutoff_memory(cutoff_per_exp)

    def divide_remaining_samples(self, exp_mem_size: int, div_cnt: int) -> \
            Dict[int, int]:
        # Find number of remaining samples
        samples_per_exp = {exp: len(mem) for exp, mem in
                           self.ext_mem.items()}
        rem_from_exps = {exp: exp_mem_size - memsize for exp, memsize in
                         samples_per_exp.items() if
                         exp_mem_size - memsize > 0}
        rem_from_div = self.mem_size % div_cnt
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

    def cutoff_memory(self, cutoff_per_exp: Dict[int, int]):
        # No need to reselect at this point, we expect the first selection to
        # have sorted the exemplars
        for exp, cutoff in cutoff_per_exp.items():
            self.ext_mem[exp] = AvalancheSubset(self.ext_mem[exp],
                                                list(range(cutoff)))


class ClassExemplarsSelectionStrategy(ABC):
    """
    Base class to define how to select class exemplars from a dataset
    """
    @abstractmethod
    def make_sorted_indices(self, strategy: "BaseStrategy",
                            data: AvalancheDataset) -> List[int]:
        """
        Should return the sorted list of indices to keep as exemplars.

        The last indices will be the first to be removed when cutoff memory.
        """


class RandomExemplarsSelectionStrategy(ClassExemplarsSelectionStrategy):
    """Select the exemplars at random in the dataset"""

    def make_sorted_indices(self, strategy: "BaseStrategy",
                            data: AvalancheDataset) -> List[int]:
        indices = list(range(len(data)))
        random.shuffle(indices)
        return indices


class FeatureBasedExemplarsSelectionStrategy(ClassExemplarsSelectionStrategy,
                                             ABC):
    """Base class to select exemplars from their features"""
    def __init__(self, model: Module, layer_name: str):
        self.feature_extractor = FeatureExtractorBackbone(model, layer_name)

    @torch.no_grad()
    def make_sorted_indices(self, strategy: "BaseStrategy",
                            data: AvalancheDataset) -> List[int]:
        self.feature_extractor.eval()
        features = cat(
            [
                self.feature_extractor(x.to(strategy.device))
                for x, *_ in DataLoader(data, batch_size=strategy.eval_mb_size)
            ]
        )
        return self.make_sorted_indices_from_features(features)

    @abstractmethod
    def make_sorted_indices_from_features(self, features: Tensor
                                          ) -> List[int]:
        """
        Should return the sorted list of indices to keep as exemplars.

        The last indices will be the first to be removed when cutoff memory.
        """


class HerdingSelectionStrategy(FeatureBasedExemplarsSelectionStrategy):
    def make_sorted_indices_from_features(self, features: Tensor
                                          ) -> List[int]:
        """
        The herding strategy as described in iCaRL

        It is a greedy algorithm, that select the remaining exemplar that get
        the center of already selected exemplars as close as possible as the
        center of all elements (in the feature space).
        """
        selected_indices = []

        center = features.mean(dim=0)
        current_center = center * 0

        for i in range(len(features)):
            # Compute distances with real center
            candidate_centers = current_center * i / (i + 1) + features / (i
                                                                           + 1)
            distances = pow(candidate_centers - center, 2).sum(dim=1)
            distances[selected_indices] = inf

            # Select best candidate
            new_index = distances.argmin().tolist()
            selected_indices.append(new_index)
            current_center = candidate_centers[new_index]

        return selected_indices


class ClosestToCenterSelectionStrategy(FeatureBasedExemplarsSelectionStrategy):
    def make_sorted_indices_from_features(self, features: Tensor
                                          ) -> List[int]:
        """
        A greedy algorithm that select the remaining exemplar that is the
        closest to the center of all elements (in feature space)
        """
        center = features.mean(dim=0)
        distances = pow(features - center, 2).sum(dim=1)
        return distances.argsort()
