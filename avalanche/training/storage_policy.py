import random
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, TYPE_CHECKING

import torch
from numpy import inf
from torch import cat, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils import (
    AvalancheDataset,
    AvalancheSubset,
    AvalancheConcatDataset,
)
from avalanche.models import FeatureExtractorBackbone

if TYPE_CHECKING:
    from .templates.supervised import SupervisedTemplate


class ExemplarsBuffer(ABC):
    """ABC for rehearsal buffers to store exemplars.

    `self.buffer` is an AvalancheDataset of samples collected from the previous
    experiences. The buffer can be updated by calling `self.update(strategy)`.
    """

    def __init__(self, max_size: int):
        """Init.

        :param max_size: max number of input samples in the replay memory.
        """
        self.max_size = max_size
        """ Maximum size of the buffer. """
        self._buffer = AvalancheConcatDataset([])

    @property
    def buffer(self) -> AvalancheDataset:
        """Buffer of samples."""
        return self._buffer

    @buffer.setter
    def buffer(self, new_buffer: AvalancheDataset):
        self._buffer = new_buffer

    @abstractmethod
    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update `self.buffer` using the `strategy` state.

        :param strategy:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def resize(self, strategy: "SupervisedTemplate", new_size: int):
        """Update the maximum size of the buffer.

        :param strategy:
        :param new_size:
        :return:
        """
        ...


class ReservoirSamplingBuffer(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update buffer."""
        self.update_from_dataset(strategy.experience.dataset)

    def update_from_dataset(self, new_data: AvalancheDataset):
        """Update the buffer using the given dataset.

        :param new_data:
        :return:
        """
        new_weights = torch.rand(len(new_data))

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        cat_data = AvalancheConcatDataset([new_data, self.buffer])
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer = AvalancheSubset(cat_data, buffer_idxs)
        self._buffer_weights = sorted_weights[: self.max_size]

    def resize(self, strategy, new_size):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = AvalancheSubset(self.buffer, torch.arange(self.max_size))
        self._buffer_weights = self._buffer_weights[: self.max_size]


class BalancedExemplarsBuffer(ExemplarsBuffer):
    """A buffer that stores exemplars for rehearsal in separate groups.

    The grouping allows to balance the data (by task, experience,
    classes..). In combination with balanced data loaders, it can be used
    to sample balanced mini-batches during training.

    `self.buffer_groups` is a dictionary that stores each group as a
    separate buffer. The buffers are updated by calling
    `self.update(strategy)`.
    """

    def __init__(
        self, max_size: int, adaptive_size: bool = True, total_num_groups=None
    ):
        """
        :param max_size: max number of input samples in the replay memory.
        :param adaptive_size: True if max_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param total_num_groups: If adaptive size is False, the fixed number
                                of groups to divide capacity over.
        """
        super().__init__(max_size)
        self.adaptive_size = adaptive_size
        self.total_num_groups = total_num_groups
        if not self.adaptive_size:
            assert self.total_num_groups > 0, (
                "You need to specify `total_num_groups` if "
                "`adaptive_size=True`."
            )
        else:
            assert self.total_num_groups is None, (
                "`total_num_groups` is not compatible with "
                "`adaptive_size=False`."
            )

        self.buffer_groups: Dict[int, ExemplarsBuffer] = {}
        """ Dictionary of buffers. """

    @property
    def buffer_datasets(self):
        """Return group buffers as a list of `AvalancheDataset`s."""
        return [g.buffer for g in self.buffer_groups.values()]

    def get_group_lengths(self, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        if self.adaptive_size:
            lengths = [self.max_size // num_groups for _ in range(num_groups)]
            # distribute remaining size among experiences.
            rem = self.max_size - sum(lengths)
            for i in range(rem):
                lengths[i] += 1
        else:
            lengths = [
                self.max_size // self.total_num_groups
                for _ in range(num_groups)
            ]
        return lengths

    @property
    def buffer(self):
        return AvalancheConcatDataset(
            [g.buffer for g in self.buffer_groups.values()]
        )

    @buffer.setter
    def buffer(self, new_buffer):
        assert NotImplementedError(
            "Cannot set `self.buffer` for this class. "
            "You should modify `self.buffer_groups instead."
        )

    @abstractmethod
    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update `self.buffer_groups` using the `strategy` state.

        :param strategy:
        :param kwargs:
        :return:
        """
        ...

    def resize(self, strategy, new_size):
        """Update the maximum size of the buffers."""
        self.max_size = new_size
        lens = self.get_group_lengths(len(self.buffer_groups))
        for ll, buffer in zip(lens, self.buffer_groups.values()):
            buffer.resize(strategy, ll)


class ExperienceBalancedBuffer(BalancedExemplarsBuffer):
    """Rehearsal buffer with samples balanced over experiences.

    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(
        self, max_size: int, adaptive_size: bool = True, num_experiences=None
    ):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset
        num_exps = strategy.clock.train_exp_counter + 1
        lens = self.get_group_lengths(num_exps)

        new_buffer = ReservoirSamplingBuffer(lens[-1])
        new_buffer.update_from_dataset(new_data)
        self.buffer_groups[num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)


class ClassBalancedBuffer(BalancedExemplarsBuffer):
    """Stores samples for replay, equally divided over classes.

    There is a separate buffer updated by reservoir sampling for each
        class.
    It should be called in the 'after_training_exp' phase (see
    ExperienceBalancedStoragePolicy).
    The number of classes can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed classes so far.
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: int = None,
    ):
        """
        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert (
                total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

    def update(self, strategy: "SupervisedTemplate", **kwargs):
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

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(
                strategy, class_to_len[class_id]
            )


class ParametricBuffer(BalancedExemplarsBuffer):
    """Stores samples for replay using a custom selection strategy and
    grouping."""

    def __init__(
        self,
        max_size: int,
        groupby=None,
        selection_strategy: Optional["ExemplarsSelectionStrategy"] = None,
    ):
        """
        :param max_size: The max capacity of the replay memory.
        :param groupby: Grouping mechanism. One of {None, 'class', 'task',
        'experience'}.
        :param selection_strategy: The strategy used to select exemplars to
                                   keep in memory when cutting it off.
        """
        super().__init__(max_size)
        assert groupby in {None, "task", "class", "experience"}, (
            "Unknown grouping scheme. Must be one of {None, 'task', "
            "'class', 'experience'}"
        )
        self.groupby = groupby
        ss = selection_strategy or RandomExemplarsSelectionStrategy()
        self.selection_strategy = ss
        self.seen_groups = set()
        self._curr_strategy = None

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset
        new_groups = self._make_groups(strategy, new_data)
        self.seen_groups.update(new_groups.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_groups))
        group_to_len = {}
        for group_id, ll in zip(self.seen_groups, lens):
            group_to_len[group_id] = ll

        # update buffers with new data
        for group_id, new_data_g in new_groups.items():
            ll = group_to_len[group_id]
            if group_id in self.buffer_groups:
                old_buffer_g = self.buffer_groups[group_id]
                old_buffer_g.update_from_dataset(strategy, new_data_g)
                old_buffer_g.resize(strategy, ll)
            else:
                new_buffer = _ParametricSingleBuffer(
                    ll, self.selection_strategy
                )
                new_buffer.update_from_dataset(strategy, new_data_g)
                self.buffer_groups[group_id] = new_buffer

        # resize buffers
        for group_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[group_id].resize(
                strategy, group_to_len[group_id]
            )

    def _make_groups(self, strategy, data):
        """Split the data by group according to `self.groupby`."""
        if self.groupby is None:
            return {0: data}
        elif self.groupby == "task":
            return self._split_by_task(data)
        elif self.groupby == "experience":
            return self._split_by_experience(strategy, data)
        elif self.groupby == "class":
            return self._split_by_class(data)
        else:
            assert False, "Invalid groupby key. Should never get here."

    def _split_by_class(self, data):
        # Get sample idxs per class
        class_idxs = {}
        for idx, target in enumerate(data.targets):
            if target not in class_idxs:
                class_idxs[target] = []
            class_idxs[target].append(idx)

        # Make AvalancheSubset per class
        new_groups = {}
        for c, c_idxs in class_idxs.items():
            new_groups[c] = AvalancheSubset(data, indices=c_idxs)
        return new_groups

    def _split_by_experience(self, strategy, data):
        exp_id = strategy.clock.train_exp_counter + 1
        return {exp_id: data}

    def _split_by_task(self, data):
        new_groups = {}
        for task_id in data.task_set:
            new_groups[task_id] = data.task_set[task_id]
        return new_groups


class _ParametricSingleBuffer(ExemplarsBuffer):
    """A buffer that stores samples for replay using a custom selection
    strategy.

    This is a private class. Use `ParametricBalancedBuffer` with
    `groupby=None` to get the same behavior.
    """

    def __init__(
        self,
        max_size: int,
        selection_strategy: Optional["ExemplarsSelectionStrategy"] = None,
    ):
        """
        :param max_size: The max capacity of the replay memory.
        :param selection_strategy: The strategy used to select exemplars to
                                   keep in memory when cutting it off.
        """
        super().__init__(max_size)
        ss = selection_strategy or RandomExemplarsSelectionStrategy()
        self.selection_strategy = ss
        self._curr_strategy = None

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset
        self.update_from_dataset(strategy, new_data)

    def update_from_dataset(self, strategy, new_data):
        self.buffer = AvalancheConcatDataset([self.buffer, new_data])
        self.resize(strategy, self.max_size)

    def resize(self, strategy, new_size: int):
        self.max_size = new_size
        idxs = self.selection_strategy.make_sorted_indices(
            strategy=strategy, data=self.buffer
        )
        self.buffer = AvalancheSubset(self.buffer, idxs[: self.max_size])


class ExemplarsSelectionStrategy(ABC):
    """
    Base class to define how to select a subset of exemplars from a dataset.
    """

    @abstractmethod
    def make_sorted_indices(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        """
        Should return the sorted list of indices to keep as exemplars.

        The last indices will be the first to be removed when cutoff memory.
        """
        ...


class RandomExemplarsSelectionStrategy(ExemplarsSelectionStrategy):
    """Select the exemplars at random in the dataset"""

    def make_sorted_indices(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        indices = list(range(len(data)))
        random.shuffle(indices)
        return indices


class FeatureBasedExemplarsSelectionStrategy(ExemplarsSelectionStrategy, ABC):
    """Base class to select exemplars from their features"""

    def __init__(self, model: Module, layer_name: str):
        self.feature_extractor = FeatureExtractorBackbone(model, layer_name)

    @torch.no_grad()
    def make_sorted_indices(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        self.feature_extractor.eval()
        features = cat(
            [
                self.feature_extractor(x.to(strategy.device))
                for x, *_ in DataLoader(data, batch_size=strategy.eval_mb_size)
            ]
        )
        return self.make_sorted_indices_from_features(features)

    @abstractmethod
    def make_sorted_indices_from_features(self, features: Tensor) -> List[int]:
        """
        Should return the sorted list of indices to keep as exemplars.

        The last indices will be the first to be removed when cutoff memory.
        """


class HerdingSelectionStrategy(FeatureBasedExemplarsSelectionStrategy):
    """The herding strategy as described in iCaRL.

    It is a greedy algorithm, that select the remaining exemplar that get
    the center of already selected exemplars as close as possible as the
    center of all elements (in the feature space).
    """

    def make_sorted_indices_from_features(self, features: Tensor) -> List[int]:
        selected_indices = []

        center = features.mean(dim=0)
        current_center = center * 0

        for i in range(len(features)):
            # Compute distances with real center
            candidate_centers = current_center * i / (i + 1) + features / (
                i + 1
            )
            distances = pow(candidate_centers - center, 2).sum(dim=1)
            distances[selected_indices] = inf

            # Select best candidate
            new_index = distances.argmin().tolist()
            selected_indices.append(new_index)
            current_center = candidate_centers[new_index]

        return selected_indices


class ClosestToCenterSelectionStrategy(FeatureBasedExemplarsSelectionStrategy):
    """A greedy algorithm that selects the remaining exemplar that is the
    closest to the center of all elements (in feature space).
    """

    def make_sorted_indices_from_features(self, features: Tensor) -> List[int]:
        center = features.mean(dim=0)
        distances = pow(features - center, 2).sum(dim=1)
        return distances.argsort()


__all__ = [
    "ExemplarsBuffer",
    "ReservoirSamplingBuffer",
    "BalancedExemplarsBuffer",
    "ExperienceBalancedBuffer",
    "ClassBalancedBuffer",
    "ParametricBuffer",
    "ExemplarsSelectionStrategy",
    "RandomExemplarsSelectionStrategy",
    "FeatureBasedExemplarsSelectionStrategy",
    "HerdingSelectionStrategy",
    "ClosestToCenterSelectionStrategy",
]
