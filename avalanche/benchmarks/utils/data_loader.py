################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
    Avalanche supports data loading using pytorch's dataloaders.
    This module provides custom dataloaders for continual learning such as
    support for balanced dataloading between different tasks or balancing
    between the current data and the replay memory.
"""
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Union,
)
import numpy as np

import torch
from torch.utils.data import RandomSampler, DistributedSampler, Dataset
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.utils.collate_functions import (
    classification_collate_mbatches_fn,
)
from avalanche.benchmarks.utils.collate_functions import (
    detection_collate_fn as _detection_collate_fn,
)
from avalanche.benchmarks.utils.collate_functions import (
    detection_collate_mbatches_fn as _detection_collate_mbatches_fn,
)
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.distributed.distributed_helper import DistributedHelper

_default_collate_mbatches_fn = classification_collate_mbatches_fn

detection_collate_fn = _detection_collate_fn

detection_collate_mbatches_fn = _detection_collate_mbatches_fn

import torch
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import Sampler


def return_identity(x):
    """
    The identity function. Can be wrapped in 'partial'
    to act as a getter function.
    Used to avoid lambda functions that cannot be pickled.
    """
    return x


def collate_from_data_or_kwargs(data, kwargs):
    if "collate_fn" in kwargs:
        return
    elif hasattr(data, "collate_fn"):
        kwargs["collate_fn"] = data.collate_fn


class TaskBalancedDataLoader:
    """Task-balanced data loader for Avalanche's datasets."""

    def __init__(
        self,
        data: AvalancheDataset,
        oversample_small_tasks: bool = False,
        **kwargs
    ):
        """Task-balanced data loader for Avalanche's datasets.

        The iterator returns a mini-batch balanced across each task, which
        makes it useful when training in multi-task scenarios whenever data is
        highly unbalanced.

        If `oversample_small_tasks == True` smaller tasks are
        oversampled to match the largest task. Otherwise, once the data for a
        specific task is terminated, that task will not be present in the
        subsequent mini-batches.

        :param data: an instance of `AvalancheDataset`.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        if "collate_mbatches" in kwargs:
            raise ValueError(
                "collate_mbatches is not needed anymore and it has been "
                "deprecated. Data loaders will use the collate function"
                "`data.collate_fn`."
            )

        self.data = data
        self.dataloaders: Dict[int, DataLoader] = dict()
        self.oversample_small_tasks = oversample_small_tasks

        # split data by task.
        task_datasets = []
        task_labels_field = getattr(self.data, 'targets_task_labels')
        assert isinstance(task_labels_field, DataAttribute)
        for task_label in task_labels_field.uniques:

            tidxs = task_labels_field.val_to_idx[task_label]
            tdata = self.data.subset(tidxs)
            task_datasets.append(tdata)

        # the iteration logic is implemented by GroupBalancedDataLoader.
        # we use kwargs to pass the arguments to avoid passing the same
        # arguments multiple times.
        if "data" in kwargs:
            del kwargs["data"]
        # needed if they are passed as positional arguments
        kwargs["oversample_small_groups"] = oversample_small_tasks
        self._dl = GroupBalancedDataLoader(datasets=task_datasets, **kwargs)

    def __iter__(self):
        for el in self._dl.__iter__():
            yield el

    def __len__(self):
        return self._dl.__len__()


class GroupBalancedDataLoader:
    """Data loader that balances data from multiple datasets."""

    def __init__(
        self,
        datasets: Sequence[AvalancheDataset],
        oversample_small_groups: bool = False,
        batch_size: int = 32,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Data loader that balances data from multiple datasets.

        Mini-batches emitted by this dataloader are created by collating
        together mini-batches from each group. It may be used to balance data
        among classes, experiences, tasks, and so on.

        If `oversample_small_groups == True` smaller groups are oversampled to
        match the largest group. Otherwise, once data from a group is
        completely iterated, the group will be skipped.

        :param datasets: an instance of `AvalancheDataset`.
        :param oversample_small_groups: whether smaller groups should be
            oversampled to match the largest one.
        :param batch_size: the size of the batch. It must be greater than or
            equal to the number of groups.
        :param kwargs: data loader arguments used to instantiate the loader for
            each group separately. See pytorch :class:`DataLoader`.
        """
        if "collate_mbatches" in kwargs:
            raise ValueError(
                "collate_mbatches is not needed anymore and it has been "
                "deprecated. Data loaders will use the collate function"
                "`data.collate_fn`."
            )

        self.datasets = datasets
        self.batch_sizes = []
        self.oversample_small_groups = oversample_small_groups
        self.distributed_sampling = distributed_sampling
        self.loader_kwargs = kwargs
        if "collate_fn" in kwargs:
            self.collate_fn = kwargs["collate_fn"]
        else:
            self.collate_fn = self.datasets[0].collate_fn

        # collate is done after we have all batches
        # so we set an empty collate for the internal dataloaders
        self.loader_kwargs["collate_fn"] = return_identity

        # check if batch_size is larger than or equal to the number of datasets
        assert batch_size >= len(datasets)

        # divide the batch between all datasets in the group
        ds_batch_size = batch_size // len(datasets)
        remaining = batch_size % len(datasets)

        for _ in self.datasets:
            bs = ds_batch_size
            if remaining > 0:
                bs += 1
                remaining -= 1
            self.batch_sizes.append(bs)

        loaders_for_len_estimation = [
            _make_data_loader(
                dataset,
                distributed_sampling,
                kwargs,
                mb_size,
                force_no_workers=True,
            )[0]
            for dataset, mb_size in zip(self.datasets, self.batch_sizes)
        ]

        self.max_len = max([len(d) for d in loaders_for_len_estimation])

    def __iter__(self):
        dataloaders = []
        samplers = []
        for dataset, mb_size in zip(self.datasets, self.batch_sizes):
            data_l, data_l_sampler = _make_data_loader(
                dataset,
                self.distributed_sampling,
                self.loader_kwargs,
                mb_size,
            )

            dataloaders.append(data_l)
            samplers.append(data_l_sampler)

        iter_dataloaders = []
        for dl in dataloaders:
            iter_dataloaders.append(iter(dl))

        max_num_mbatches = max([len(d) for d in dataloaders])
        for it in range(max_num_mbatches):
            mb_curr = []
            removed_dataloaders_idxs = []
            # copy() is necessary because we may remove keys from the
            # dictionary. This would break the generator.
            for tid, (t_loader, t_loader_sampler) in enumerate(
                zip(iter_dataloaders, samplers)
            ):
                try:
                    batch = next(t_loader)
                except StopIteration:
                    # StopIteration is thrown if dataset ends.
                    if self.oversample_small_groups:
                        # reinitialize data loader
                        if isinstance(t_loader_sampler, DistributedSampler):
                            # Manage shuffling in DistributedSampler
                            t_loader_sampler.set_epoch(
                                t_loader_sampler.epoch + 1
                            )

                        iter_dataloaders[tid] = iter(dataloaders[tid])
                        batch = next(iter_dataloaders[tid])
                    else:
                        # We iteratated over all the data from this group
                        # and we don't need the iterator anymore.
                        iter_dataloaders[tid] = None
                        samplers[tid] = None
                        removed_dataloaders_idxs.append(tid)
                        continue
                mb_curr.extend(batch)
            yield self.collate_fn(mb_curr)

            # clear empty data-loaders
            for tid in reversed(removed_dataloaders_idxs):
                del iter_dataloaders[tid]
                del samplers[tid]

    def __len__(self):
        return self.max_len


class GroupBalancedInfiniteDataLoader:
    """Data loader that balances data from multiple datasets emitting an
    infinite stream."""

    def __init__(
        self,
        datasets: Sequence[AvalancheDataset],
        collate_mbatches=_default_collate_mbatches_fn,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Data loader that balances data from multiple datasets emitting an
        infinite stream.
        Mini-batches emitted by this dataloader are created by collating
        together mini-batches from each group. It may be used to balance data
        among classes, experiences, tasks, and so on.
        :param datasets: an instance of `AvalancheDataset`.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param kwargs: data loader arguments used to instantiate the loader for
            each group separately. See pytorch :class:`DataLoader`.
        """
        self.datasets = datasets
        self.dataloaders = []
        self.collate_mbatches = collate_mbatches

        for data in self.datasets:
            if DistributedHelper.is_distributed and distributed_sampling:
                seed = torch.randint(
                    0,
                    2 ** 32 - 1 - DistributedHelper.world_size,
                    (1,),
                    dtype=torch.int64,
                )
                seed += DistributedHelper.rank
                generator = torch.Generator()
                generator.manual_seed(int(seed))
            else:
                generator = None  # Default
            infinite_sampler = RandomSampler(
                data,
                replacement=True,
                num_samples=10 ** 10,
                generator=generator,
            )
            collate_from_data_or_kwargs(data, kwargs)
            dl = DataLoader(data, sampler=infinite_sampler, **kwargs)
            self.dataloaders.append(dl)
        self.max_len = 10 ** 10

    def __iter__(self):
        iter_dataloaders = []
        for dl in self.dataloaders:
            iter_dataloaders.append(iter(dl))

        while True:
            mb_curr = []
            for tid, t_loader in enumerate(iter_dataloaders):
                batch = next(t_loader)
                mb_curr.append(batch)
            yield self.collate_mbatches(mb_curr)

    def __len__(self):
        return self.max_len


class ReplayDataLoader:
    """Custom data loader for rehearsal/replay strategies."""

    def __init__(
        self,
        data: AvalancheDataset,
        memory: Optional[AvalancheDataset] = None,
        oversample_small_tasks: bool = False,
        batch_size: int = 32,
        batch_size_mem: int = 32,
        task_balanced_dataloader: bool = False,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Custom data loader for rehearsal strategies.

        This dataloader iterates in parallel two datasets, the current `data`
        and the rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).

        The length of the loader is determined only by the current 
        task data and is the same than what it would be when creating a 
        data loader for this dataset.

        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param batch_size: the size of the data batch. It must be greater
            than or equal to the number of tasks.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks.
        :param task_balanced_dataloader: if true, buffer data loaders will be
            task-balanced, otherwise it creates a single data loader for the
            buffer samples.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        if "collate_mbatches" in kwargs:
            raise ValueError(
                "collate_mbatches is not needed anymore and it has been "
                "deprecated. Data loaders will use the collate function"
                "`data.collate_fn`."
            )

        self.data = data
        self.memory = memory
        self.oversample_small_tasks = oversample_small_tasks
        self.task_balanced_dataloader = task_balanced_dataloader
        self.data_batch_sizes: Union[int, Dict[int, int]] = dict()
        self.memory_batch_sizes: Union[int, Dict[int, int]] = dict()
        self.distributed_sampling = distributed_sampling
        self.loader_kwargs = kwargs

        # Only used if persistent_workers == True in loader kwargs
        self._persistent_loader = None

        if "collate_fn" not in self.loader_kwargs:
            self.loader_kwargs["collate_fn"] = self.data.collate_fn
            
        self.data_batch_sizes, _ = self._get_batch_sizes(
            data, batch_size, 0, False
        )

        # Create dataloader for memory items
        if task_balanced_dataloader:
            memory_task_labels = getattr(self.memory, 'targets_task_labels')
            assert isinstance(memory_task_labels, DataAttribute)
            num_keys = len(memory_task_labels.uniques)

            # Ensure that the per-task batch size will end up > 0
            assert batch_size_mem >= num_keys, (
                "Batch size must be greator or equal "
                "to the number of tasks in the memory "
                "and current data."
            )

            single_group_batch_size = batch_size_mem // num_keys
            remaining_example = batch_size_mem % num_keys
        else:
            single_group_batch_size = batch_size_mem
            remaining_example = 0

        self.memory_batch_sizes, _ = self._get_batch_sizes(
            memory,
            single_group_batch_size,
            remaining_example,
            task_balanced_dataloader,
        )

        loaders_for_len_estimation = []

        if isinstance(self.data_batch_sizes, int):
            loaders_for_len_estimation.append(
                _make_data_loader(
                    data,
                    distributed_sampling,
                    kwargs,
                    self.data_batch_sizes,
                    force_no_workers=True,
                )[0]
            )
        else:
            # Task balanced
            data_task_set: Mapping[int, AvalancheDataset] = \
                getattr(data, 'task_set')
            for task_id in data_task_set:
                dataset = data_task_set[task_id]
                mb_sz = self.data_batch_sizes[task_id]

                loaders_for_len_estimation.append(
                    _make_data_loader(
                        dataset,
                        distributed_sampling,
                        kwargs,
                        mb_sz,
                        force_no_workers=True,
                    )[0]
                )

        self.max_len = max([len(d) for d in loaders_for_len_estimation])

    def __iter__(self):
        # Adapted from the __iter__ of PyTorch DataLoader:
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
        # Needed to support 'persistent_workers'

        use_persistent_workers = self.loader_kwargs.get(
            'persistent_workers', False)
        num_workers = self.loader_kwargs.get(
            'num_workers', 0)

        if use_persistent_workers and num_workers > 0:
            if self._persistent_loader is None:
                self._persistent_loader = self._get_loader()

            yield from self._persistent_loader
        else:
            yield from self._get_loader()
        
    def _get_loader(self):
        data_datasets, data_samplers = self._create_samplers(
            self.data,
            self.data_batch_sizes,
            self.distributed_sampling,
            self.loader_kwargs
        )

        memory_datasets, memory_samplers = self._create_samplers(
            self.memory,
            self.memory_batch_sizes,
            self.distributed_sampling,
            self.loader_kwargs
        )

        overall_dataset = ConcatDataset(data_datasets + memory_datasets)
        overall_samplers = data_samplers + memory_samplers

        # The longest sampler is the one that defines when an epoch ends
        # Note: this is aligned with the behavior of LegacyReplayDataLoader
        longest_data_sampler = np.array(
            len(d) for d in data_samplers
        ).argmax().item()

        multi_dataset_batch_sampler = MultiDatasetSampler(
            overall_dataset.datasets,
            overall_samplers,
            termination_dataset_idx=longest_data_sampler,
            oversample_small_tasks=self.oversample_small_tasks
        )

        loader = _make_data_loader_with_batched_sampler(
            overall_dataset,
            batch_sampler=multi_dataset_batch_sampler,
            data_loader_args=self.loader_kwargs
        )

        return loader

    def __len__(self):
        return self.max_len
    
    @staticmethod
    def _create_samplers(
        data: AvalancheDataset, 
        batch_sizes: Union[int, List[int], Dict[int, int]],
        distributed_sampling: bool,
        loader_kwargs: Dict[str, Any]
    ):
        datasets = []
        samplers = []

        if isinstance(batch_sizes, int):
            sampler = _make_sampler(
                data,
                distributed_sampling,
                loader_kwargs,
                batch_sizes
            )
            datasets.append(data)
            samplers.append(sampler)
        else:
            for task_id in data.task_set:  # TODO: sorted (deterministic) loop
                dataset = data.task_set[task_id]
                mb_sz = batch_sizes[task_id]

                sampler = _make_sampler(
                    dataset,
                    distributed_sampling,
                    loader_kwargs,
                    mb_sz
                )

                datasets.append(dataset)
                samplers.append(sampler)
        return datasets, samplers

    @staticmethod
    def _get_batch_sizes(
        data_dict,
        single_exp_batch_size,
        remaining_example,
        task_balanced_dataloader,
    ):
        batch_sizes = dict()
        if task_balanced_dataloader:
            for task_id in data_dict.task_set:
                current_batch_size = single_exp_batch_size
                if remaining_example > 0:
                    current_batch_size += 1
                    remaining_example -= 1
                batch_sizes[task_id] = current_batch_size
        else:
            # Current data is loaded without task balancing
            batch_sizes = single_exp_batch_size
        return batch_sizes, remaining_example
    

class LegacyReplayDataLoader:
    """Custom data loader for rehearsal/replay strategies."""

    def __init__(
        self,
        data: AvalancheDataset,
        memory: Optional[AvalancheDataset] = None,
        oversample_small_tasks: bool = False,
        batch_size: int = 32,
        batch_size_mem: int = 32,
        task_balanced_dataloader: bool = False,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Custom data loader for rehearsal strategies.

        This dataloader iterates in parallel two datasets, the current `data`
        and the rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).

        The length of the loader is determined only by the current 
        task data and is the same than what it would be when creating a 
        data loader for this dataset.

        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param batch_size: the size of the data batch. It must be greater
            than or equal to the number of tasks.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks.
        :param task_balanced_dataloader: if true, buffer data loaders will be
            task-balanced, otherwise it creates a single data loader for the
            buffer samples.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        if "collate_mbatches" in kwargs:
            raise ValueError(
                "collate_mbatches is not needed anymore and it has been "
                "deprecated. Data loaders will use the collate function"
                "`data.collate_fn`."
            )

        self.data = data
        self.memory = memory
        self.oversample_small_tasks = oversample_small_tasks
        self.task_balanced_dataloader = task_balanced_dataloader
        self.data_batch_sizes: Union[int, Dict[int, int]] = dict()
        self.memory_batch_sizes: Union[int, Dict[int, int]] = dict()
        self.distributed_sampling = distributed_sampling
        self.loader_kwargs = kwargs

        if "collate_fn" in kwargs:
            self.collate_fn = kwargs["collate_fn"]
        else:
            self.collate_fn = self.data.collate_fn

        # collate is done after we have all batches
        # so we set an empty collate for the internal dataloaders
        self.loader_kwargs["collate_fn"] = lambda x: x

        if task_balanced_dataloader:
            memory_task_labels = getattr(self.memory, 'targets_task_labels')
            assert isinstance(memory_task_labels, DataAttribute)
            num_keys = len(memory_task_labels.uniques)
            assert batch_size_mem >= num_keys, (
                "Batch size must be greator or equal "
                "to the number of tasks in the memory "
                "and current data."
            )

        self.data_batch_sizes, _ = self._get_batch_sizes(
            data, batch_size, 0, False
        )

        # Create dataloader for memory items
        if task_balanced_dataloader:
            memory_task_labels = getattr(self.memory, 'targets_task_labels')
            assert isinstance(memory_task_labels, DataAttribute)
            num_keys = len(memory_task_labels.uniques)
            single_group_batch_size = batch_size_mem // num_keys
            remaining_example = batch_size_mem % num_keys
        else:
            single_group_batch_size = batch_size_mem
            remaining_example = 0

        self.memory_batch_sizes, _ = self._get_batch_sizes(
            memory,
            single_group_batch_size,
            remaining_example,
            task_balanced_dataloader,
        )

        loaders_for_len_estimation = []

        if isinstance(self.data_batch_sizes, int):
            loaders_for_len_estimation.append(
                _make_data_loader(
                    data,
                    distributed_sampling,
                    kwargs,
                    self.data_batch_sizes,
                    force_no_workers=True,
                )[0]
            )
        else:
            # Task balanced
            data_task_set: Mapping[int, AvalancheDataset] = \
                getattr(data, 'task_set')
            for task_id in data_task_set:
                dataset = data_task_set[task_id]
                mb_sz = self.data_batch_sizes[task_id]

                loaders_for_len_estimation.append(
                    _make_data_loader(
                        dataset,
                        distributed_sampling,
                        kwargs,
                        mb_sz,
                        force_no_workers=True,
                    )[0]
                )

        self.max_len = max([len(d) for d in loaders_for_len_estimation])

    def __iter__(self):
        loader_data, sampler_data = self._create_loaders_and_samplers(
            self.data, self.data_batch_sizes
        )

        loader_memory, sampler_memory = self._create_loaders_and_samplers(
            self.memory, self.memory_batch_sizes
        )

        iter_data_dataloaders = {}
        iter_buffer_dataloaders = {}

        for t in loader_data.keys():
            iter_data_dataloaders[t] = iter(loader_data[t])
        for t in loader_memory.keys():
            iter_buffer_dataloaders[t] = iter(loader_memory[t])

        max_len = max([len(d) for d in loader_data.values()])

        try:
            for it in range(max_len):
                mb_curr: List[Any] = []
                ReplayDataLoader._get_mini_batch_from_data_dict(
                    iter_data_dataloaders,
                    sampler_data,
                    loader_data,
                    self.oversample_small_tasks,
                    mb_curr,
                )

                ReplayDataLoader._get_mini_batch_from_data_dict(
                    iter_buffer_dataloaders,
                    sampler_memory,
                    loader_memory,
                    self.oversample_small_tasks,
                    mb_curr,
                )

                yield self.collate_fn(mb_curr)
        except StopIteration:
            return

    def __len__(self):
        return self.max_len

    @staticmethod
    def _get_mini_batch_from_data_dict(
        iter_dataloaders,
        iter_samplers,
        loaders_dict,
        oversample_small_tasks,
        mb_curr,
    ):
        # list() is necessary because we may remove keys from the
        # dictionary. This would break the generator.
        for t in list(iter_dataloaders.keys()):
            t_loader = iter_dataloaders[t]
            t_sampler = iter_samplers[t]
            try:
                tbatch = next(t_loader)
            except StopIteration:
                # StopIteration is thrown if dataset ends.
                # reinitialize data loader
                if oversample_small_tasks:
                    # reinitialize data loader
                    if isinstance(t_sampler, DistributedSampler):
                        # Manage shuffling in DistributedSampler
                        t_sampler.set_epoch(t_sampler.epoch + 1)

                    iter_dataloaders[t] = iter(loaders_dict[t])
                    tbatch = next(iter_dataloaders[t])
                else:
                    del iter_dataloaders[t]
                    del iter_samplers[t]
                    continue
            mb_curr.extend(tbatch)

    def _create_loaders_and_samplers(self, data, batch_sizes):
        loaders = dict()
        samplers = dict()

        if isinstance(batch_sizes, int):
            loader, sampler = _make_data_loader(
                data,
                self.distributed_sampling,
                self.loader_kwargs,
                batch_sizes,
            )
            loaders[0] = loader
            samplers[0] = sampler
        else:
            for task_id in data.task_set:
                dataset = data.task_set[task_id]
                mb_sz = batch_sizes[task_id]

                loader, sampler = _make_data_loader(
                    dataset,
                    self.distributed_sampling,
                    self.loader_kwargs,
                    mb_sz,
                )

                loaders[task_id] = loader
                samplers[task_id] = sampler
        return loaders, samplers

    @staticmethod
    def _get_batch_sizes(
        data_dict,
        single_exp_batch_size,
        remaining_example,
        task_balanced_dataloader,
    ):
        batch_sizes = dict()
        if task_balanced_dataloader:
            for task_id in data_dict.task_set:
                current_batch_size = single_exp_batch_size
                if remaining_example > 0:
                    current_batch_size += 1
                    remaining_example -= 1
                batch_sizes[task_id] = current_batch_size
        else:
            # Current data is loaded without task balancing
            batch_sizes = single_exp_batch_size
        return batch_sizes, remaining_example


class MultiDatasetSampler(Sampler):
    """
    Iterate over datasets and provide a batch per dataset in each mini-batch.
    """
    def __init__(
            self,
            datasets: Sequence[Sized],
            samplers: Sequence[BatchSampler],
            termination_dataset_idx: int = 0,
            oversample_small_tasks: bool = False):
        assert len(datasets) == len(samplers)
        assert termination_dataset_idx >= 0 and \
            termination_dataset_idx < len(datasets)

        self.datasets = list(datasets)
        self.samplers = list(samplers)
        self.cumulative_sizes = ConcatDataset.cumsum(self.datasets)

        # termination_dataset_idx == dataset used to determine the epoch end
        self.termination_dataset_idx = termination_dataset_idx
        self.termination_dataset_iterations = \
            len(self.samplers[self.termination_dataset_idx])

        self.oversample_small_tasks = oversample_small_tasks
       
    def __len__(self):
        return self.termination_dataset_iterations

    def __iter__(self):
        number_of_datasets = len(self.datasets)
        samplers_list = []
        sampler_iterators = []

        for dataset_idx in range(number_of_datasets):
            sampler = self.samplers[dataset_idx]
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        index_offsets = np.array([0] + self.cumulative_sizes[:-1])

        while True:
            per_dataset_indices: List[Optional[np.ndarray]] = \
                [None] * number_of_datasets

            # Obtain the indices for the "main" dataset first
            sampling_dataset_order = [self.termination_dataset_idx] + list(
                x for x in range(number_of_datasets)
                if x != self.termination_dataset_idx
            )
            is_termination_dataset = \
                [True] + ([False] * (number_of_datasets - 1))

            for dataset_idx, is_term_dataset in zip(
                    sampling_dataset_order, 
                    is_termination_dataset):

                sampler = samplers_list[dataset_idx]
                sampler_iterator = sampler_iterators[dataset_idx]

                if sampler is None:
                    continue

                should_stop_if_ended = is_term_dataset or not \
                    self.oversample_small_tasks

                continue_epoch, updated_iterator, next_batch_indices = \
                    self._next_batch(
                        sampler,
                        sampler_iterator,
                        stop_on_last_batch=should_stop_if_ended
                    )

                if not continue_epoch:
                    if is_term_dataset:
                        # The main dataset terminated -> exit
                        return
                    else:
                        # Not the main dataset
                        # Happens if oversample_small_tasks is False
                        # Remove the dataset and sampler from the list
                        samplers_list[dataset_idx] = None
                        sampler_iterators[dataset_idx] = None
                        continue
                
                assert next_batch_indices is not None
                next_batch_indices = np.array(next_batch_indices)

                # Shift indices according to the position of the 
                # dataset in the list
                next_batch_indices += index_offsets[dataset_idx]
                
                sampler_iterators[dataset_idx] = updated_iterator
                per_dataset_indices[dataset_idx] = next_batch_indices
            per_dataset_indices = [x for x in per_dataset_indices 
                                   if x is not None]
            yield np.concatenate(per_dataset_indices).tolist()
    
    @staticmethod
    def _next_batch(
            sampler: Sampler,
            sampler_iterator: Iterator[Sequence[int]],
            stop_on_last_batch: bool):
        try:
            next_batch_indices = next(sampler_iterator)
            return True, sampler_iterator, next_batch_indices
        except StopIteration:
            if stop_on_last_batch:
                return False, None, None
        
        # Re-create the iterator
        # This time, do not catch StopIteration

        if isinstance(sampler, BatchSampler):
            if isinstance(sampler.sampler, DistributedSampler):
                sampler.sampler.set_epoch(
                    sampler.sampler.epoch + 1
                )
        elif isinstance(sampler, DistributedSampler):
            # Manage shuffling in DistributedSampler
            sampler.set_epoch(
                sampler.epoch + 1
            )
        
        sampler_iterator = iter(sampler)
        next_batch_indices = next(sampler_iterator)
        return True, sampler_iterator, next_batch_indices


def _make_data_loader(
    dataset: Dataset,
    distributed_sampling: bool,
    data_loader_args: Dict[str, Any],
    batch_size: int,
    force_no_workers: bool = False,
):
    data_loader_args = data_loader_args.copy()

    collate_from_data_or_kwargs(dataset, data_loader_args)

    if force_no_workers:
        data_loader_args['num_workers'] = 0
        if 'persistent_workers' in data_loader_args:
            data_loader_args['persistent_workers'] = False
        if 'prefetch_factor' in data_loader_args:
            data_loader_args['prefetch_factor'] = 2

    if DistributedHelper.is_distributed and distributed_sampling:
        # Note: shuffle only goes in the sampler, while
        # drop_last must be passed to both the sampler
        # and the DataLoader
        drop_last = data_loader_args.pop("drop_last", False)
        sampler = DistributedSampler(
            dataset,
            shuffle=data_loader_args.pop("shuffle", True),
            drop_last=drop_last,
        )
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            **data_loader_args
        )
    else:
        sampler = None
        data_loader = DataLoader(
            dataset, batch_size=batch_size, **data_loader_args
        )

    return data_loader, sampler


def _make_data_loader_with_batched_sampler(
    dataset: Dataset,
    batch_sampler: Any,
    data_loader_args: Dict[str, Any]
):
    data_loader_args = data_loader_args.copy()

    # See documentation of batch_sampler:
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # In fact, "generator" could be dropped too
    data_loader_args.pop("batch_size", False)
    data_loader_args.pop("shuffle", False)
    data_loader_args.pop("sampler", False)
    data_loader_args.pop("drop_last", False)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        **data_loader_args
    )


def _make_sampler(
    dataset: Any,
    distributed_sampling: bool,
    data_loader_args: Dict[str, Any],
    batch_size: int
):
    loader, _ = _make_data_loader(
        dataset,
        distributed_sampling,
        data_loader_args,
        batch_size,
        force_no_workers=True)
    
    sampler = loader.batch_sampler
    return sampler


__all__ = [
    "detection_collate_fn",
    "detection_collate_mbatches_fn",
    "collate_from_data_or_kwargs",
    "TaskBalancedDataLoader",
    "GroupBalancedDataLoader",
    "ReplayDataLoader",
    "GroupBalancedInfiniteDataLoader",
]
