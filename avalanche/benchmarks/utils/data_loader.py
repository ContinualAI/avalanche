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
from itertools import chain
from typing import Dict, Optional, Sequence, Union

import torch
from torch.utils.data import RandomSampler, DistributedSampler
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.utils import make_classification_dataset
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

_default_collate_mbatches_fn = classification_collate_mbatches_fn

detection_collate_fn = _detection_collate_fn

detection_collate_mbatches_fn = _detection_collate_mbatches_fn


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
        for task_label in self.data.targets_task_labels.uniques:
            tidxs = self.data.targets_task_labels.val_to_idx[task_label]
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
        datasets: Sequence[make_classification_dataset],
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
        self.loader_kwargs["collate_fn"] = lambda x: x

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
        datasets: Sequence[make_classification_dataset],
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
            if _DistributedHelper.is_distributed and distributed_sampling:
                seed = torch.randint(
                    0,
                    2 ** 32 - 1 - _DistributedHelper.world_size,
                    (1,),
                    dtype=torch.int64,
                )
                seed += _DistributedHelper.rank
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
            num_keys = len(self.memory.targets_task_labels.uniques)
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
            num_keys = len(self.memory.targets_task_labels.uniques)
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
            for task_id in data.task_set:
                dataset = data.task_set[task_id]
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

        if isinstance(self.memory_batch_sizes, int):
            loaders_for_len_estimation.append(
                _make_data_loader(
                    memory,
                    distributed_sampling,
                    kwargs,
                    self.memory_batch_sizes,
                    force_no_workers=True,
                )[0]
            )
        else:
            for task_id in memory.task_set:
                dataset = memory.task_set[task_id]
                mb_sz = self.memory_batch_sizes[task_id]

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

        max_len = max(
            [
                len(d)
                for d in chain(
                    loader_data.values(),
                    loader_memory.values(),
                )
            ]
        )

        try:
            for it in range(max_len):
                mb_curr = []
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


def _make_data_loader(
    dataset,
    distributed_sampling,
    data_loader_args,
    batch_size,
    force_no_workers=False,
):
    data_loader_args = data_loader_args.copy()

    collate_from_data_or_kwargs(dataset, data_loader_args)

    if force_no_workers:
        data_loader_args['num_workers'] = 0
        if 'persistent_workers' in data_loader_args:
            data_loader_args['persistent_workers'] = False

    if _DistributedHelper.is_distributed and distributed_sampling:
        sampler = DistributedSampler(
            dataset,
            shuffle=data_loader_args.pop("shuffle", False),
            drop_last=data_loader_args.pop("drop_last", False),
        )
        data_loader = DataLoader(
            dataset, sampler=sampler, batch_size=batch_size, **data_loader_args
        )
    else:
        sampler = None
        data_loader = DataLoader(
            dataset, batch_size=batch_size, **data_loader_args
        )

    return data_loader, sampler


class __DistributedHelperPlaceholder:
    is_distributed = False
    world_size = 1
    rank = 0


_DistributedHelper = __DistributedHelperPlaceholder()


__all__ = [
    "detection_collate_fn",
    "detection_collate_mbatches_fn",
    "collate_from_data_or_kwargs",
    "TaskBalancedDataLoader",
    "GroupBalancedDataLoader",
    "ReplayDataLoader",
    "GroupBalancedInfiniteDataLoader",
]
