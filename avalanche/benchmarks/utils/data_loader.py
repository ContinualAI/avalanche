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
from itertools import chain
from typing import Dict

import torch
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.utils import AvalancheDataset


class MultiTaskDataLoader:
    def __init__(self, data: AvalancheDataset,
                 oversample_small_tasks: bool = False,
                 **kwargs):
        """ Custom data loader for Avalanche's datasets.

        When iterating over the data, it returns sequentially a different
        batch for each task (i.e. first a batch for task 1, then task 2,
        and so on). If `oversample_small_tasks == True` smaller tasks are
        oversampled to match the largest task.

        It is suggested to use this loader only if tasks have approximately the
        same length.

        :param data: an instance of `AvalancheDataset`.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        self.data = data
        self.dataloaders: Dict[int, DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks

        for task_label in self.data.task_set:
            tdata = self.data.task_set[task_label]
            self.dataloaders[task_label] = DataLoader(tdata, **kwargs)
        self.max_len = max([len(d) for d in self.dataloaders.values()])

    def __iter__(self):
        iter_dataloaders = {}
        for t in self.dataloaders.keys():
            iter_dataloaders[t] = iter(self.dataloaders[t])
        max_len = max([len(d) for d in iter_dataloaders.values()])

        try:
            for it in range(max_len):
                # list() is necessary because we may remove keys from the
                # dictionary. This would break the generator.
                for tid, t_loader in list(iter_dataloaders.items()):
                    try:
                        batch = next(t_loader)
                    except StopIteration:
                        # StopIteration is thrown if dataset ends.
                        # reinitialize data loader
                        if self.oversample_small_tasks:
                            # reinitialize data loader
                            iter_dataloaders[t] = iter(self.dataloaders[tid])
                            batch = next(iter_dataloaders[tid])
                        else:
                            del iter_dataloaders[t]
                            continue
                    yield batch
        except StopIteration:
            return

    def __len__(self):
        return self.max_len * len(self.dataloaders)


class MultiTaskMultiBatchDataLoader:
    def __init__(self, data: AvalancheDataset,
                 oversample_small_tasks: bool = False,
                 **kwargs):
        """ Custom data loader for task-balanced multi-task training.

        Mini-batches emitted by this dataloader are dictionaries with task
        labels as keys and mini-batches as values. Therefore, each mini-batch
        contains separate data for each task (i.e. key 1 batch for task 1).
        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: an instance of `AvalancheDataset`.
        :param oversample_small_task: whether smaller tasks should be
            oversampled to match the largest one.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        self.data = data
        self.dataloaders = {}
        self.oversample_small_tasks = oversample_small_tasks

        for task_label in self.data.task_set:
            tdata = self.data.task_set[task_label]
            self.dataloaders[task_label] = DataLoader(tdata, **kwargs)
        self.max_len = max([len(d) for d in self.dataloaders.values()])

    def __iter__(self):
        iter_dataloaders = {}
        for tid, dl in self.dataloaders.items():
            iter_dataloaders[tid] = iter(dl)

        max_num_mbatches = max([len(d) for d in iter_dataloaders.values()])
        for it in range(max_num_mbatches):
            mb_curr = {}
            # list() is necessary because we may remove keys from the
            # dictionary. This would break the generator.
            for tid, t_loader in list(iter_dataloaders.items()):
                t_loader = iter_dataloaders[tid]
                try:
                    batch = next(t_loader)
                except StopIteration:
                    # StopIteration is thrown if dataset ends.
                    if self.oversample_small_tasks:
                        # reinitialize data loader
                        iter_dataloaders[tid] = iter(self.dataloaders[tid])
                        batch = next(iter_dataloaders[tid])
                    else:
                        del iter_dataloaders[tid]
                        continue
                mb_curr[tid] = batch
            yield mb_curr

    def __len__(self):
        return self.max_len


class MultiTaskJoinedBatchDataLoader:
    def __init__(self, data: AvalancheDataset, memory: AvalancheDataset = None,
                 oversample_small_tasks: bool = False, batch_size: int = 32,
                 **kwargs):
        """ Custom data loader for rehearsal strategies.

        The current experience `data` and rehearsal `memory` are used to create
        the mini-batches by concatenating them together. Each mini-batch
        contains examples from each task (i.e. a batch containing a balanced
        number of examples from all the tasks in the `data` and `memory`).
        
        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: a dictionary with task ids as keys and Datasets
            (training data) as values.
        :param memory: a dictionary with task ids as keys and Datasets
            (patterns in memory) as values.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param batch_size: the size of the batch. It must be greater than or
            equal to the number of tasks.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        self.data = data
        self.memory = memory
        self.loader_data: Dict[int, DataLoader] = {}
        self.loader_memory: Dict[int, DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks

        num_keys = len(self.data.task_set) + len(self.memory.task_set)
        assert batch_size >= num_keys, "Batch size must be greator or equal " \
                                       "to the number of tasks"
        single_exp_batch_size = batch_size // num_keys
        remaining_example = batch_size % num_keys
        # print()
        # print('num keys: ' + str(num_keys))
        # print('batch size: ' + str(single_exp_batch_size))
        # print('resto: ' + str(remaining_example))
        self.loader_data, remaining_example = self._create_dataloaders(
            data, single_exp_batch_size,
            remaining_example, **kwargs)
        self.loader_memory, remaining_example = self._create_dataloaders(
            memory, single_exp_batch_size,
            remaining_example, **kwargs)
        self.max_len = max([len(d) for d in chain(
            self.loader_data.values(), self.loader_memory.values())]
                           )

    def __iter__(self):
        iter_data_dataloaders = {}
        iter_buffer_dataloaders = {}

        for t in self.loader_data.keys():
            iter_data_dataloaders[t] = iter(self.loader_data[t])
        for t in self.loader_memory.keys():
            iter_buffer_dataloaders[t] = iter(self.loader_memory[t])

        max_len = max([len(d) for d in chain(iter_data_dataloaders.values(),
                                             iter_buffer_dataloaders.values())])
        try:
            for it in range(max_len):
                mb_x = mb_y = None
                mb_curr = {}
                self._get_mini_batch_from_data_dict(
                    self.data, iter_data_dataloaders,
                    self.loader_data, self.oversample_small_tasks,
                    mb_curr)

                self._get_mini_batch_from_data_dict(
                    self.memory, iter_buffer_dataloaders,
                    self.loader_memory, self.oversample_small_tasks,
                    mb_curr)

                yield mb_curr
        except StopIteration:
            return

    def __len__(self):
        return self.max_len

    def _get_mini_batch_from_data_dict(self, data, iter_dataloaders,
                                       loaders_dict, oversample_small_tasks,
                                       mb_curr):
        # list() is necessary because we may remove keys from the
        # dictionary. This would break the generator.
        for t in list(data.task_set):
            t_loader = iter_dataloaders[t]
            try:
                tbatch = next(t_loader)
            except StopIteration:
                # StopIteration is thrown if dataset ends.
                # reinitialize data loader
                if oversample_small_tasks:
                    # reinitialize data loader
                    iter_dataloaders[t] = iter(loaders_dict[t])
                    tbatch = next(iter_dataloaders[t])
                else:
                    del iter_dataloaders[t]
                    continue
            if t in mb_curr:
                cat_batch = []
                for el1, el2 in zip(tbatch, mb_curr[t]):
                    cat_tensor = torch.cat([el1, el2], dim=0)
                    cat_batch.append(cat_tensor)
                mb_curr[t] = cat_batch
            else:
                mb_curr[t] = tbatch

    def _create_dataloaders(self, data_dict, single_exp_batch_size,
                            remaining_example, **kwargs):
        loaders_dict: Dict[int, DataLoader] = {}
        for task_id in data_dict.task_set:
            data = data_dict.task_set[task_id]
            current_batch_size = single_exp_batch_size
            if remaining_example > 0:
                current_batch_size += 1
                remaining_example -= 1
            loaders_dict[task_id] = DataLoader(
                data, batch_size=current_batch_size, **kwargs)
        return loaders_dict, remaining_example


__all__ = [
    'MultiTaskDataLoader',
    'MultiTaskMultiBatchDataLoader',
    'MultiTaskJoinedBatchDataLoader'
]
