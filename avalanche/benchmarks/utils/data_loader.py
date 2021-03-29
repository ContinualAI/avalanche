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
from torch.utils.data.dataloader import DataLoader
from typing import Dict, List
import torch
from itertools import chain


class MultiTaskDataLoader:
    def __init__(self, data_dict: Dict, oversample_small_tasks: bool = False,
                 **kwargs):
        """ Custom data loader for multi-task training.
        The dictionary `data_dict` maps task ids into their
        corresponding datasets.

        When iterating over the data, it returns sequentially a different
        batch for each task (i.e. first a batch for task 1, then task 2,
        and so on). If `oversample_small_tasks == True` smaller tasks are
        oversampled to match the largest task.

        It is suggested to use this loader only if tasks have approximately the
        same length.

        :param data_dict: a dictionary with task ids as keys and Datasets
            as values.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        self.data_dict = data_dict
        self.loaders_dict: Dict[int, DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks

        for task_id, data in self.data_dict.items():
            self.loaders_dict[task_id] = DataLoader(data, **kwargs)
        self.max_len = max([len(d) for d in self.loaders_dict.values()])

    def __iter__(self):
        iter_dataloaders = {}
        for t in self.loaders_dict.keys():
            iter_dataloaders[t] = iter(self.loaders_dict[t])
        max_len = max([len(d) for d in iter_dataloaders.values()])

        try:
            for it in range(max_len):
                # list() is necessary because we may remove keys from the
                # dictionary. This would break the generator.
                for t in list(self.data_dict.keys()):
                    t_loader = iter_dataloaders[t]
                    try:
                        x, y, *_ = next(t_loader)
                        yield t, x, y
                    except StopIteration:
                        # StopIteration is thrown if dataset ends.
                        # reinitialize data loader
                        if self.oversample_small_tasks:
                            # reinitialize data loader
                            iter_dataloaders[t] = iter(self.loaders_dict[t])
                            x, y, *_ = next(iter_dataloaders[t])
                        else:
                            del iter_dataloaders[t]
                            continue
                        yield t, x, y
        except StopIteration:
            return

    def __len__(self):
        return self.max_len * len(self.loaders_dict)


class MultiTaskMultiBatchDataLoader:
    def __init__(self, data_dict: Dict, oversample_small_tasks: bool = False,
                 **kwargs):
        """ Custom data loader for multi-task training.
        The dictionary `data_dict` maps task ids into their
        corresponding datasets.

        mini-batches emitted by this dataloader are dictionaries with task
        labels as keys and mini-batches as values. Therefore, each mini-batch
        contains separate data for each task (i.e. key 1 batch for task 1).
        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        It is suggested to use this loader only if tasks have approximately the
        same length.

        :param data_dict: a dictionary with task ids as keys and Datasets
            as values.
        :param oversample_small_task: whether smaller tasks should be
            oversampled to match the largest one.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        self.data_dict = data_dict
        self.loaders_dict: Dict[int, DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks

        for task_id, data in self.data_dict.items():
            self.loaders_dict[task_id] = DataLoader(data, **kwargs)
        self.max_len = max([len(d) for d in self.loaders_dict.values()])

    def __iter__(self):
        iter_dataloaders = {}
        for t in self.loaders_dict.keys():
            iter_dataloaders[t] = iter(self.loaders_dict[t])
        max_len = max([len(d) for d in iter_dataloaders.values()])
        try:
            for it in range(max_len):
                mb_curr = {}
                # list() is necessary because we may remove keys from the
                # dictionary. This would break the generator.
                for t in list(self.data_dict.keys()):
                    t_loader = iter_dataloaders[t]
                    try:
                        x, y, *_ = next(t_loader)
                    except StopIteration:
                        # StopIteration is thrown if dataset ends.
                        if self.oversample_small_tasks:
                            # reinitialize data loader
                            iter_dataloaders[t] = iter(self.loaders_dict[t])
                            x, y, *_ = next(iter_dataloaders[t])
                        else:
                            del iter_dataloaders[t]
                            continue
                    mb_curr[t] = x, y
                yield mb_curr
        except StopIteration:
            return

    def __len__(self):
        return self.max_len


class MultiTaskJoinedBatchDataLoader:
    def __init__(self, data_dict: Dict, memory_dict: Dict = None, 
                 oversample_small_tasks: bool = False, batch_size: int = 32,
                 **kwargs):
        """ Custom data loader for multi-task training.
        The dictionary `data_dict` maps task ids into their corresponding
        training datasets. 
        The dictionary `memory_dict` maps task ids into their corresponding
        datasets of memories. 

        When iterating over the data, it returns a single batch containing
        example from different tasks (i.e. a batch containing a balanced number
        of examples from all the tasks in the `data_dict` and `memory_dict`). 
        
        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data_dict: a dictionary with task ids as keys and Datasets
            (training data) as values.
        :param memory_dict: a dictionary with task ids as keys and Datasets 
            (patterns in memory) as values.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param batch_size: the size of the batch. It must be greater than or
            equal to the number of tasks.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        self.data_dict = data_dict
        self.memory_dict = memory_dict
        self.loaders_data_dict: Dict[int, DataLoader] = {}
        self.loaders_memory_dict: Dict[int, DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks

        num_keys = len(self.data_dict.keys()) + len(self.memory_dict.keys())
        assert batch_size >= num_keys, "Batch size must be greator or equal " \
                                       "to the number of tasks"
        single_exp_batch_size = batch_size // num_keys
        remaining_example = batch_size % num_keys
        # print()
        # print('num keys: ' + str(num_keys))
        # print('batch size: ' + str(single_exp_batch_size))
        # print('resto: ' + str(remaining_example))
        self.loaders_data_dict, remaining_example = self._create_dataloaders(
                                data_dict, single_exp_batch_size,
                                remaining_example, **kwargs)
        self.loaders_memory_dict, remaining_example = self._create_dataloaders(
                                memory_dict, single_exp_batch_size, 
                                remaining_example, **kwargs)
        self.max_len = max([len(d) for d in chain(
            self.loaders_data_dict.values(), self.loaders_memory_dict.values())]
            )
    
    def __iter__(self):
        iter_data_dataloaders = {}
        iter_buffer_dataloaders = {}

        for t in self.loaders_data_dict.keys():
            iter_data_dataloaders[t] = iter(self.loaders_data_dict[t])
        for t in self.loaders_memory_dict.keys():
            iter_buffer_dataloaders[t] = iter(self.loaders_memory_dict[t])

        max_len = max([len(d) for d in chain(iter_data_dataloaders.values(), 
                       iter_buffer_dataloaders.values())])
        try:
            for it in range(max_len):
                mb_x = mb_y = None
                mb_curr = {}
                self._get_mini_batch_from_data_dict(
                    self.data_dict, iter_data_dataloaders, 
                    self.loaders_data_dict, self.oversample_small_tasks,
                    mb_curr)
                
                self._get_mini_batch_from_data_dict(
                    self.memory_dict, iter_buffer_dataloaders, 
                    self.loaders_memory_dict, self.oversample_small_tasks,
                    mb_curr)

                yield mb_curr
        except StopIteration:
            return

    def __len__(self):
        return self.max_len
    
    def _get_mini_batch_from_data_dict(self, data_dict, iter_dataloaders, 
                                       loaders_dict, oversample_small_tasks,
                                       mb_curr):
        # list() is necessary because we may remove keys from the
        # dictionary. This would break the generator.
        for t in list(data_dict.keys()):
            t_loader = iter_dataloaders[t]
            try:
                x, y, *_ = next(t_loader)
            except StopIteration:
                # StopIteration is thrown if dataset ends.
                # reinitialize data loader
                if oversample_small_tasks:
                    # reinitialize data loader
                    iter_dataloaders[t] = iter(loaders_dict[t])
                    x, y, *_ = next(iter_dataloaders[t])
                else:
                    del iter_dataloaders[t]
                    continue
            if t in mb_curr: 
                x_curr = torch.cat((mb_curr[t][0], x))
                y_curr = torch.cat((mb_curr[t][1], y))
                mb_curr[t] = x_curr, y_curr
            else:
                mb_curr[t] = x, y

    def _create_dataloaders(self, data_dict, single_exp_batch_size, 
                            remaining_example, **kwargs):
        loaders_dict: Dict[int, DataLoader] = {}
        for task_id, data in data_dict.items():
            current_batch_size = single_exp_batch_size
            if remaining_example > 0:
                current_batch_size += 1
                remaining_example -= 1
            loaders_dict[task_id] = DataLoader(
                data, batch_size=current_batch_size, **kwargs)
        return loaders_dict, remaining_example
