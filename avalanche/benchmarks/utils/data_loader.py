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
from typing import Dict


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
                            iter_dataloaders[t] = iter(t_loader)
                            self.current_dataloader = iter_dataloaders[t]
                            x, y = next(t_loader)
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
                            iter_dataloaders[t] = iter(t_loader)
                            self.current_dataloader = iter_dataloaders[t]
                            x, y = next(t_loader)
                        else:
                            del iter_dataloaders[t]
                            continue
                    mb_curr[t] = x, y
                yield mb_curr
        except StopIteration:
            return

    def __len__(self):
        return self.max_len
