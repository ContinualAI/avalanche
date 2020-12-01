################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################
from torch.utils.data import DataLoader
from typing import Dict


class MultiTaskDataLoader:
    def __init__(self, data_dict: Dict, **kwargs):
        """ Custom data loader for multi-task training.
        The dictionary `data_dict` maps the task ids into their
        corresponding datasets.

        When iterating over the data, it returns sequentially a different
        batch for each task (i.e. first a batch for task 1, then task 2,
        and so on). If datasets for different tasks have different lengths,
        smaller tasks are oversampled to match the largest task. It is
        suggested to use this loader only if tasks have approximately the
        same length.

        :param data_dict: a dictionary with task ids as keys and Datasets
            as values.
        :param kwargs: data loader arguments. See pytorch :class:`DataLoader`.
        """
        self.data_dict = data_dict
        self.max_len = max([len(d) for d in self.data_dict.values()])
        self.loaders_dict: Dict[int, DataLoader] = {}

        for task_id, data in self.data_dict.items():
            self.loaders_dict[task_id] = DataLoader(data, **kwargs)

    def __iter__(self):
        iter_dataloaders = {}
        for t in self.loaders_dict.keys():
            iter_dataloaders[t] = iter(self.loaders_dict[t])

        try:
            for it in range(self.max_len):
                for t in self.data_dict.keys():
                    t_loader = iter_dataloaders[t]
                    try:
                        x, y = next(t_loader)
                        yield t, x, y
                    except StopIteration:
                        # StopIteration is thrown if dataset ends.
                        # reinitialize data loader
                        iter_dataloaders[t] = iter(t_loader)
                        self.current_dataloader = iter_dataloaders[t]
                        x, y = next(t_loader)
                        yield t, x, y
        except StopIteration:
            return

    def __len__(self):
        return sum([len(dl) for dl in self.data_dict.values()])
