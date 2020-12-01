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
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.benchmarks.utils.data_loader import MultiTaskDataLoader
from avalanche.training.strategies import BaseStrategy


class MultiTaskStrategy(BaseStrategy):
    def __init__(self, model: Module, optimizer: Optimizer, criterion, known_train_labels: bool = True,
                 known_test_labels: bool = True,
                 **kwargs):
        """
        MultiTaskStrategy is a CL strategy that provides task identities at train
        and test time (if needed).

        The main difference with :class:`BaseStrategy` is the use of task
        identities. See :class:`BaseStrategy` for additional documentation
        and constructor arguments. MultiTaskStrategy uses a
        :class:`MultiTaskDataLoader`, see its documentation for details
        about data loading order.

        :param known_train_labels: determines if labels are available
            at train time.
        :param known_test_labels: determines if labels are available
            at test time.
        :param **kwargs: See :class:`BaseStrategy`
        """
        self.known_train_labels = known_train_labels
        self.known_test_labels = known_test_labels
        super().__init__(model, optimizer, criterion, **kwargs)

        # State variables
        # MultiTaskStrategy adds a task-id.
        self.mb_x, self.mb_y, self.mb_task_id = None, None, None

    def adapt_train_dataset(self, **kwargs):
        """
        Called after the dataset initialization and before the
        dataloader initialization. Allows to customize the dataset.
        :param kwargs:
        :return:
        """
        self.current_data = {self.step_info.task_label: self.current_data}
        super().adapt_train_dataset(**kwargs)

    def make_train_dataloader(self, num_workers=0, shuffle=True, **kwargs):
        """
        Called after the dataset instantiation. Initialize the data loader.
        :param num_workers: number of thread workers for the data laoding.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """
        self.current_dataloader = MultiTaskDataLoader(self.current_data,
                                             num_workers=num_workers,
                                             batch_size=self.train_mb_size,
                                             shuffle=shuffle)

    def adapt_test_dataset(self, **kwargs):
        self.current_data = {self.step_info.task_label: self.current_data}
        super().adapt_test_dataset(**kwargs)

    def make_test_dataloader(self, num_workers=0, **kwargs):
        """
        Initialize the test data loader.
        :param num_workers:
        :param kwargs:
        :return:
        """
        self.current_dataloader = MultiTaskDataLoader(
              self.current_data,
              num_workers=num_workers,
              batch_size=self.test_mb_size)

    def training_epoch(self, **kwargs):
        """
        Training epoch.
        :param kwargs:
        :return:
        """
        for self.mb_it, (self.mb_task_id, self.mb_x, self.mb_y) in \
                enumerate(self.current_dataloader):
            if not self.known_train_labels:
                self.mb_task_id = None
            self.before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.mb_x = self.mb_x.to(self.device)
            self.mb_y = self.mb_y.to(self.device)

            # Forward
            self.before_forward(**kwargs)
            self.logits = self.model(self.mb_x)
            self.after_forward(**kwargs)

            # Loss & Backward
            self.loss = self.criterion(self.logits, self.mb_y)
            self.before_backward(**kwargs)
            self.loss.backward()
            self.after_backward(**kwargs)

            # Optimization step
            self.before_update(**kwargs)
            self.optimizer.step()
            self.after_update(**kwargs)

            self.after_training_iteration(**kwargs)

    def test_epoch(self, **kwargs):
        for self.mb_it, (self.mb_task_id, self.mb_x, self.mb_y) in \
                enumerate(self.current_dataloader):
            if not self.known_test_labels:
                self.mb_task_id = None
            self.before_test_iteration(**kwargs)

            self.mb_x = self.mb_x.to(self.device)
            self.mb_y = self.mb_y.to(self.device)

            self.before_test_forward(**kwargs)
            self.logits = self.model(self.mb_x)
            self.after_test_forward(**kwargs)
            self.loss = self.criterion(self.logits, self.mb_y)

            self.after_test_iteration(**kwargs)


__all__ = ['MultiTaskStrategy']
