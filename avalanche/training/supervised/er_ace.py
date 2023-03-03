#!/usr/bin/env python3
import copy
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.benchmarks.utils import concat_datasets
from avalanche.core import SupervisedPlugin
from avalanche.models.utils import avalanche_forward
from avalanche.training import ACECriterion
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import (OnlineSupervisedTemplate,
                                          SupervisedTemplate)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


class OnlineER_ACE(OnlineSupervisedTemplate):
    """
    ER ACE Online version, as originally proposed in
    "New Insights on Reducing Abrupt Representation
    Change in Online Continual Learning"
    by Lucas Caccia et. al.
    https://openreview.net/forum?id=N8MaByOzUfb
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        mem_size: int = 200,
        batch_size_mem: int = None,
        train_mb_size: int = 1,
        train_passes: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=default_evaluator(),
        eval_every=-1,
        peval_mode="experience",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: int       : Fixed memory size
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param train_mb_size: mini-batch size for training.
        :param train_passes: number of training passes.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` experiences and at the end of
            the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experience or iterations (Default='experience').
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_passes,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.replay_loader = None
        self.ace_criterion = ACECriterion()

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            if self.replay_loader is not None:
                self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = next(
                    self.replay_loader
                )
                self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = (
                    self.mb_buffer_x.to(self.device),
                    self.mb_buffer_y.to(self.device),
                    self.mb_buffer_tid.to(self.device),
                )

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            if self.replay_loader is not None:
                self.mb_buffer_out = avalanche_forward(
                    self.model, self.mb_buffer_x, self.mb_buffer_tid
                )
            self._after_forward(**kwargs)

            # Loss & Backward
            if self.replay_loader is None:
                self.loss += self.criterion()
            else:
                self.loss += self.ace_criterion(
                    self.mb_output,
                    self.mb_y,
                    self.mb_buffer_out,
                    self.mb_buffer_y,
                )

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _before_training_exp(self, **kwargs):
        self.storage_policy.update(self, **kwargs)
        self.ace_criterion.update(torch.tensor(self.experience.dataset.targets))
        # Take all classes for ER ACE loss
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                )
            )
        else:
            self.replay_loader = None

        super()._before_training_exp(**kwargs)


class ER_ACE(SupervisedTemplate):
    """
    ER ACE, as proposed in
    "New Insights on Reducing Abrupt Representation
    Change in Online Continual Learning"
    by Lucas Caccia et. al.
    https://openreview.net/forum?id=N8MaByOzUfb

    This version is adapted to non-online scenario,
    the difference with OnlineER_ACE is that it introduces
    all of the exemples from the new classes in the buffer at the
    beggining of the task instead of introducing them progressively.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        mem_size: int = 200,
        batch_size_mem: int = 10,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=default_evaluator(),
        eval_every=-1,
        peval_mode="epoch",
    ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: int       : Fixed memory size
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.replay_loader = None
        self.ace_criterion = ACECriterion()

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            if self.replay_loader is not None:
                self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = next(
                    self.replay_loader
                )
                self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = (
                    self.mb_buffer_x.to(self.device),
                    self.mb_buffer_y.to(self.device),
                    self.mb_buffer_tid.to(self.device),
                )

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            if self.replay_loader is not None:
                self.mb_buffer_out = avalanche_forward(
                    self.model, self.mb_buffer_x, self.mb_buffer_tid
                )
            self._after_forward(**kwargs)

            # Loss & Backward
            if self.replay_loader is None:
                self.loss += self.criterion()
            else:
                self.loss += self.ace_criterion(
                    self.mb_output,
                    self.mb_y,
                    self.mb_buffer_out,
                    self.mb_buffer_y,
                )

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _before_training_exp(self, **kwargs):
        # Update buffer before training exp so that we have current data in
        self.storage_policy.update(self, **kwargs)
        self.ace_criterion.update(torch.tensor(self.experience.dataset.targets))
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                )
            )
        super()._before_training_exp(**kwargs)
