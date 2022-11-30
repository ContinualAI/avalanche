#!/usr/bin/env python3
import copy
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.benchmarks.utils import concat_datasets
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import (OnlineSupervisedTemplate,
                                          SupervisedTemplate)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def er_acl_criterion(
    out_in, target_in, out_buffer, target_buffer, all_classes, new_classes
):
    loss_buffer = F.cross_entropy(out_buffer, target_buffer)
    loss_current = F.cross_entropy(
        out_in[:, len(all_classes) - len(new_classes) :],
        target_in - (len(all_classes) - len(new_classes)),
    )
    return loss_buffer + loss_current


class OnlineER_ACL(OnlineSupervisedTemplate):
    """
    ER ACL Online version, as originally proposed in
    "New Insights on Reducing Abrupt Representation Change in Online Continual Learning"
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
        plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
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

        # Use this that way we do not depend on
        # the ones present in self.experience attributes
        self.old_classes = set()
        self.new_classes = set()

    def _before_training_iteration(self, **kwargs):
        # Update self.classes_seen_so_far and self.new_classes
        current_classes = set(torch.unique(self.mb_y).cpu().numpy())
        inter_new = current_classes.intersection(self.new_classes)
        inter_old = current_classes.intersection(self.old_classes)
        if len(self.new_classes) == 0:
            self.new_classes = current_classes
        elif len(inter_new) == 0:
            # Intersection is null, new task has arrived
            self.old_classes.update(self.new_classes)
            self.new_classes = current_classes
        elif len(inter_new) > 0 and (
            len(current_classes.union(self.new_classes)) > len(self.new_classes)
        ):
            self.new_classes.update(current_classes)
        elif len(inter_new) > 0 and len(inter_old) > 0:
            raise ValueError(
                "Online ER ACL strategy cannot handle mixing of same classes in different tasks"
            )
        super()._before_training_iteration(**kwargs)

    @property
    def classes_seen_so_far(self):
        return self.new_classes.union(self.old_classes)

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

            if (
                len(self.new_classes) != len(self.classes_seen_so_far)
            ) and self.replay_loader is not None:
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
            if (
                len(self.new_classes) != len(self.classes_seen_so_far)
            ) and self.replay_loader is not None:
                self.mb_buffer_out = avalanche_forward(
                    self.model, self.mb_buffer_x, self.mb_buffer_tid
                )
            self._after_forward(**kwargs)

            # Loss & Backward
            if (
                len(self.new_classes) == len(self.classes_seen_so_far)
            ) or self.replay_loader is None:
                self.loss += self.criterion()
            else:
                self.loss += self.er_acl_criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _before_training_exp(self, **kwargs):
        ##############################
        #  Update Buffer and Loader  #
        ##############################
        self.storage_policy.update(self, **kwargs)

        # Take all classes for ER ACL loss
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

    def er_acl_criterion(self):
        return er_acl_criterion(
            self.mb_output,
            self.mb_y,
            self.mb_buffer_out,
            self.mb_buffer_y,
            self.classes_seen_so_far,
            self.new_classes,
        )


class ER_ACL(SupervisedTemplate):
    """
    ER ACL, as proposed in
    "New Insights on Reducing Abrupt Representation Change in Online Continual Learning"
    by Lucas Caccia et. al.
    https://openreview.net/forum?id=N8MaByOzUfb

    This version is adapted to non-online scenario, the difference with OnlineER_ACL
    is that it introduces all of the exemples from the new classes in the buffer at the
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
        plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
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

    @property
    def new_classes(self):
        return np.unique(self.experience.classes_in_this_experience)

    @property
    def classes_seen_so_far(self):
        return np.unique(self.experience.classes_seen_so_far)

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

            if len(self.new_classes) != len(self.classes_seen_so_far):
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
            if len(self.new_classes) != len(self.classes_seen_so_far):
                self.mb_buffer_out = avalanche_forward(
                    self.model, self.mb_buffer_x, self.mb_buffer_tid
                )
            self._after_forward(**kwargs)

            # Loss & Backward
            if len(self.new_classes) == len(self.classes_seen_so_far):
                self.loss += self.criterion()
            else:
                self.loss += self.er_acl_criterion()

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

    def er_acl_criterion(self):
        return er_acl_criterion(
            self.mb_output,
            self.mb_y,
            self.mb_buffer_out,
            self.mb_buffer_y,
            self.classes_seen_so_far,
            self.new_classes,
        )
