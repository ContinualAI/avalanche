from typing import Callable, List, Sequence, Optional, Union
from packaging.version import parse
import warnings
import torch

from avalanche.training.templates.strategy_mixin_protocol import CriterionType

if parse(torch.__version__) < parse("2.0.0"):
    warnings.warn(f"LaMAML requires torch >= 2.0.0.")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch import Tensor
import math
from copy import deepcopy

from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedMetaLearningTemplate
from avalanche.training.storage_policy import ReservoirSamplingBuffer


class LaMAML(SupervisedMetaLearningTemplate):
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType = CrossEntropyLoss(),
        n_inner_updates: int = 5,
        second_order: bool = True,
        grad_clip_norm: float = 1.0,
        learn_lr: bool = True,
        lr_alpha: float = 0.25,
        sync_update: bool = False,
        alpha_init: float = 0.1,
        max_buffer_size: int = 200,
        buffer_mb_size: int = 10,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
    ):
        """Implementation of Look-ahead MAML (LaMAML) strategy.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param n_inner_updates: number of inner updates.
        :param second_order: If True, it computes the second-order derivative
               of the inner update trajectory for the meta-loss. Otherwise,
               it computes the meta-loss with a first-order approximation.
        :param grad_clip_norm: gradient clipping norm.
        :param learn_lr: if True, it learns the LR for each batch of data.
        :param lr_alpha: LR for learning the main update's learning rate.
        :param sync_update: if True, it updates the meta-model with a fixed
                            learning rate. Mutually exclusive with learn_lr and
                            lr_alpha.
        :param alpha_init: initialization value for learnable LRs.
        :param max_buffer_size: maximum buffer size. The default storage
               policy is reservoir-sampling.
        :param buffer_mb_size: number of buffer samples in each step.
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
        self.n_inner_updates = n_inner_updates
        self.second_order = second_order
        self.grad_clip_norm = grad_clip_norm
        self.learn_lr = learn_lr
        self.lr_alpha = lr_alpha
        self.sync_update = sync_update
        self.alpha_init = alpha_init
        self.alpha_params: nn.ParameterDict = nn.ParameterDict()
        self.alpha_params_initialized: bool = False
        self.meta_losses: List[Tensor] = []

        self.buffer = Buffer(
            max_buffer_size=max_buffer_size,
            buffer_mb_size=buffer_mb_size,
            device=device,
        )
        self.model.apply(init_kaiming_normal)

    def _before_training_exp(self, **kwargs):
        super()._before_training_exp(drop_last=True, **kwargs)

        # Initialize alpha-lr parameters
        if not self.alpha_params_initialized:
            self.alpha_params_initialized = True
            # Iterate through model parameters and add the corresponding
            # alpha_lr parameter
            for n, p in self.model.named_parameters():
                alpha_param = nn.Parameter(
                    torch.ones(p.shape) * self.alpha_init, requires_grad=True
                )
                self.alpha_params[n.replace(".", "_")] = alpha_param
            self.alpha_params.to(self.device)

            # Create optimizer for the alpha_lr parameters
            self.optimizer_alpha = torch.optim.SGD(
                self.alpha_params.parameters(), lr=self.lr_alpha
            )

        # update alpha-lr parameters
        for n, p in self.model.named_parameters():
            n = n.replace(".", "_")  # dict does not support names with '.'
            if n in self.alpha_params:
                if self.alpha_params[n].shape != p.shape:
                    old_shape = self.alpha_params[n].shape
                    # parameter expansion
                    expanded = False
                    assert len(p.shape) == len(
                        old_shape
                    ), "Expansion cannot add new dimensions"
                    for i, (snew, sold) in enumerate(zip(p.shape, old_shape)):
                        assert snew >= sold, "Shape cannot decrease."
                        if snew > sold:
                            assert not expanded, (
                                "Expansion cannot occur " "in more than one dimension."
                            )
                            expanded = True
                            exp_idx = i

                    alpha_param = torch.ones(p.shape) * self.alpha_init
                    idx = [
                        slice(el) if i != exp_idx else slice(old_shape[exp_idx])
                        for i, el in enumerate(p.shape)
                    ]
                    alpha_param[idx] = self.alpha_params[n].detach().clone()
                    alpha_param = nn.Parameter(alpha_param, requires_grad=True)
                    self.alpha_params[n] = alpha_param
            else:
                # Add new alpha_lr for the new parameter
                alpha_param = nn.Parameter(
                    torch.ones(p.shape) * self.alpha_init, requires_grad=True
                )
                self.alpha_params[n] = alpha_param

            self.alpha_params.to(self.device)
            # Re-init optimizer for the new set of alpha_lr parameters
            self.optimizer_alpha = torch.optim.SGD(
                self.alpha_params.parameters(), lr=self.lr_alpha
            )

    def copy_grads(self, params_1, params_2):
        for p1, p2 in zip(params_1, params_2):
            if p2.grad is not None:
                p1.grad = p2.grad

    def inner_update_step(self, fast_params, x, y, t):
        """Update fast weights using current samples and
        return the updated fast model.
        """
        logits = torch.func.functional_call(self.model, fast_params, (x, t))
        loss = self._criterion(logits, y)

        # Compute gradient with respect to the current fast weights
        grads = list(
            torch.autograd.grad(
                loss,
                fast_params.values(),
                retain_graph=self.second_order,
                create_graph=self.second_order,
                allow_unused=True,
            )
        )

        # Clip grad norms
        grads = [
            (
                torch.clamp(g, min=-self.grad_clip_norm, max=self.grad_clip_norm)
                if g is not None
                else g
            )
            for g in grads
        ]

        # New fast parameters
        new_fast_params = {
            n: param - alpha * grad if grad is not None else param
            for ((n, param), alpha, grad) in zip(
                fast_params.items(), self.alpha_params.parameters(), grads
            )
        }

        return new_fast_params

    def _inner_updates(self, **kwargs):
        # Make a copy of model parameters for fast updates
        self.initial_fast_params = {
            n: deepcopy(p) for (n, p) in self.model.named_parameters()
        }

        # Keep reference to the initial fast params
        fast_params = self.initial_fast_params

        # Samples from the current batch
        batch_x, batch_y, batch_t = self.mb_x, self.mb_y, self.mb_task_id

        # Get batches from the buffer
        if self.clock.train_exp_counter > 0:
            buff_x, buff_y, buff_t = self.buffer.get_buffer_batch()
            mixed_x = torch.cat([batch_x, buff_x], dim=0)
            mixed_y = torch.cat([batch_y, buff_y], dim=0)
            mixed_t = torch.cat([batch_t, buff_t], dim=0)
        else:
            mixed_x, mixed_y, mixed_t = batch_x, batch_y, batch_t

        # Split the current batch into smaller chuncks
        bsize_data = batch_x.shape[0]
        rough_sz = math.ceil(bsize_data / self.n_inner_updates)
        self.meta_losses = [torch.empty(0) for _ in range(self.n_inner_updates)]

        # Iterate through the chunks as inner-loops
        for i in range(self.n_inner_updates):
            batch_x_i = batch_x[i * rough_sz : (i + 1) * rough_sz]
            batch_y_i = batch_y[i * rough_sz : (i + 1) * rough_sz]
            batch_t_i = batch_t[i * rough_sz : (i + 1) * rough_sz]

            # We assume that samples for inner update are from the same task
            fast_params = self.inner_update_step(
                fast_params, batch_x_i, batch_y_i, batch_t_i
            )

            # Compute meta-loss with the combination of batch and buffer samples
            logits_meta = torch.func.functional_call(
                self.model, fast_params, (mixed_x, mixed_t)
            )
            meta_loss_i = self._criterion(logits_meta, mixed_y)
            self.meta_losses[i] = meta_loss_i

    def _outer_update(self, **kwargs):
        self.model.zero_grad()
        self.alpha_params.zero_grad()

        # Compute meta-gradient for the main model
        meta_loss = sum(self.meta_losses) / len(self.meta_losses)
        meta_loss.backward()

        self.copy_grads(self.model.parameters(), self.initial_fast_params.values())

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        if self.learn_lr:
            # Update lr for the current batch
            torch.nn.utils.clip_grad_norm_(
                self.alpha_params.parameters(), self.grad_clip_norm
            )
            self.optimizer_alpha.step()

        # If sync-update: update with self.optimizer
        # o.w: use the learned LRs to update the model
        if self.sync_update:
            self.optimizer.step()
        else:
            for p, alpha in zip(
                self.model.parameters(), self.alpha_params.parameters()
            ):
                # Use relu on updated LRs to avoid negative values
                if p.grad is not None:
                    p.data = p.data - p.grad * F.relu(alpha)

        self.loss = meta_loss

    def _after_training_exp(self, **kwargs):
        self.buffer.update(self)
        super()._after_training_exp(**kwargs)


class Buffer:
    def __init__(
        self, max_buffer_size=100, buffer_mb_size=10, device=torch.device("cpu")
    ):
        self.storage_policy = ReservoirSamplingBuffer(max_size=max_buffer_size)
        self.buffer_mb_size = buffer_mb_size
        self.device = device

    def update(self, strategy):
        self.storage_policy.update(strategy)

    def __len__(self):
        return len(self.storage_policy.buffer)

    def get_buffer_batch(self):
        rnd_ind = torch.randperm(len(self))[: self.buffer_mb_size]
        buff = self.storage_policy.buffer.subset(rnd_ind)
        buff_x, buff_y, buff_t = [], [], []
        for bx, by, bt in buff:
            buff_x.append(bx)
            buff_y.append(by)
            buff_t.append(bt)
        buff_x = torch.stack(buff_x, dim=0).to(self.device)
        buff_y = torch.tensor(buff_y).to(self.device).long()
        buff_t = torch.tensor(buff_t).to(self.device).long()
        return buff_x, buff_y, buff_t


def init_kaiming_normal(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()

    elif isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
