from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

try:
    import higher
except ImportError:
    raise ModuleNotFoundError("higher not found, if you want to use "
                              "MAML please install avalanche with "
                              "the extra dependencies: "
                              "pip install avalanche-lib[extra]")

from avalanche.NEW_core import BaseSGDPlugin
from avalanche.models.utils import avalanche_forward


class LaMAMLPlugin(BaseSGDPlugin):
    """LaMAML Plugin.
    """

    def __init__(
            self,
            n_inner_updates: int = 5,
            second_order: bool = True,
            grad_clip_norm: float = 1.0,
            learn_lr: bool = True,
            lr_alpha: float = 0.25,
            sync_update: bool = False,
            alpha_init: float = 0.1,
    ):
        """Implementation of Look-ahead MAML (LaMAML) algorithm in Avalanche
            using Higher library for applying fast updates.

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

        """

        super().__init__()

        self.n_inner_updates = n_inner_updates
        self.second_order = second_order
        self.grad_clip_norm = grad_clip_norm
        self.learn_lr = learn_lr
        self.lr_alpha = lr_alpha
        self.sync_update = sync_update
        self.alpha_init = alpha_init
        self.alpha_params = None
        self.is_model_initialized = False

    def before_training(self, strategy, **kwargs):
        if not self.is_model_initialized:
            strategy.model.apply(init_kaiming_normal)
            self.is_model_initialized = True

    def before_training_exp(self, strategy, **kwargs):
        # Initialize alpha-lr parameters
        if self.alpha_params is None:
            self.alpha_params = nn.ParameterList([])
            # Iterate through model parameters and add the corresponding
            # alpha_lr parameter
            for p in strategy.model.parameters():
                alpha_param = nn.Parameter(
                    torch.ones(p.shape) * self.alpha_init, requires_grad=True
                )
                self.alpha_params.append(alpha_param)
            self.alpha_params.to(strategy.device)

            # Create optimizer for the alpha_lr parameters
            self.optimizer_alpha = torch.optim.SGD(
                self.alpha_params.parameters(), lr=self.lr_alpha
            )

        # For task-incremental heads:
        # If new parameters are added to the model, update alpha_lr
        # parameters respectively
        if len(self.alpha_params) < len(list(strategy.model.parameters())):
            for iter_p, p in enumerate(strategy.model.parameters()):
                # Skip the older parameters
                if iter_p < len(self.alpha_params):
                    continue
                # Add new alpha_lr for the new parameter
                alpha_param = nn.Parameter(
                    torch.ones(p.shape) * self.alpha_init, requires_grad=True
                )
                self.alpha_params.append(alpha_param)

            self.alpha_params.to(strategy.device)
            # Re-init optimizer for the new set of alpha_lr parameters
            self.optimizer_alpha = torch.optim.SGD(
                self.alpha_params.parameters(), lr=self.lr_alpha
            )

    def before_inner_updates(self, strategy, **kwargs):
        # Create a stateless copy of the model for inner-updates
        self.fast_model = higher.patch.monkeypatch(
            strategy.model,
            copy_initial_weights=True,
            track_higher_grads=self.second_order,
        )
        if strategy.clock.train_exp_counter > 0:
            self.batch_x = strategy.mb_x[: strategy.train_mb_size]
            self.batch_y = strategy.mb_y[: strategy.train_mb_size]
            self.batch_t = strategy.mb_task_id[: strategy.train_mb_size]
        else:
            self.batch_x = strategy.mb_x
            self.batch_y = strategy.mb_y
            self.batch_t = strategy.mb_task_id

        bsize_data = self.batch_x.shape[0]
        self.rough_sz = math.ceil(bsize_data / self.n_inner_updates)
        self.meta_losses = [0 for _ in range(self.n_inner_updates)]

    def single_inner_update(self, x, y, t, criterion):
        logits = avalanche_forward(self.fast_model, x, t)
        loss = criterion(logits, y)

        # Compute gradient with respect to the current fast weights
        grads = list(
            torch.autograd.grad(
                loss,
                self.fast_model.fast_params,
                create_graph=self.second_order,
                retain_graph=self.second_order,
                allow_unused=True,
            )
        )

        # Clip grad norms
        grads = [
            torch.clamp(g, min=-self.grad_clip_norm, max=self.grad_clip_norm)
            if g is not None
            else g
            for g in grads
        ]

        # New fast parameters
        new_fast_params = [
            param - alpha * grad if grad is not None else param
            for (param, alpha, grad) in zip(
                self.fast_model.fast_params, self.alpha_params.parameters(),
                grads
            )
        ]

        # Update fast model's weights
        self.fast_model.update_params(new_fast_params)

    def inner_updates(self, strategy, **kwargs):
        """Update fast weights using current samples and
                return the updated fast model.
                """
        for i in range(self.n_inner_updates):
            batch_x_i = self.batch_x[i * self.rough_sz:
                                     (i + 1) * self.rough_sz]
            batch_y_i = self.batch_y[i * self.rough_sz:
                                     (i + 1) * self.rough_sz]
            batch_t_i = self.batch_t[i * self.rough_sz:
                                     (i + 1) * self.rough_sz]

            # We assume that samples for inner update are from the same task
            self.single_inner_update(batch_x_i, batch_y_i, batch_t_i,
                                     strategy._criterion)

            # Compute meta-loss with the combination of batch and buffer samples
            logits_meta = avalanche_forward(
                self.fast_model, strategy.mb_x, strategy.mb_task_id
            )
            meta_loss = strategy._criterion(logits_meta, strategy.mb_y)
            self.meta_losses[i] = meta_loss

    def apply_grad(self, module, grads, device):
        for i, p in enumerate(module.parameters()):
            grad = grads[i]
            if grad is None:
                grad = torch.zeros(p.shape).float().to(device)

            if p.grad is None:
                p.grad = grad
            else:
                p.grad += grad

    def outer_update(self, strategy, **kwargs):
        # Compute meta-gradient for the main model
        meta_loss = sum(self.meta_losses) / len(self.meta_losses)
        meta_grad_model = torch.autograd.grad(
            meta_loss,
            self.fast_model.parameters(time=0),
            retain_graph=True,
            allow_unused=True,
        )
        strategy.model.zero_grad()
        self.apply_grad(strategy.model, meta_grad_model, strategy.device)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            strategy.model.parameters(), self.grad_clip_norm
        )

        if self.learn_lr:
            # Compute meta-gradient for alpha-lr parameters
            meta_grad_alpha = torch.autograd.grad(
                meta_loss, self.alpha_params.parameters(), allow_unused=True
            )
            self.alpha_params.zero_grad()
            self.apply_grad(self.alpha_params, meta_grad_alpha, strategy.device)

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
                    strategy.model.parameters(), self.alpha_params.parameters()
            ):
                # Use relu on updated LRs to avoid negative values
                p.data = p.data - p.grad * F.relu(alpha)

        strategy.loss = meta_loss


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
