from typing import Callable, Sequence, Optional, Union

import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
import math

from avalanche.training.templates.strategy_mixin_protocol import CriterionType

try:
    import higher
except ImportError:
    warnings.warn(
        "higher not found, if you want to use "
        "MAML please install avalanche with "
        "the extra dependencies: "
        "pip install avalanche-lib[extra]"
    )

from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedMetaLearningTemplate
from avalanche.models.utils import avalanche_forward


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
        """Implementation of Look-ahead MAML (LaMAML) algorithm in Avalanche
            using Higher library for applying fast updates.

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

    def apply_grad(self, module, grads):
        for i, p in enumerate(module.parameters()):
            grad = grads[i]
            if grad is None:
                grad = torch.zeros(p.shape).float().to(self.device)

            if p.grad is None:
                p.grad = grad
            else:
                p.grad += grad

    def inner_update_step(self, fast_model, x, y, t):
        """Update fast weights using current samples and
        return the updated fast model.
        """
        logits = avalanche_forward(fast_model, x, t)
        loss = self._criterion(logits, y)

        # Compute gradient with respect to the current fast weights
        grads = list(
            torch.autograd.grad(
                loss,
                fast_model.fast_params,
                create_graph=self.second_order,
                retain_graph=self.second_order,
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
        new_fast_params = [
            param - alpha * grad if grad is not None else param
            for (param, alpha, grad) in zip(
                fast_model.fast_params, self.alpha_params.parameters(), grads
            )
        ]

        # Update fast model's weights
        fast_model.update_params(new_fast_params)

    def _inner_updates(self, **kwargs):
        # Create a stateless copy of the model for inner-updates
        self.fast_model = higher.patch.monkeypatch(
            self.model,
            copy_initial_weights=True,
            track_higher_grads=self.second_order,
        )
        if self.clock.train_exp_counter > 0:
            batch_x = self.mb_x[: self.train_mb_size]
            batch_y = self.mb_y[: self.train_mb_size]
            batch_t = self.mb_task_id[: self.train_mb_size]
        else:
            batch_x, batch_y, batch_t = self.mb_x, self.mb_y, self.mb_task_id

        bsize_data = batch_x.shape[0]
        rough_sz = math.ceil(bsize_data / self.n_inner_updates)
        self.meta_losses = [0 for _ in range(self.n_inner_updates)]

        for i in range(self.n_inner_updates):
            batch_x_i = batch_x[i * rough_sz : (i + 1) * rough_sz]
            batch_y_i = batch_y[i * rough_sz : (i + 1) * rough_sz]
            batch_t_i = batch_t[i * rough_sz : (i + 1) * rough_sz]

            # We assume that samples for inner update are from the same task
            self.inner_update_step(self.fast_model, batch_x_i, batch_y_i, batch_t_i)

            # Compute meta-loss with the combination of batch and buffer samples
            logits_meta = avalanche_forward(self.fast_model, self.mb_x, self.mb_task_id)
            meta_loss = self._criterion(logits_meta, self.mb_y)
            self.meta_losses[i] = meta_loss

    def _outer_update(self, **kwargs):
        # Compute meta-gradient for the main model
        meta_loss = sum(self.meta_losses) / len(self.meta_losses)
        meta_grad_model = torch.autograd.grad(
            meta_loss,
            self.fast_model.parameters(time=0),
            retain_graph=True,
            allow_unused=True,
        )
        self.model.zero_grad()
        self.apply_grad(self.model, meta_grad_model)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        if self.learn_lr:
            # Compute meta-gradient for alpha-lr parameters
            meta_grad_alpha = torch.autograd.grad(
                meta_loss, self.alpha_params.parameters(), allow_unused=True
            )
            self.alpha_params.zero_grad()
            self.apply_grad(self.alpha_params, meta_grad_alpha)

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
                p.data = p.data - p.grad * F.relu(alpha)

        self.loss = meta_loss


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


__all__ = ["LaMAML"]
