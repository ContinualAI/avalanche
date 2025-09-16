################################################################################
# Copyright (c) 2025.
# Copyrights licensed under the MIT License.
# See the accompanying LICENSE file for terms.
#
# Date: 2025-09-13
# Author(s): Muhammad Aniq, GPT-5 assistant
# Website: avalanche.continualai.org
################################################################################
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: F401 (kept for future extensions)

from avalanche.training.plugins.ewc import EWCPlugin


class HuberEWCPlugin(EWCPlugin):
    """
    Elastic Weight Consolidation with Huber loss regularization.

    This plugin extends the standard :class:`EWCPlugin` by replacing the
    quadratic penalty with a Huber penalty to improve stability when the
    regularization strength is large. The importance matrix computation and
    state handling are inherited from :class:`EWCPlugin`.

    The Huber loss is applied to the scaled parameter difference
    z = sqrt(importance) * (theta - theta_old) with threshold ``beta``.
    """

    def __init__(
        self,
        ewc_lambda: float,
        *,
        beta: float = 1.0,
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
    ):
        super().__init__(
            ewc_lambda=ewc_lambda,
            mode=mode,
            decay_factor=decay_factor,
            keep_importance_data=keep_importance_data,
        )
        self.beta = float(beta)

    @staticmethod
    def _huber_sum(x: torch.Tensor, beta: float) -> torch.Tensor:
        """Element-wise Huber with threshold beta, summed over all elements."""
        abs_x = x.abs()
        quad = 0.5 * (x**2)
        lin = beta * (abs_x - 0.5 * beta)
        return torch.where(abs_x <= beta, quad, lin).sum()

    def before_backward(self, strategy, **kwargs):
        """
        Compute Huber-based EWC penalty and add it to strategy.loss.
        """
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        device = strategy.device
        penalty = torch.tensor(0.0, device=device)

        if self.mode == "separate":
            for experience in range(exp_counter):
                for k, cur_param in strategy.model.named_parameters():
                    if k not in self.saved_params[experience]:
                        continue
                    saved_param = self.saved_params[experience][k]
                    imp = self.importances[experience][k]
                    new_shape = cur_param.shape
                    delta = cur_param - saved_param.expand(new_shape)
                    # Scale by sqrt(importance)
                    scaled_delta = imp.expand(new_shape).sqrt() * delta
                    # Use custom huber to avoid version mismatches
                    penalty = penalty + self._huber_sum(scaled_delta, self.beta)
        elif self.mode == "online":
            prev_exp = exp_counter - 1
            for k, cur_param in strategy.model.named_parameters():
                if k not in self.saved_params[prev_exp]:
                    continue
                saved_param = self.saved_params[prev_exp][k]
                imp = self.importances[prev_exp][k]
                new_shape = cur_param.shape
                delta = cur_param - saved_param.expand(new_shape)
                scaled_delta = imp.expand(new_shape).sqrt() * delta
                penalty = penalty + self._huber_sum(scaled_delta, self.beta)
        else:
            raise ValueError("Wrong EWC mode.")

        strategy.loss += self.ewc_lambda * penalty


__all__ = ["HuberEWCPlugin"]
