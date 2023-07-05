from tqdm.auto import tqdm
from typing import Dict, Union

from torch.utils.data import DataLoader
import torch

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, ParamData


class MASPlugin(SupervisedPlugin):
    """
    Memory Aware Synapses (MAS) plugin.

    Similarly to EWC, the MAS plugin computes the importance of each
    parameter at the end of each experience. The approach computes
    importance via a second pass on the dataset. MAS does not require
    supervision and estimates importance using the gradients of the
    L2 norm of the output. Importance is then used to add a penalty
    term to the loss function.

    Technique introduced in:
    "Memory Aware Synapses: Learning what (not) to forget"
    by Aljundi et. al (2018).

    Implementation based on FACIL, as in:
    https://github.com/mmasana/FACIL/blob/master/src/approach/mas.py
    """

    def __init__(self, lambda_reg: float = 1.0, alpha: float = 0.5, verbose=False):
        """
        :param lambda_reg: hyperparameter weighting the penalty term
               in the loss.
        :param alpha: hyperparameter used to update the importance
               by also considering the influence in the previous
               experience.
        :param verbose: when True, the computation of the influence
               shows a progress bar using tqdm.
        """

        # Init super class
        super().__init__()

        # Regularization Parameters
        self._lambda = lambda_reg
        self.alpha = alpha

        # Model parameters
        self.params: Union[Dict, None] = None
        self.importance: Union[Dict, None] = None

        # Progress bar
        self.verbose = verbose

    def _get_importance(self, strategy):
        # Initialize importance matrix
        importance = dict(zerolike_params_dict(strategy.model))

        if not strategy.experience:
            raise ValueError("Current experience is not available")

        if strategy.experience.dataset is None:
            raise ValueError("Current dataset is not available")

        # Do forward and backward pass to accumulate L2-loss gradients
        strategy.model.train()
        collate_fn = (
            strategy.experience.dataset.collate_fn
            if hasattr(strategy.experience.dataset, "collate_fn")
            else None
        )
        dataloader = DataLoader(
            strategy.experience.dataset,
            batch_size=strategy.train_mb_size,
            collate_fn=collate_fn,
        )  # type: ignore

        # Progress bar
        if self.verbose:
            print("Computing importance")
            dataloader = tqdm(dataloader)

        for _, batch in enumerate(dataloader):
            # Get batch
            if len(batch) == 2 or len(batch) == 3:
                x, _, t = batch[0], batch[1], batch[-1]
            else:
                raise ValueError("Batch size is not valid")

            # Move batch to device
            x = x.to(strategy.device)

            # Forward pass
            strategy.optimizer.zero_grad()
            out = avalanche_forward(strategy.model, x, t)

            # Average L2-Norm of the output
            loss = torch.norm(out, p="fro", dim=1).pow(2).mean()
            loss.backward()

            # Accumulate importance
            for name, param in strategy.model.named_parameters():
                if param.requires_grad:
                    # In multi-head architectures, the gradient is going
                    # to be None for all the heads different from the
                    # current one.
                    if param.grad is not None:
                        importance[name].data += param.grad.abs()

        # Normalize importance
        for k in importance.keys():
            importance[k].data /= float(len(dataloader))

        return importance

    def before_backward(self, strategy, **kwargs):
        # Check if the task is not the first
        exp_counter = strategy.clock.train_exp_counter

        if exp_counter == 0:
            return

        loss_reg = 0.0

        # Check if properties have been initialized
        if not self.importance:
            raise ValueError("Importance is not available")
        if not self.params:
            raise ValueError("Parameters are not available")
        if not strategy.loss:
            raise ValueError("Loss is not available")

        # Apply penalty term
        for name, param in strategy.model.named_parameters():
            if name in self.importance.keys():
                loss_reg += torch.sum(
                    self.importance[name].expand(param.shape)
                    * (param - self.params[name].expand(param.shape)).pow(2)
                )

        # Update loss
        strategy.loss += self._lambda * loss_reg

    def after_training_exp(self, strategy, **kwargs):
        self.params = dict(copy_params_dict(strategy.model))

        # Get importance
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            self.importance = self._get_importance(strategy)
            return
        else:
            curr_importance = self._get_importance(strategy)

        # Check if previous importance is available
        if not self.importance:
            raise ValueError("Importance is not available")

        # Update importance
        for name in curr_importance.keys():
            new_shape = curr_importance[name].data.shape
            if name not in self.importance:
                self.importance[name] = ParamData(
                    name,
                    curr_importance[name].shape,
                    device=curr_importance[name].device,
                    init_tensor=curr_importance[name].data.clone(),
                )
            else:
                self.importance[name].data = (
                    self.alpha * self.importance[name].expand(new_shape)
                    + (1 - self.alpha) * curr_importance[name].data
                )
