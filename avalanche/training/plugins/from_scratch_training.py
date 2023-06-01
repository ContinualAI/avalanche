from copy import deepcopy
from typing import Dict, Optional

from torch import Tensor

from avalanche.core import BaseSGDPlugin
from avalanche.training.templates import BaseSGDTemplate
from avalanche.models.dynamic_optimizers import reset_optimizer


class FromScratchTrainingPlugin(BaseSGDPlugin):
    """From Scratch Training Plugin.

    This plugin resets the strategy's model weights and optimizer state after
    each experience. It expects the strategy to have a single model and
    optimizer. It can be used with the Naive strategy to produce
    "from-scratch training" baselines.
    """

    def __init__(self, reset_optimizer: bool = True):
        """
        Creates a `FromScratchTrainingPlugin` instance.

        :param reset_optimizer: if True, the startegy's optimizer state is
            reset after each experience.

        """
        super().__init__()
        self.reset_optimizer = reset_optimizer
        self.initial_weights: Optional[Dict[str, Tensor]] = None

    def before_training(self, strategy: BaseSGDTemplate, *args, **kwargs):
        """Called before `train` by the `BaseTemplate`."""

        # Save model's initial weights in the first experience training step
        if self.initial_weights is None:
            # Save initial weights
            self.initial_weights = deepcopy(strategy.model.state_dict())

    def before_training_exp(self, strategy: BaseSGDTemplate, *args, **kwargs):
        """Called after `train_exp` by the `BaseTemplate`."""
        # Copy the initial weights to the model
        init_weights = self.initial_weights
        assert init_weights is not None

        for n, p in strategy.model.named_parameters():
            if n in init_weights.keys():
                if p.data.shape == init_weights[n].data.shape:
                    p.data.copy_(init_weights[n].data)

        # Update the initial weights (in case new parameters are added)
        self.initial_weights = deepcopy(strategy.model.state_dict())

        # Reset the optimizer state
        if self.reset_optimizer:
            reset_optimizer(strategy.optimizer, strategy.model)


__all__ = ["FromScratchTrainingPlugin"]
