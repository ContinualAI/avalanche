from copy import deepcopy

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.models.dynamic_optimizers import reset_optimizer


class FromScratchTrainingPlugin(SupervisedPlugin):
    """ From Scratch Training Plugin.

    This plugin resets the strategy's model weights and optimizer state after
    each experience. It expects the strategy to have a single model and
    optimizer. It can be used with the Naive strategy to produce
    "from-scratch training" baselines.
    """

    def __init__(
            self,
            reset_optimizer: bool = True
    ):
        """
        Creates a `FromScratchTrainingPlugin` instance.

        :param reset_optimizer: if True, the startegy's optimizer state is
            reset after each experience.

        """
        super().__init__()
        self.reset_optimizer = reset_optimizer
        self.initial_weights = None

    def before_training(self, strategy, *args, **kwargs):
        """Called before `train` by the `BaseTemplate`."""

        # Save model's initial weights in the first experience training step
        if self.initial_weights is None:
            # Save initial weights
            self.initial_weights = deepcopy(strategy.model.state_dict())

    def before_training_exp(self, strategy: SupervisedTemplate,
                            *args, **kwargs):
        """Called after `train_exp` by the `BaseTemplate`."""
        # Copy the initial weights to the model
        for (n, p) in strategy.model.named_parameters():
            if n in self.initial_weights.keys():
                if p.data.shape == self.initial_weights[n].data.shape:
                    p.data.copy_(self.initial_weights[n].data)

        # Update the initial weights (in case new parameters are added)
        self.initial_weights = deepcopy(strategy.model.state_dict())

        # Reset the optimizer state
        if self.reset_optimizer:
            reset_optimizer(strategy.optimizer, strategy.model)
