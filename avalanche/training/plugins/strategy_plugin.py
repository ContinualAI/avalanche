from typing import Any

from avalanche.training import PluggableStrategy
from avalanche.training.strategy_callbacks import StrategyCallbacks


class StrategyPlugin(StrategyCallbacks[Any]):
    """
    Base class for strategy plugins. Implements all the callbacks required
    by the BaseStrategy with an empty function. Subclasses must override
    the callbacks.
    """

    def __init__(self):
        super().__init__()
        pass

    def before_training(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_training_exp(self, strategy: PluggableStrategy, **kwargs):
        pass

    def adapt_train_dataset(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_training_epoch(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_training_iteration(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_forward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_forward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_backward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_backward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_training_iteration(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_update(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_update(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_training_epoch(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_training_exp(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_training(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_eval(self, strategy: PluggableStrategy, **kwargs):
        pass

    def adapt_eval_dataset(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_eval_exp(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_eval_exp(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_eval(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_eval_iteration(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_eval_forward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_eval_forward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_eval_iteration(self, strategy: PluggableStrategy, **kwargs):
        pass