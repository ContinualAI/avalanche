from abc import ABC

from typing import TypeVar, Generic

CallbackResult = TypeVar('CallbackResult')


class StrategyCallbacks(Generic[CallbackResult], ABC):
    """
    Base class for all classes dealing with strategy callbacks. Implements all
    the callbacks of the BaseStrategy with an empty function.
    Subclasses must override the desired callbacks.

    The main two direct subclasses are :class:`StrategyPlugin` and
    :class:`StrategyLogger`. The first defines a common interface for all
    plugins
    """

    def __init__(self):
        pass

    def before_training(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_training_exp(self, *args, **kwargs) -> CallbackResult:
        pass

    def adapt_train_dataset(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_training_epoch(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_training_iteration(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_forward(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_forward(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_backward(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_backward(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_training_iteration(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_update(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_update(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_training_epoch(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_training_exp(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_training(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_eval(self, *args, **kwargs) -> CallbackResult:
        pass

    def adapt_eval_dataset(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_eval_exp(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_eval_exp(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_eval(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_eval_iteration(self, *args, **kwargs) -> CallbackResult:
        pass

    def before_eval_forward(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_eval_forward(self, *args, **kwargs) -> CallbackResult:
        pass

    def after_eval_iteration(self, *args, **kwargs) -> CallbackResult:
        pass
