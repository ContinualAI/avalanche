from abc import ABC
from typing import TypeVar, Generic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avalanche.training.templates.base import BaseTemplate

CallbackResult = TypeVar("CallbackResult")
Template = TypeVar("Template", covariant=True)


class BasePlugin(Generic[Template], ABC):
    """ABC for BaseTemplate plugins.

    A plugin is simply an object implementing some strategy callbacks.
    Plugins are called automatically during the strategy execution.

    Callbacks provide access before/after each phase of the execution.
    In general, for each method of the training and evaluation loops,
    `StrategyCallbacks`
    provide two functions `before_{method}` and `after_{method}`, called
    before and after the method, respectively.
    Therefore plugins can "inject" additional code by implementing callbacks.
    Each callback has a `strategy` argument that gives access to the state.

    In Avalanche, callbacks are used to implement continual strategies, metrics
    and loggers.
    """

    def __init__(self):
        pass

    def before_training(self, strategy: Template, *args, **kwargs):
        """Called before `train` by the `BaseTemplate`."""
        pass

    def before_training_exp(self, strategy: Template, *args, **kwargs):
        """Called before `train_exp` by the `BaseTemplate`."""
        pass

    def after_training_exp(self, strategy: Template, *args, **kwargs):
        """Called after `train_exp` by the `BaseTemplate`."""
        pass

    def after_training(self, strategy: Template, *args, **kwargs):
        """Called after `train` by the `BaseTemplate`."""
        pass

    def before_eval(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before `eval` by the `BaseTemplate`."""
        pass

    def before_eval_exp(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before `eval_exp` by the `BaseTemplate`."""
        pass

    def after_eval_exp(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called after `eval_exp` by the `BaseTemplate`."""
        pass

    def after_eval(self, strategy: Template, *args, **kwargs) -> CallbackResult:
        """Called after `eval` by the `BaseTemplate`."""
        pass


class BaseSGDPlugin(BasePlugin[Template], ABC):
    """ABC for BaseSGDTemplate plugins.

    See `BaseSGDTemplate` for complete description of the train/eval loop.
    """

    def before_training_epoch(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before `train_epoch` by the `BaseTemplate`."""
        pass

    def before_training_iteration(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before the start of a training iteration by the
        `BaseTemplate`."""
        pass

    def before_forward(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before `model.forward()` by the `BaseTemplate`."""
        pass

    def after_forward(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called after `model.forward()` by the `BaseTemplate`."""
        pass

    def before_backward(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before `criterion.backward()` by the `BaseTemplate`."""
        pass

    def after_backward(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called after `criterion.backward()` by the `BaseTemplate`."""
        pass

    def after_training_iteration(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called after the end of a training iteration by the
        `BaseTemplate`."""
        pass

    def before_update(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before `optimizer.update()` by the `BaseTemplate`."""
        pass

    def after_update(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called after `optimizer.update()` by the `BaseTemplate`."""
        pass

    def after_training_epoch(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called after `train_epoch` by the `BaseTemplate`."""
        pass

    def before_eval_iteration(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before the start of a training iteration by the
        `BaseTemplate`."""
        pass

    def before_eval_forward(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before `model.forward()` by the `BaseTemplate`."""
        pass

    def after_eval_forward(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called after `model.forward()` by the `BaseTemplate`."""
        pass

    def after_eval_iteration(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called after the end of an iteration by the
        `BaseTemplate`."""
        pass


class SupervisedPlugin(BaseSGDPlugin[Template], ABC):
    """ABC for SupervisedTemplate plugins.

    See `BaseTemplate` for complete description of the train/eval loop.
    """

    def before_train_dataset_adaptation(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before `train_dataset_adapatation` by the `BaseTemplate`."""
        pass

    def after_train_dataset_adaptation(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called after `train_dataset_adapatation` by the `BaseTemplate`."""
        pass

    def before_eval_dataset_adaptation(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called before `eval_dataset_adaptation` by the `BaseTemplate`."""
        pass

    def after_eval_dataset_adaptation(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        """Called after `eval_dataset_adaptation` by the `BaseTemplate`."""
        pass
