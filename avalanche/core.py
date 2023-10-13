from abc import ABC
from typing import Any, TypeVar, Generic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avalanche.training.templates.base import BaseTemplate

Template = TypeVar("Template", bound="BaseTemplate")


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

    supports_distributed: bool = False
    """
    A flag describing whether this plugin supports distributed training.
    """

    def __init__(self):
        """
        Inizializes an instance of a supervised plugin.
        """
        super().__init__()

    def before_training(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `train` by the `BaseTemplate`."""
        pass

    def before_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `train_exp` by the `BaseTemplate`."""
        pass

    def after_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after `train_exp` by the `BaseTemplate`."""
        pass

    def after_training(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after `train` by the `BaseTemplate`."""
        pass

    def before_eval(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `eval` by the `BaseTemplate`."""
        pass

    def before_eval_exp(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `eval_exp` by the `BaseTemplate`."""
        pass

    def after_eval_exp(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after `eval_exp` by the `BaseTemplate`."""
        pass

    def after_eval(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after `eval` by the `BaseTemplate`."""
        pass

    def __init_subclass__(cls, supports_distributed: bool = False, **kwargs) -> None:
        cls.supports_distributed = supports_distributed
        return super().__init_subclass__(**kwargs)


class BaseSGDPlugin(BasePlugin[Template], ABC):
    """ABC for BaseSGDTemplate plugins.

    See `BaseSGDTemplate` for complete description of the train/eval loop.
    """

    def __init__(self):
        """
        Inizializes an instance of a base SGD plugin.
        """
        super().__init__()

    def before_training_epoch(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `train_epoch` by the `BaseTemplate`."""
        pass

    def before_training_iteration(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before the start of a training iteration by the
        `BaseTemplate`."""
        pass

    def before_forward(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `model.forward()` by the `BaseTemplate`."""
        pass

    def after_forward(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after `model.forward()` by the `BaseTemplate`."""
        pass

    def before_backward(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `criterion.backward()` by the `BaseTemplate`."""
        pass

    def after_backward(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after `criterion.backward()` by the `BaseTemplate`."""
        pass

    def after_training_iteration(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after the end of a training iteration by the
        `BaseTemplate`."""
        pass

    def before_update(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `optimizer.update()` by the `BaseTemplate`."""
        pass

    def after_update(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after `optimizer.update()` by the `BaseTemplate`."""
        pass

    def after_training_epoch(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after `train_epoch` by the `BaseTemplate`."""
        pass

    def before_eval_iteration(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before the start of a training iteration by the
        `BaseTemplate`."""
        pass

    def before_eval_forward(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `model.forward()` by the `BaseTemplate`."""
        pass

    def after_eval_forward(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after `model.forward()` by the `BaseTemplate`."""
        pass

    def after_eval_iteration(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after the end of an iteration by the
        `BaseTemplate`."""
        pass

    def before_train_dataset_adaptation(
        self, strategy: Template, *args, **kwargs
    ) -> Any:
        """Called before `train_dataset_adapatation` by the `BaseTemplate`."""
        pass

    def after_train_dataset_adaptation(
        self, strategy: Template, *args, **kwargs
    ) -> Any:
        """Called after `train_dataset_adapatation` by the `BaseTemplate`."""
        pass

    def before_eval_dataset_adaptation(
        self, strategy: Template, *args, **kwargs
    ) -> Any:
        """Called before `eval_dataset_adaptation` by the `BaseTemplate`."""
        pass

    def after_eval_dataset_adaptation(self, strategy: Template, *args, **kwargs) -> Any:
        """Called after `eval_dataset_adaptation` by the `BaseTemplate`."""
        pass


class SupervisedPlugin(BaseSGDPlugin[Template], ABC):
    """ABC for SupervisedTemplate plugins.

    See `BaseTemplate` for complete description of the train/eval loop.
    """

    def __init__(self):
        """
        Inizializes an instance of a supervised plugin.
        """
        super().__init__()


class SupervisedMetaLearningPlugin(SupervisedPlugin[Template], ABC):
    """ABC for SupervisedMetaLearningTemplate plugins.

    See `BaseTemplate` for complete description of the train/eval loop.
    """

    def before_inner_updates(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `_inner_updates` by the `BaseTemplate`."""
        pass

    def after_inner_updates(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `_outer_updates` by the `BaseTemplate`."""
        pass

    def before_outer_update(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `_outer_updates` by the `BaseTemplate`."""
        pass

    def after_outer_update(self, strategy: Template, *args, **kwargs) -> Any:
        """Called before `_outer_updates` by the `BaseTemplate`."""
        pass
