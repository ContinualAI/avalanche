from typing import Set, TypeVar, Generic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avalanche.training.templates.base import BaseTemplate

CallbackResult = TypeVar("CallbackResult")
Template = TypeVar("Template", covariant=True)


class Plugin():
    """
    A plugin is simply an object implementing some strategy callbacks.

    Callbacks provide access before/after each phase of the execution.
    In general, for each method of the training and evaluation loops,
    Most callbacks provide two functions `before_{method}` and 
    `after_{method}`, called before and after the method, respectively.
    Therefore plugins can "inject" additional code by implementing callbacks.
    Each callback has a `strategy` argument that gives access to the state.

    In Avalanche, callbacks are used to implement continual strategies, metrics
    and loggers.

    Continual learning strategies are diverse and as such not all plugin are 
    compatible with all strategies. Therefore the `Plugin` class tracks
    interfaces that a plugin requires. At runtime the plugin checks this feature
    set to see if any unsupported features exist. When creating a plugin
    simply inherit from the plugin features you need. ::

        class MyPlugin(TrainingEvents, RequiresSGD):
            pass

    As an advanced feature you may register a plugin as requiring 
    implementation by a continual learning strategy it should inherit from
    `Plugin` and be decorated with `@Plugin.requires_compatibility`. ::

        @Plugin.requires_compatibility
        class MyPlugin(Plugin):

            def new_callback():
                pass
    """
    
    # _features is a set of classes that must be supported for the plugin to
    # be usable
    _features: Set['Plugin'] = set()

    def __init_subclass__(cls) -> None:
        """
        When inheriting from one or more `Plugin`s the plugin also inherit the
        union of its required features. 
        """
        for base in cls.__bases__:
            if issubclass(base, Plugin):
                cls._features = cls._features.copy()
                cls._features = cls._features.union(base._features)
    
    @classmethod
    def features(cls) -> Set['Plugin']:
        """
        Returns a set of classes that a strategy should be compatible with
        """
        return cls._features.copy()

    @staticmethod
    def requires_compatibility(cls: 'Plugin'):
        """
        For defining a class as implementing new features that require 
        compatibility with the continual learning strategy.
        """
        assert issubclass(cls, Plugin), f"{cls} must be a {Plugin}"
        cls._features = cls._features.copy()   
        cls._features.add(cls)
        return cls


@Plugin.requires_compatibility
class TrainingEvents(Generic[Template], Plugin):
    """Defines the basic events used for training a model. Requires support by
    the strategy to be used.

    See `BaseSGDTemplate` for complete description of the train/eval loop.
    See `Plugin` for overview of how plugins work.
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


@Plugin.requires_compatibility
class EvalEvents(Generic[Template], Plugin):
    """
    Defines the basic events used for evaluating a model. Requires support by
    the strategy to be used.

    See `BaseSGDTemplate` for complete description of the train/eval loop.
    See `Plugin` for overview of how plugins work.
    """

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


@Plugin.requires_compatibility
class FittingEvents(Generic[Template], Plugin):
    """Defines the events for fitting a model. Requires support by
    the strategy to be used.

    See `BaseSGDTemplate` for complete description of the train/eval loop.
    See `Plugin` for overview of how plugins work.
    """

    # TODO Splitting this up might be advantageous?

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


@Plugin.requires_compatibility
class RequiresSGD(Plugin):
    """
    Marker interface used to indicate that a plugin requires the strategy to 
    use SGD. Requires support by the strategy to be used.

    See `Plugin` for overview of how plugins work.
    """


@Plugin.requires_compatibility
class DatasetAdaptationEvents(Generic[Template], Plugin):
    """Defines the events for dataset adapatation plugins. Requires support by
    the strategy to be used.

    See `BaseTemplate` for complete description of the train/eval loop.
    See `Plugin` for overview of how plugins work.
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


class BasePlugin(TrainingEvents, EvalEvents):
    """
    `BasePlugin` is a `Plugin` that requires the continual learning strategy
    support `TrainingEvents` and `EvalEvents`
    """


class BaseSGDPlugin(TrainingEvents, EvalEvents, FittingEvents, RequiresSGD):
    """
    `BaseSGDPlugin` is a `Plugin` that requires a continual learning strategy
    to support `TrainingEvents`, `EvalEvents`, `FittingEvents`, and requires
    that the strategy is optimized through gradient descent.
    """


class SupervisedPlugin(BaseSGDPlugin, DatasetAdaptationEvents):
    """
    `SupervisedPlugin` is a `Plugin` that requires the continual learning 
    strategy to support the same features as `BaseSGDPlugin` and 
    `DatasetAdaptationEvents`
    """
