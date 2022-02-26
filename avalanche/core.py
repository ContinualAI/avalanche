from typing import Set, TypeVar, Generic, Type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avalanche.training.templates.base import BaseTemplate

CallbackResult = TypeVar("CallbackResult")
Template = TypeVar("Template", covariant=True)


FeatureSet = Set[Type['Plugin']]
"""
A feature set is a set of classes (inheriting from `Plugin`) that are used to 
communicate the requirements of a plugin to a strategy.
"""


class Plugin():
    """
    A plugin is used to add functionality to a strategy.

    To create your own plugin you should inherit from one or more plugin
    subclasses. ::

        class MyPlugin(TrainingEvents, RequiresSGD):
            def before_training(self, strategy, *args, **kwargs):
                pass

    Plugins typically provide access before/after each phase of the execution.
    In general, for each method of the training and evaluation loops,
    Most callbacks provide two functions `before_{method}` and 
    `after_{method}`, called before and after the method, respectively.

    In Avalanche, callbacks are used to implement continual strategies, metrics
    and loggers.

    Continual learning strategies are diverse and as such not all plugin are 
    compatible with all strategies. Therefore the `Plugin` class automatically
    tracks the features that a plugin requires. 
    
    For example `MyPlugin` requires that the strategy calls the callbacks in
    defined by `TrainingEvents` and that the strategy uses SGD (not all 
    strategies use SGD!). ::

        MyPlugin.required_featureset() == {TrainingEvents, RequiresSGD}

    At runtime a warning will be generated in the event an incompatible `Plugin`
    is used.
    """

    @classmethod
    def required_featureset(cls) -> FeatureSet:
        """
        Returns a set of classes that need to be explicitly supported by the
        strategy for the `Plugin` to be compatible. 
        """
        return set()

    @staticmethod
    def make_required_feature(feature_cls: Type['Plugin']) -> Type['Plugin']:
        """
        Decorator to note that a `Plugin` requires a strategy to explicitly
        support it.

        ::

            @Plugin.make_required_feature
            class FeatureA(Plugin):
                pass

            @Plugin.make_required_feature
            class FeatureB(Plugin):
                pass

            class MyPlugin(FeatureA, FeatureB):
                pass
            
            MyPlugin.required_featureset() == {FeatureA, FeatureB}
        """

        """
        Attach a class method to recursively build a FeatureSet. Since we take
        the union of the sets this can be used safely regardless of the method
        resolution order.
        """
        assert issubclass(feature_cls, Plugin), \
            f"{feature_cls} must inherit from {Plugin}"

        def _add_feature(cls: Type['Plugin']) -> FeatureSet:
            feature_set = super(feature_cls, cls).required_featureset()
            return feature_set.union({feature_cls})
        feature_cls.required_featureset = classmethod(_add_feature)
        return feature_cls


@Plugin.make_required_feature
class TrainingEvents(Generic[Template], Plugin):
    """Defines the basic events used for training a model. Requires explicit
    support from a strategy to be attached.

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


@Plugin.make_required_feature
class EvalEvents(Generic[Template], Plugin):
    """
    Defines the basic events used for evaluating a model. Requires explicit
    support from a strategy to be attached.

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


@Plugin.make_required_feature
class FittingEvents(Generic[Template], Plugin):
    """Defines the events for fitting a model. Requires explicit
    support from a strategy to be attached.

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


@Plugin.make_required_feature
class RequireSGD(Plugin):
    """
    Marker interface used to indicate that a plugin requires the strategy to 
    use SGD. Requires explicit support from a strategy to be attached.

    See `Plugin` for overview of how plugins work.
    """


@Plugin.make_required_feature
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


class BaseSGDPlugin(TrainingEvents, EvalEvents, FittingEvents, RequireSGD):
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
