"""
This module contains Protocols for some of the main components of Avalanche,
such as strategy plugins and the agent state.

Most of these protocols are checked dynamically at runtime, so it is often not
necessary to inherit explicit from them or implement all the methods.
"""

from abc import ABC
from typing import Any, TypeVar, Generic, Protocol, runtime_checkable
from typing import TYPE_CHECKING

from avalanche.benchmarks import CLExperience

if TYPE_CHECKING:
    from avalanche.training.templates.base import BaseTemplate

Template = TypeVar("Template", bound="BaseTemplate")


class Agent:
    """Avalanche Continual Learning Agent.

    The agent stores the state needed by continual learning training methods,
    such as optimizers, models, regularization losses.
    You can add any objects as attributes dynamically:

    .. code-block::

        agent = Agent()
        agent.replay = ReservoirSamplingBuffer(max_size=200)
        agent.loss = MaskedCrossEntropy()
        agent.reg_loss = LearningWithoutForgetting(alpha=1, temperature=2)
        agent.model = my_model
        agent.opt = SGD(agent.model.parameters(), lr=0.001)
        agent.scheduler = ExponentialLR(agent.opt, gamma=0.999)

    Many CL objects will need to perform some operation before or
    after training on each experience. This is supported via the `Adaptable`
    Protocol, which requires the `pre_adapt` and `post_adapt` methods.
    To call the pre/post adaptation you can implement your training loop
    like in the following example:

    .. code-block::

        def train(agent, exp):
            agent.pre_adapt(exp)
            # do training here
            agent.post_adapt(exp)

    Objects that implement the `Adaptable` Protocol will be called by the Agent.

    You can also add additional functionality to the adaptation phases with
    hooks. For example:

    .. code-block::
        agent.add_pre_hooks(lambda a, e: update_optimizer(a.opt, new_params={}, optimized_params=dict(a.model.named_parameters())))
        # we update the lr scheduler after each experience (not every epoch!)
        agent.add_post_hooks(lambda a, e: a.scheduler.step())


    """

    def __init__(self, verbose=False):
        """Init.

        :param verbose: If True, print every time an adaptable object or hook
            is called during the adaptation. Useful for debugging.
        """
        self._updatable_objects = []
        self.verbose = verbose
        self._pre_hooks = []
        self._post_hooks = []

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(value, "pre_adapt") or hasattr(value, "post_adapt"):
            self._updatable_objects.append(value)
            if self.verbose:
                print("Added updatable object ", value)

    def pre_adapt(self, exp):
        """Pre-adaptation.

        Remember to call this before training on a new experience.

        :param exp: current experience
        """
        for uo in self._updatable_objects:
            if hasattr(uo, "pre_adapt"):
                uo.pre_adapt(self, exp)
                if self.verbose:
                    print("pre_adapt ", uo)
        for foo in self._pre_hooks:
            if self.verbose:
                print("pre_adapt hook ", foo)
            foo(self, exp)

    def post_adapt(self, exp):
        """Post-adaptation.

        Remember to call this after training on a new experience.

        :param exp: current experience
        """
        for uo in self._updatable_objects:
            if hasattr(uo, "post_adapt"):
                uo.post_adapt(self, exp)
                if self.verbose:
                    print("post_adapt ", uo)
        for foo in self._post_hooks:
            if self.verbose:
                print("post_adapt hook ", foo)
            foo(self, exp)

    def add_pre_hooks(self, foo):
        """Add a pre-adaptation hooks

        Hooks take two arguments: `<agent, experience>`.

        :param foo: the hook function
        """
        self._pre_hooks.append(foo)

    def add_post_hooks(self, foo):
        """Add a post-adaptation hooks

        Hooks take two arguments: `<agent, experience>`.

        :param foo: the hook function
        """
        self._post_hooks.append(foo)


class Adaptable(Protocol):
    """Adaptable objects Protocol.

    These class documents the Adaptable objects API but it is not necessary
    for an object to inherit from it since the `Agent` will search for the methods
    dynamically.

    Adaptable objects are objects that require to run their `pre_adapt` and
    `post_adapt` methods before (and after, respectively) training on each
    experience.

    Adaptable objects can implement only the method that they need since the
    `Agent` will look for the methods dynamically and call it only if it is
    implemented.
    """

    def pre_adapt(self, agent: Agent, exp: CLExperience):
        pass

    def post_adapt(self, agent: Agent, exp: CLExperience):
        pass


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
