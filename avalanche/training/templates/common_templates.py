import sys
from typing import Any, Callable, Dict, List, Sequence, Optional, TypeVar, Union
import warnings
import torch
import inspect

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch import Tensor
from ...benchmarks.scenarios.deprecated import DatasetExperience

from avalanche.core import BasePlugin, BaseSGDPlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import (
    EvaluationPlugin,
    default_evaluator,
)
from avalanche.training.templates.strategy_mixin_protocol import (
    SupervisedStrategyProtocol,
)

from .observation_type import *
from .problem_type import *
from .update_type import *
from .base_sgd import BaseSGDTemplate, CriterionType


TDatasetExperience = TypeVar("TDatasetExperience", bound=DatasetExperience)
TMBInput = TypeVar("TMBInput")
TMBOutput = TypeVar("TMBOutput")


class PositionalArgumentDeprecatedWarning(UserWarning):
    pass


def _merge_legacy_positional_arguments(
    self,
    args: Optional[Sequence[Any]],
    kwargs: Dict[str, Any],
    strict_init_check=True,
    allow_pos_args=True,
):
    """
    Manage the legacy positional constructor parameters.

    Used to warn the user about the deprecation of positional parameters
    in strategy constructors.

    To allow for a smooth transition, we allow the user to pass positional
    arguments to the constructor. However, we want to warn the user that
    this is deprecated and will be removed in the future.
    """

    init_method = getattr(self, "__init__")
    init_kwargs = inspect.signature(init_method).parameters

    if self.__class__ in [SupervisedTemplate, SupervisedMetaLearningTemplate]:
        return

    has_transition_varargs = any(
        x.kind == inspect.Parameter.VAR_POSITIONAL for x in init_kwargs.values()
    )
    if (not has_transition_varargs and args is not None) or (
        has_transition_varargs and args is None
    ):
        error_str = (
            "While trainsitioning to the new keyword-only strategy constructors, "
            "the ability to pass positional arguments (as in older versions of Avalanche) should be granted. "
            "You should catch all positional arguments (excluding self) using *args and do not declare other positional arguments! "
            "Then, pass them to SupervisedTemplate/SupervisedMetaLearningTemplate super constructor as the legacy_args argument."
            "Those legacy positional arguments will then be converted to keyword arguments "
            "according to the declaration order of those keyword arguments."
        )

        if strict_init_check:
            raise PositionalArgumentDeprecatedWarning(error_str)
        else:
            warnings.warn(error_str, category=PositionalArgumentDeprecatedWarning)

    positional_args = {
        (k, x)
        for k, x in init_kwargs.items()
        if x.kind
        in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]
    }

    if len(positional_args) > 0:
        all_positional_args_names_as_str = ", ".join([x[0] for x in positional_args])
        error_str = (
            "Avalanche is transitioning to strategy constructors that allow for keyword arguments only. "
            "Your strategy __init__ method still has some positional-only or positional-or-keyword arguments. "
            "Consider removing them. Offending arguments: "
            + all_positional_args_names_as_str
        )

        if strict_init_check:
            raise PositionalArgumentDeprecatedWarning(error_str)
        else:
            warnings.warn(error_str, category=PositionalArgumentDeprecatedWarning)

    # Discard all non-keyword params (also exclude VAR_KEYWORD)
    kwargs_order = [
        k for k, v in init_kwargs.items() if v.kind == inspect.Parameter.KEYWORD_ONLY
    ]

    print(init_kwargs)

    if len(args) > 0:

        error_str = (
            "Passing positional arguments to Strategy constructors is "
            "deprecated. Please use keyword arguments instead."
        )

        if allow_pos_args:
            warnings.warn(error_str, category=PositionalArgumentDeprecatedWarning)
        else:
            raise PositionalArgumentDeprecatedWarning(error_str)

        for i, arg in enumerate(args):
            kwargs[kwargs_order[i]] = arg

    for key, value in kwargs.items():
        if value == "not_set":
            raise ValueError(f"Parameter {key} is not set")

    return kwargs


class SupervisedTemplate(
    BatchObservation,
    SupervisedProblem,
    SGDUpdate,
    SupervisedStrategyProtocol[TDatasetExperience, TMBInput, TMBOutput],
    BaseSGDTemplate[TDatasetExperience, TMBInput, TMBOutput],
):
    """Base class for continual learning strategies.

    SupervisedTemplate is the super class of all supervised task-based
    continual learning strategies. It implements a basic training loop and
    callback system that allows to execute code at each experience of the
    training loop. Plugins can be used to implement callbacks to augment the
    training loop with additional behavior (e.g. a memory buffer for replay).

    **Scenarios**
    This strategy supports several continual learning scenarios:

    * class-incremental scenarios (no task labels)
    * multi-task scenarios, where task labels are provided)
    * multi-incremental scenarios, where the same task may be revisited

    The exact scenario depends on the data stream and whether it provides
    the task labels.

    **Training loop**
    The training loop is organized as follows::

        train
            train_exp  # for each experience
                adapt_train_dataset
                train_dataset_adaptation
                make_train_dataloader
                train_epoch  # for each epoch
                    # forward
                    # backward
                    # model update

    **Evaluation loop**
    The evaluation loop is organized as follows::

        eval
            eval_exp  # for each experience
                adapt_eval_dataset
                eval_dataset_adaptation
                make_eval_dataloader
                eval_epoch  # for each epoch
                    # forward

    """

    PLUGIN_CLASS = SupervisedPlugin

    # TODO: remove default values of model and optimizer when legacy positional arguments are definitively removed
    def __init__(
        self,
        *,
        legacy_positional_args: Optional[Sequence[Any]],
        model: Module = "not_set",
        optimizer: Optimizer = "not_set",
        criterion: CriterionType = CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[Sequence[BasePlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
        **kwargs,
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training. The default
            dataloader is a task-balanced dataloader that divides each
            mini-batch evenly between samples from all existing tasks in
            the dataset.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
        kwargs["model"] = model
        kwargs["optimizer"] = optimizer
        kwargs["criterion"] = criterion
        kwargs["train_mb_size"] = train_mb_size
        kwargs["train_epochs"] = train_epochs
        kwargs["eval_mb_size"] = eval_mb_size
        kwargs["device"] = device
        kwargs["plugins"] = plugins
        kwargs["evaluator"] = evaluator
        kwargs["eval_every"] = eval_every
        kwargs["peval_mode"] = peval_mode

        kwargs = _merge_legacy_positional_arguments(
            self, legacy_positional_args, kwargs
        )

        if sys.version_info >= (3, 11):
            super().__init__(**kwargs)
        else:
            super().__init__()
            BaseSGDTemplate.__init__(self=self, **kwargs)


class SupervisedMetaLearningTemplate(
    BatchObservation,
    SupervisedProblem,
    MetaUpdate,
    BaseSGDTemplate[TDatasetExperience, TMBInput, TMBOutput],
):
    """Base class for continual learning strategies.

    SupervisedMetaLearningTemplate is the super class of all supervised
    meta-learning task-based continual learning strategies. It implements a
    basic training loop and callback system that allows to execute code at
    each experience of the training loop. Plugins can be used to implement
    callbacks to augment the training loop with additional behavior
    (e.g. a memory buffer for replay).

    **Scenarios**
    This strategy supports several continual learning scenarios:

    * class-incremental scenarios (no task labels)
    * multi-task scenarios, where task labels are provided)
    * multi-incremental scenarios, where the same task may be revisited

    The exact scenario depends on the data stream and whether it provides
    the task labels.

    **Training loop**
    The training loop is organized as follows::

        train
            train_exp  # for each experience
                adapt_train_dataset
                train_dataset_adaptation
                make_train_dataloader
                train_epoch  # for each epoch
                    # inner_updates
                    # outer_update

    **Evaluation loop**
    The evaluation loop is organized as follows::

        eval
            eval_exp  # for each experience
                adapt_eval_dataset
                eval_dataset_adaptation
                make_eval_dataloader
                eval_epoch  # for each epoch
                    # forward

    """

    PLUGIN_CLASS = SupervisedPlugin

    # TODO: remove default values of model and optimizer when legacy positional arguments are definitively removed
    def __init__(
        self,
        *,
        legacy_positional_args: Optional[Sequence[Any]],
        model: Module = "not_set",
        optimizer: Optimizer = "not_set",
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[Sequence[BasePlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
        **kwargs,
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training. The default
            dataloader is a task-balanced dataloader that divides each
            mini-batch evenly between samples from all existing tasks in
            the dataset.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
        kwargs["model"] = model
        kwargs["optimizer"] = optimizer
        kwargs["criterion"] = criterion
        kwargs["train_mb_size"] = train_mb_size
        kwargs["train_epochs"] = train_epochs
        kwargs["eval_mb_size"] = eval_mb_size
        kwargs["device"] = device
        kwargs["plugins"] = plugins
        kwargs["evaluator"] = evaluator
        kwargs["eval_every"] = eval_every
        kwargs["peval_mode"] = peval_mode

        kwargs = _merge_legacy_positional_arguments(
            self, legacy_positional_args, kwargs
        )

        if sys.version_info >= (3, 11):
            super().__init__(**kwargs)
        else:
            super().__init__()
            BaseSGDTemplate.__init__(self=self, **kwargs)


__all__ = [
    "SupervisedTemplate",
    "SupervisedMetaLearningTemplate",
]
