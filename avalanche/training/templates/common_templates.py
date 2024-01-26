from typing import Callable, Sequence, Optional, TypeVar, Union
import warnings
import torch

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


def _merge_legacy_positional_arguments(args, kwargs):
    # Manage the legacy positional constructor parameters
    if len(args) > 0:
        warnings.warn(
            "Passing positional arguments to Strategy constructors is "
            "deprecated. Please use keyword arguments instead."
        )

        # unroll args and apply it to kwargs
        legacy_kwargs_order = [
            "model",
            "optimizer",
            "criterion",
            "train_mb_size",
            "train_epochs",
            "eval_mb_size",
            "device",
            "plugins",
            "evaluator",
            "eval_every",
            "peval_mode",
        ]

        for i, arg in enumerate(args):
            kwargs[legacy_kwargs_order[i]] = arg

    for key, value in kwargs.items():
        if value == "not_set":
            raise ValueError(f"Parameter {key} is not set")

    return kwargs


class SupervisedTemplate(
    BatchObservation,
    SupervisedProblem,
    SGDUpdate,
    BaseSGDTemplate[TDatasetExperience, TMBInput, TMBOutput],
    SupervisedStrategyProtocol[TDatasetExperience, TMBInput, TMBOutput],
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
        *args,
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

        kwargs = _merge_legacy_positional_arguments(args, kwargs)

        super().__init__(**kwargs)
        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################

        # self.adapted_dataset = None
        # """ Data used to train. It may be modified by plugins. Plugins can
        # append data to it (e.g. for replay).
        #
        # .. note::
        #
        #    This dataset may contain samples from different experiences. If you
        #    want the original data for the current experience
        #    use :attr:`.BaseTemplate.experience`.


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
        *args,
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

        kwargs = _merge_legacy_positional_arguments(args, kwargs)

        super().__init__(**kwargs)
        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################

        # self.adapted_dataset = None
        # """ Data used to train. It may be modified by plugins. Plugins can
        # append data to it (e.g. for replay).
        #
        # .. note::
        #
        #    This dataset may contain samples from different experiences. If you
        #    want the original data for the current experience
        #    use :attr:`.BaseTemplate.experience`.


__all__ = [
    "SupervisedTemplate",
    "SupervisedMetaLearningTemplate",
]
