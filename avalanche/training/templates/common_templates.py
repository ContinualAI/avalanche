import sys
from typing import Callable, Sequence, Optional, TypeVar, Union
import torch

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from avalanche.benchmarks import DatasetExperience

from avalanche.core import BasePlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import (
    EvaluationPlugin,
    default_evaluator,
)
from avalanche.training.templates.strategy_mixin_protocol import (
    CriterionType,
    SupervisedStrategyProtocol,
    TMBOutput,
    TMBInput,
)

from .observation_type import *
from .problem_type import *
from .update_type import *
from .base_sgd import BaseSGDTemplate


TDatasetExperience = TypeVar("TDatasetExperience", bound=DatasetExperience)


class SupervisedTemplate(
    BatchObservation[TDatasetExperience, TMBInput, TMBInput],
    SupervisedProblem[TDatasetExperience, TMBInput, TMBInput],
    SGDUpdate[TDatasetExperience, TMBInput, TMBOutput],
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

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
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

        if sys.version_info >= (3, 11):
            super().__init__(**kwargs)
        else:
            super().__init__()
            BaseSGDTemplate.__init__(self=self, **kwargs)


class SupervisedMetaLearningTemplate(
    BatchObservation[TDatasetExperience, TMBInput, TMBInput],
    SupervisedProblem[TDatasetExperience, TMBInput, TMBInput],
    MetaUpdate[TDatasetExperience, TMBInput, TMBInput],
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

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
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

        if sys.version_info >= (3, 11):
            super().__init__(**kwargs)
        else:
            super().__init__()
            BaseSGDTemplate.__init__(self=self, **kwargs)


__all__ = [
    "SupervisedTemplate",
    "SupervisedMetaLearningTemplate",
]
