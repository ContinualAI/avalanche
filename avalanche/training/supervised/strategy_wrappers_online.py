################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta, Andrea Cossu, Hamed Hemati                         #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from typing import Callable, Optional, Sequence, Union
import torch

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from avalanche.core import BasePlugin

from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import (
    SupervisedTemplate,
)
from avalanche._annotations import deprecated
from avalanche.training.templates.strategy_mixin_protocol import CriterionType


@deprecated(
    0.5,
    "Online strategies are not differentiated"
    " from normal strategies anymore."
    "Please use Naive strategy instead",
)
class OnlineNaive(SupervisedTemplate):
    """Online naive finetuning.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType = CrossEntropyLoss(),
        train_passes: int = 1,
        train_mb_size: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[Sequence[BasePlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **kwargs
    ):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param num_passes: The number of passes for each sub-experience.
            Defaults to 1.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_epochs=train_passes,
            train_mb_size=train_mb_size,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **kwargs
        )


__all__ = ["OnlineNaive"]
