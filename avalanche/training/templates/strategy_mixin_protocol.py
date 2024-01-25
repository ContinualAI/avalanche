from typing import Iterable, List, Optional, TypeVar, Protocol

from torch import Tensor
import torch

from torch.optim.optimizer import Optimizer
from torch.nn import Module

from avalanche.benchmarks.scenarios.generic_scenario import (
    CLExperience,
)
from avalanche.benchmarks import DatasetExperience
from avalanche.core import BasePlugin

TExperienceType = TypeVar("TExperienceType", bound=CLExperience)
TSGDExperienceType = TypeVar("TSGDExperienceType", bound=DatasetExperience)
TMBinput = TypeVar("TMBinput")
TMBoutput = TypeVar("TMBoutput")


class BaseStrategyProtocol(Protocol[TExperienceType]):
    model: Module

    device: torch.device

    plugins: List[BasePlugin]

    experience: Optional[TExperienceType]

    is_training: bool

    current_eval_stream: Iterable[TExperienceType]


class SGDStrategyProtocol(
    BaseStrategyProtocol[TSGDExperienceType],
    Protocol[TSGDExperienceType, TMBinput, TMBoutput],
):
    """
    A protocol for strategies to be used for typing mixin classes.
    """

    mbatch: Optional[TMBinput]

    mb_output: Optional[TMBoutput]

    dataloader: Iterable[TMBinput]

    _stop_training: bool

    optimizer: Optimizer

    loss: Tensor

    _criterion: Module

    def forward(self) -> TMBoutput:
        ...

    def criterion(self) -> Tensor:
        ...

    def backward(self) -> None:
        ...

    def _make_empty_loss(self) -> Tensor:
        ...

    def make_optimizer(self, **kwargs):
        ...

    def optimizer_step(self) -> None:
        ...

    def model_adaptation(self, model: Optional[Module] = None) -> Module:
        ...

    def _unpack_minibatch(self):
        ...

    def _before_training_iteration(self, **kwargs):
        ...

    def _before_forward(self, **kwargs):
        ...

    def _after_forward(self, **kwargs):
        ...

    def _before_backward(self, **kwargs):
        ...

    def _after_backward(self, **kwargs):
        ...

    def _before_update(self, **kwargs):
        ...

    def _after_update(self, **kwargs):
        ...

    def _after_training_iteration(self, **kwargs):
        ...


class SupervisedStrategyProtocol(
    SGDStrategyProtocol[TSGDExperienceType, TMBinput, TMBoutput], Protocol
):
    mb_x: Tensor

    mb_y: Tensor

    mb_task_id: Tensor


class MetaLearningStrategyProtocol(
    SGDStrategyProtocol[TSGDExperienceType, TMBinput, TMBoutput], Protocol
):
    def _before_inner_updates(self, **kwargs):
        ...

    def _inner_updates(self, **kwargs):
        ...

    def _after_inner_updates(self, **kwargs):
        ...

    def _before_outer_update(self, **kwargs):
        ...

    def _outer_update(self, **kwargs):
        ...

    def _after_outer_update(self, **kwargs):
        ...


__all__ = [
    "SGDStrategyProtocol",
    "SupervisedStrategyProtocol",
    "MetaLearningStrategyProtocol",
]
