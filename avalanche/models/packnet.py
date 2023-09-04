"""
Implements (Mallya & Lazebnik, 2018) PackNet algorithm for fixed-network
parameter isolation. PackNet is a task-incremental learning algorithm that
uses task identities to isolate parameters.

Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a
Single Network by Iterative Pruning. 2018 IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 7765-7773.
https://doi.org/10.1109/CVPR.2018.00810
"""

import typing as t

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from abc import ABC, abstractmethod
from enum import Enum
from avalanche.core import BaseSGDPlugin, Template
from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.models.simple_mlp import SimpleMLP
from avalanche.training.templates.base_sgd import BaseSGDTemplate


class PackNetModule(ABC):
    class State(Enum):
        """PackNet requires a procedure to be followed and we model this with
        the following states
        """

        TRAINING = 0
        """Train all of the remaining capacity"""
        POST_PRUNE = 1
        """Train only the un-pruned weights."""
        EVAL = 2
        """Activate a task-specific subset and mask the remaining weights. This
        state freezes all weights."""

    @abstractmethod
    def prune(self, prune_proportion: float) -> None:
        """Prune a proportion of the prunable parameters (parameters on the
        top of the stack) using the absolute value of the weights as a
        heuristic for importance.

        Prune may only be called when PackNet is in the TRAINING state. Prune
        will move PackNet to the POST_PRUNE state.

        :param prune_proportion: A proportion of the prunable parameters to
            prune
        """

    @abstractmethod
    def freeze_weights(self) -> None:
        """
        Commits the layer by incrementing counters and moving pruned parameters
        to the top of the stack. Biases are frozen as a side-effect.

        Freeze may only be called when PackNet is in the POST_PRUNE state.
        Freeze will move PackNet to the EVAL state.
        """

    @abstractmethod
    def train_uncommitted(self):
        """Unmasks all weights and freezes the weights belonging to past tasks.
        This allows the uncommitted layer to be trained without affecting the
        previous layer. Can only be called when PackNet is in the EVAL state
        i.e after `freeze_weights`. Moves PackNet to the TRAINING state.
        """

    @abstractmethod
    def activate_task(self, task_id: int):
        """Activates a task-specific subset in PackNet, by masking some weights.
        To avoid forgetting the network becomes immutable. Can only be called
        when PackNet is in the EVAL or TRAINING state. Moves PackNet to the
        EVAL state.

        :param task_id: The task id of the subset to activate
        """

    @abstractmethod
    def task_count(self) -> int:
        """Counts the number of task-specific subsets in PackNet.

        :return: The number of task-specific subsets
        """


class _ModuleDecorator(nn.Module):
    wrappee: nn.Module

    def forward(self, input: Tensor) -> Tensor:
        return self.wrappee.forward(input)

    def __init__(self, wrappee: nn.Module):
        super().__init__()
        self.add_module("wrappee", wrappee)


class StateError(Exception):
    pass


class PackNetDecorator(PackNetModule, _ModuleDecorator):
    def __init__(self, wrappee: nn.Module) -> None:
        super().__init__(wrappee)
        self.PRUNED_CODE: Tensor
        """Value used to mark pruned weights"""
        self.register_buffer("PRUNED_CODE", torch.tensor(255, dtype=torch.int))

        self._task_count: Tensor
        """Index top of the 'stack'. Should only increase"""
        self.register_buffer("_task_count", torch.tensor(0, dtype=torch.int))

        self.task_index: Tensor
        """Index of the task each weight belongs to"""
        self.register_buffer(
            "task_index",
            torch.ones(self.weight.shape).byte() * self._task_count,
        )

        self.visible_mask: Tensor
        """Mask of weights that are visible"""
        self.register_buffer(
            "visible_mask", torch.ones_like(self.weight, dtype=torch.bool)
        )

        self.unfrozen_mask: Tensor
        """Mask of weights that are mutable"""
        self.register_buffer(
            "unfrozen_mask", torch.ones_like(self.weight, dtype=torch.bool)
        )

        self._state: Tensor
        """The current state of the PackNet, see `State` for more information"""
        self.register_buffer(
            "_state", torch.tensor(self.State.TRAINING.value, dtype=torch.int)
        )

        self.weight.register_hook(self._remove_gradient_hook)

    @property
    @abstractmethod
    def weight(self) -> Tensor:
        raise NotImplementedError("weight not implemented")

    @property
    @abstractmethod
    def bias(self) -> Tensor:
        raise NotImplementedError("weight not implemented")

    @property
    def pruned_mask(self) -> Tensor:
        """Return a mask of weights that have been pruned"""
        return self.task_index.eq(self.PRUNED_CODE)

    @property
    def state(self) -> PackNetModule.State:
        return self.State(self._state.item())

    @state.setter
    def state(self, state: PackNetModule.State):
        self._state.fill_(state.value)

    def prune(self, prune_proportion: float):
        self._state_guard([self.State.TRAINING], self.State.POST_PRUNE)
        ranked = self._rank_prunable()
        prune_count = int(len(ranked) * prune_proportion)
        self._prune_weights(ranked[:prune_count])
        self.unfrozen_mask = self.task_index.eq(self._task_count)

    def available_weights(self) -> Tensor:
        return self.visible_mask * self.weight

    def train_uncommitted(self):
        self.activate_task(self.task_count())
        self._state_guard([self.State.EVAL], self.State.TRAINING)

        self.unfrozen_mask = self.task_index.eq(self.task_count())
        self.visible_mask = self.visible_mask | self.unfrozen_mask

    def activate_task(self, task_id: int):
        self._state_guard(
            [self.State.EVAL, self.State.TRAINING],
            self.State.EVAL,
        )
        self.visible_mask.zero_()
        self._is_subset_id_valid(task_id)
        self.visible_mask = self.visible_mask | self.task_index.eq(task_id)

    def freeze_weights(self):
        self._state_guard([self.State.POST_PRUNE], self.State.EVAL)
        self._task_count += 1
        self.task_index[self.pruned_mask] = self._task_count.item()
        # Change the active z_index
        self.unfrozen_mask.zero_()

        if self.bias is not None:
            self.bias.requires_grad = False

    def task_count(self) -> int:
        return int(self._task_count.item())

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def _remove_gradient_hook(self, grad: Tensor) -> Tensor:
        """
        Only the top layer should be trained. Todo so all other gradients
        are zeroed. Caution should be taken when optimizers with momentum are
        used, since they can cause parameters to be modified even when no
        gradient exists
        """
        return grad * self.unfrozen_mask

    def _rank_prunable(self) -> Tensor:
        """
        Returns a 1D tensor of the weights ranked based on their absolute value.
        Sorted to be in ascending order.
        """
        # "We use the simple heuristic to quantify the importance of the
        # weights using their absolute value." (Han et al., 2017)
        importance = self.weight.abs()
        un_prunable = ~self.unfrozen_mask
        # Mark un-prunable weights using -1.0 so they can be cutout after sort
        importance[un_prunable] = -1.0
        # Rank the importance
        rank = torch.argsort(importance.flatten())
        # Cut out un-prunable weights
        return rank[un_prunable.count_nonzero() :]

    def _prune_weights(self, indices: Tensor):
        self.task_index.flatten()[indices] = self.PRUNED_CODE.item()
        self.visible_mask.flatten()[indices] = False

    def _is_subset_id_valid(self, subset_id: t.List[int]):
        assert (
            0 <= subset_id <= self._task_count
        ), f"Given Subset ID {subset_id} must be between 0 and {self._task_count}"

    def _state_guard(
        self,
        previous: t.Sequence[PackNetModule.State],
        next: PackNetModule.State,
    ):
        """Ensure that the state is in the correct state and transition to the
        next correct state. If the state is not in the correct state then raise
        a StateError. This ensures that the correct procedure is followed.
        """
        if self.state not in previous:
            raise StateError(
                f"Function only valid for {previous} instead PackNet was "
                + f"in the {self.state} state"
            )
        self.state = next


class _PnLinear(PackNetDecorator):
    def __init__(self, wrappee: nn.Linear) -> None:
        self.wrappee: nn.Linear
        super().__init__(wrappee)

    @property
    def bias(self) -> Tensor:
        return self.wrappee.bias

    @property
    def weight(self) -> Tensor:
        return self.wrappee.weight

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.available_weights(), self.bias)


class _PnConv2d(PackNetDecorator):
    def __init__(self, wrappee: nn.Conv2d) -> None:
        wrappee: nn.Conv2d
        super().__init__(wrappee)

    @property
    def weight(self) -> Tensor:
        return self.wrappee.weight

    @property
    def bias(self) -> Tensor:
        return self.wrappee.bias

    @property
    def transposed(self) -> bool:
        return self.wrappee.transposed

    def forward(self, input: Tensor) -> Tensor:
        return self.wrappee._conv_forward(input, self.available_weights(), self.bias)


class _PnConvTransposed2d(PackNetDecorator):
    def __init__(self, wrappee: nn.ConvTranspose2d) -> None:
        wrappee: nn.ConvTranspose2d
        super().__init__(wrappee)

    @property
    def weight(self) -> Tensor:
        return self.wrappee.weight

    @property
    def bias(self) -> Tensor:
        return self.wrappee.bias

    @property
    def transposed(self) -> bool:
        return self.wrappee.transposed

    def forward(
        self, input: Tensor, output_size: t.Optional[t.List[int]] = None
    ) -> Tensor:
        w = self.wrappee
        if w.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(w.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding"
        # because torch.Script does not support `Sequence[T]` or
        # `Tuple[T, ...]`.
        output_padding = w._output_padding(
            input, output_size, w.stride, w.padding, w.kernel_size, w.dilation
        )  # type: ignore[arg-type]

        return F.conv_transpose2d(
            input,
            self.available_weights(),
            w.bias,
            w.stride,
            w.padding,
            output_padding,
            w.groups,
            w.dilation,
        )


def PackNetSimpleMLP(
    num_classes=10,
    input_size=28 * 28,
    hidden_size=512,
    hidden_layers=1,
    drop_rate=0.5,
):
    return PackNet(
        SimpleMLP(num_classes, input_size, hidden_size, hidden_layers, drop_rate)
    )


class PackNet(_ModuleDecorator, PackNetModule, MultiTaskModule):
    """
    PackNet implements the PackNet algorithm for parameter isolation. It
    can upgrade some PyTorch modules to be PackNet compatible. It currently only
    supports `nn.Linear`, `nn.Conv2d` and `nn.ConvTranspose2d` modules but
    can be extended to support other modules.


    Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a
    Single Network by Iterative Pruning. 2018 IEEE/CVF Conference on Computer
    Vision and Pattern Recognition, 7765-7773.
    https://doi.org/10.1109/CVPR.2018.00810
    """

    @staticmethod
    def wrap(wrappee: nn.Module):
        # Remove Weight Norm
        if hasattr(wrappee, "weight_g") and hasattr(wrappee, "weight_v"):
            raise RuntimeError("PackNet does not support weight norm")

        # Recursive cases
        if isinstance(wrappee, nn.Linear):
            return _PnLinear(wrappee)
        elif isinstance(wrappee, nn.Conv2d):
            return _PnConv2d(wrappee)
        elif isinstance(wrappee, nn.ConvTranspose2d):
            return _PnConvTransposed2d(wrappee)
        elif isinstance(wrappee, nn.Sequential):
            # Wrap each submodule
            for i, x in enumerate(wrappee):
                wrappee[i] = PackNet.wrap(x)
        else:
            for submodule_name, submodule in wrappee.named_children():
                setattr(wrappee, submodule_name, PackNet.wrap(submodule))
        return wrappee

    def __init__(self, wrappee: nn.Module) -> None:
        super().__init__(PackNet.wrap(wrappee))
        self._subset_count: Tensor
        self._active_task_id: Tensor
        self.register_buffer("_active_task_id", torch.tensor(0))
        self.register_buffer("_subset_count", torch.tensor(0))

    def _pn_apply(self, func: t.Callable[["PackNet"], None]):
        @torch.no_grad()
        def __pn_apply(module):
            # Apply function to all child packnets but not other parents.
            # If we were to apply to other parents we would duplicate
            # applications to their children
            if isinstance(module, PackNetModule) and not isinstance(module, PackNet):
                func(module)

        self.apply(__pn_apply)

    def prune(self, to_prune_proportion: float) -> None:
        """
        Prunes the layer by removing the smallest weights and freezing them.
        Biases are frozen as a side-effect.
        """
        self._pn_apply(lambda x: x.prune(to_prune_proportion))

    def freeze_weights(self) -> None:
        """
        Pushes the pruned weights to the next layer.
        """
        self._pn_apply(lambda x: x.freeze_weights())
        self._subset_count += 1

    def train_uncommitted(self):
        """
        Activates the subsets given subsets making them visible. The remaining
        capacity is mutable.
        """
        self._pn_apply(lambda x: x.train_uncommitted())
        self._active_task_id.fill_(self._subset_count)

    def activate_task(self, task_id: int):
        """
        Activates the given subsets in the layer making them visible. The
        remaining capacity is mutable.
        """
        if self._active_task_id.eq(task_id).all():
            return
        else:
            self._active_task_id.fill_(task_id)
            self._pn_apply(lambda x: x.activate_task(task_id))

    def task_count(self) -> int:
        return int(self._subset_count)

    def forward(self, input: Tensor, task_id: Tensor) -> Tensor:
        task_id_ = task_id[0]
        assert task_id.eq(task_id_).all(), "All task ids must be the same"
        self.activate_task(min(task_id_, self._subset_count))
        return super().forward(input)


class PackNetPlugin(BaseSGDPlugin):
    """A plugin calling PackNet's pruning and freezing procedures at the
    appropriate times. This plugin can only be used with `PackNet` models.
    """

    def __init__(
        self,
        post_prune_epochs: int,
        prune_proportion: float = 0.5,
    ):
        super().__init__()
        self.post_prune_epochs = post_prune_epochs
        self.total_epochs: int | None = None
        self.prune_proportion = prune_proportion
        assert 0 <= self.prune_proportion <= 1, (
            f"`prune_proportion` must be between 0 and 1, got "
            f"{self.prune_proportion}"
        )

    def before_training(self, strategy: "BaseSGDTemplate", *args, **kwargs):
        assert isinstance(
            strategy, BaseSGDTemplate
        ), "Strategy must be a `BaseSGDTemplate` or derived class."

        if not isinstance(strategy.model, PackNet):
            raise ValueError(
                f"`PackNetPlugin` can only be used with a `PackNet` model, "
                f"got {type(strategy.model)}. Try wrapping your model with "
                "`PackNet` before using this plugin."
            )

        if not hasattr(strategy, "train_epochs"):
            raise ValueError(
                "`PackNetPlugin` can only be used with a `BaseStrategy` that "
                "has a `train_epochs` attribute."
            )

        # Check the scenario has enough epochs for the post-pruning phase
        self.total_epochs = strategy.train_epochs
        if self.post_prune_epochs >= self.total_epochs:
            raise ValueError(
                f"`PackNetPlugin` can only be used with a `BaseStrategy`"
                "that has a `train_epochs` attribute greater than "
                f"{self.post_prune_epochs}. "
                f"Strategy has only {self.total_epochs} training epochs."
            )

    def before_training_epoch(self, strategy: "BaseSGDTemplate", *args, **kwargs):
        """When the initial training phase is over, prune the model and
        transition to the post-pruning phase.
        """
        epoch = strategy.clock.train_exp_epochs
        model = self._get_model(strategy)

        if epoch == (self.total_epochs - self.post_prune_epochs):
            model.prune(self.prune_proportion)

    def after_training_exp(self, strategy: "Template", *args, **kwargs):
        """After each experience, commit the model so that the next experience
        does not interfere with the previous one.
        """
        model = self._get_model(strategy)
        model.freeze_weights()

    def before_training_exp(self, strategy: "Template", *args, **kwargs):
        """Before each experience, setup to train the uncommitted layers
        during the training phase.
        """
        model = self._get_model(strategy)
        model.train_uncommitted()

    def _get_model(self, strategy: "BaseSGDTemplate"):
        """Get the model from the strategy."""
        model = strategy.model
        assert isinstance(model, PackNet), "Model must be a `PackNet`"
        return model
