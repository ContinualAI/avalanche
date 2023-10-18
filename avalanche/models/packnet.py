"""
Implements (Mallya & Lazebnik, 2018) PackNet algorithm for fixed-network
parameter isolation. PackNet is a task-incremental learning algorithm that
uses task identities during testing.

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
from typing import Union
from avalanche.training.templates.base_sgd import BaseSGDTemplate


class PackNetModule(ABC, nn.Module):
    """Defines the interface for implementing PackNet compatible PyTorch modules.

    The core idea of PackNet is to build a single network containing multiple
    task-specific subsets. Each subset builds on the previous subset and
    therefore shares parameters with the previous subset. But only the
    parameters not shared with the previous subset are mutable. This allows
    PackNet to isolate parameters for each task.

    Caution should be taken when optimizers with momentum are used, since they
    can cause parameters to be modified even when no gradient exists.

    PackNet has internal state that changes its behaviour and this class is
    responsible for ensuring that no invalid state transitions occur. When
    an invalid state transition occurs a `StateError` is thrown.
    """

    class State(Enum):
        """PackNet requires a procedure to be followed and we model this with
        the following states.
        """

        TRAINING = 0
        """The PackNet module is training all of the unfrozen capacity"""
        POST_PRUNE = 1
        """The PackNet module is training only on the unpruned parameters that
        will be frozen next"""
        EVAL = 2
        """Activate a task-specific subset and mask the remaining parameters.
        This state freezes all parameters."""

    class StateError(RuntimeError):
        """An invalid state transition occured"""

    def __init__(self) -> None:
        super().__init__()
        init_state = self.State.TRAINING
        self._state: Tensor
        """The current state of the PackNet"""
        self._active_task: Tensor
        """The id of the task that is currently active"""
        self._task_count: Tensor
        """The number of tasks that have been trained"""
        self.register_buffer("_state", torch.tensor(init_state.value, dtype=torch.int))
        self.register_buffer("_active_task", torch.tensor(0, dtype=torch.int))
        self.register_buffer("_task_count", torch.tensor(0, dtype=torch.int))

    def prune(self, prune_proportion: float):
        """Prune a proportion of the unfrozen parameters from the module.

        The pruned parameters will be reused for the next task, while the
        remaining will be fine-tuned further on the current task, in the
        post-pruning phase.

        Prune may only be called when PackNet is in the `State.TRAINING` state.
        Prune will move PackNet to the `State.POST_PRUNE` state.

        :param prune_proportion: A proportion of the prunable parameters to
            prune. Must be between 0 and 1.
        """
        if not 0 <= prune_proportion <= 1:
            raise ValueError(
                f"`prune_proportion` must be between 0 and 1, got "
                f"{prune_proportion}"
            )
        self._state_guard(
            self.prune.__name__, [self.State.TRAINING], self.State.POST_PRUNE
        )
        self._prune(prune_proportion)

    def freeze_pruned(self):
        """
        Freeze the pruned parameters, commiting them to become immutable.

        This prevents subsequent tasks from affecting any parameters associated
        with this task.

        This function can only be called when PackNet is in the
        `State.POST_PRUNE` state. It will then move PackNet to the `State.EVAL`
        state.
        """
        self._state_guard(
            self.freeze_pruned.__name__, [self.State.POST_PRUNE], self.State.EVAL
        )
        self._task_count += 1
        self._freeze_pruned()

    def activate_task(self, task_id: int):
        """Activates a task-specific subset of PackNet.

        When `task_id` is the active task, the active task can be trained using
        the remaining capacity. Otherwise, all parameters are frozen and
        the active task cannot be trained.

        This function can only be called when PackNet is in the `State.EVAL`,
        `State.TRAINING`, or `State.POST_PRUNE` state. Moving PackNet to the
        `State.EVAL` state if the `task_id` is not the active task. Otherwise,
        PackNet remains in the same state.

        :param task_id: The task to activate. Must be between 0 and the number
            of tasks seen so far.
        """
        if not (0 <= task_id <= self.task_count):
            raise ValueError(
                f"`task_id` must be between 0 and {self.task_count}, " f"got {task_id}"
            )
        if task_id != self.task_count:
            next_state = self.State.EVAL
        elif self.state == self.State.POST_PRUNE:
            next_state = self.State.POST_PRUNE
        else:
            next_state = self.State.TRAINING
        # Stop if the task is already active
        if task_id == self.active_task and self.state == next_state:
            return

        self._state_guard(
            self.activate_task.__name__,
            [self.State.EVAL, self.State.TRAINING],
            next_state,
        )
        self._activate_task(task_id)
        self._active_task.fill_(task_id)

    @abstractmethod
    def _prune(self, prune_proportion: float) -> None:
        """Implementation of `prune` once the state has been checked"""

    @abstractmethod
    def _freeze_pruned(self) -> None:
        """Implementation of `freeze_pruned` once the state has been checked"""

    @abstractmethod
    def _activate_task(self, task_id: int) -> None:
        """Implementation of `activate_task` once the state has been checked"""

    @property
    def active_task(self) -> int:
        """Returns the id of the task that is currently active.

        :return: The id of the task that is currently active.
        """
        return int(self._active_task.item())

    @property
    def task_count(self) -> int:
        """Counts the number of task-specific subsets in PackNet.

        :return: The number of task-specific subsets
        """
        return int(self._task_count.item())

    @property
    def state(self) -> State:
        return self.State(self._state.item())

    def _state_guard(
        self,
        func_name: str,
        previous: t.Sequence[State],
        next: State,
    ):
        """Ensure that the state is in the correct state and transition to the
        next correct state.
        """
        if self.state not in previous:
            previous_str = ", ".join([str(x) for x in previous])
            raise self.StateError(
                f"Calling `{func_name}` is only valid for `{previous_str}` "
                f"instead PackNet was in the `{self.state}` state"
            )
        self._state.fill_(next.value)


class WeightAndBiasPackNetModule(PackNetModule):
    """A PackNet module that has a weight and bias. This can be used to wrap
    many PyTorch modules such as `nn.Linear`, `nn.Conv2d` and `nn.ConvTranspose2d`
    """

    def __init__(self, wrappee: nn.Module) -> None:
        super().__init__()
        self.wrappee: nn.Module = wrappee
        # The following attributes are used to check that the wrappee is
        # compatible
        if not hasattr(wrappee, "weight") or not isinstance(
            wrappee.weight, nn.Parameter
        ):
            raise ValueError(f"weight must be defined in {wrappee}")
        self.has_bias = hasattr(wrappee, "bias") and isinstance(
            wrappee.weight, nn.Parameter
        )

        self.PRUNED_CODE: Tensor
        """Value used to code for a pruned weight, during the post-pruning phase"""
        self.register_buffer("PRUNED_CODE", torch.tensor(255, dtype=torch.int))

        self.task_index: Tensor
        """Tracks which task each weight belongs to. Of the same shape as the
        weight tensor."""
        self.register_buffer(
            "task_index",
            torch.ones_like(self.wrappee.weight).byte() * self._task_count,
        )

        self.visible_mask: Tensor
        """Mask of weights that are visible. Can be computed from `task_index`"""
        self.register_buffer(
            "visible_mask", torch.ones_like(self.task_index, dtype=torch.bool)
        )

        self.unfrozen_mask: Tensor
        """Mask of weights that are mutable. Can be computed from `task_index`"""
        self.register_buffer(
            "unfrozen_mask", torch.ones_like(self.task_index, dtype=torch.bool)
        )

        wrappee.weight.register_hook(self._remove_gradient_hook)

    @property
    def pruned_mask(self) -> Tensor:
        """Return a mask of weights that have been pruned"""
        return self.task_index.eq(self.PRUNED_CODE)

    def _prune(self, prune_proportion: float):
        ranked = self._rank_prunable()
        prune_count = int(len(ranked) * prune_proportion)
        self._prune_weights(ranked[:prune_count])
        self.unfrozen_mask = self.task_index.eq(self._task_count)

    def available_weights(self) -> Tensor:
        return self.visible_mask * self.wrappee.weight

    def _activate_task(self, task_id: int):
        self.visible_mask.zero_()
        self.unfrozen_mask.zero_()
        self.visible_mask = self.task_index.less_equal(task_id)
        if task_id == self.task_count:
            self.unfrozen_mask = self.task_index.eq(task_id)

    def _freeze_pruned(self):
        self.task_index[self.pruned_mask] = self.task_count
        self.unfrozen_mask.zero_()
        if self.has_bias:
            self.wrappee.bias.requires_grad = False

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def _remove_gradient_hook(self, grad: Tensor) -> Tensor:
        """Gradients that are frozen are zeroed out. Preventing them from
        being modifed after they have been frozen."""
        return grad * self.unfrozen_mask

    def _rank_prunable(self) -> Tensor:
        """
        Returns a 1D tensor of the weights ranked based on their absolute value.
        Sorted to be in ascending order.
        """
        # "We use the simple heuristic to quantify the importance of the
        # weights using their absolute value." (Han et al., 2017)
        # Han, S., Pool, J., Narang, S., Mao, H., Gong, E., Tang, S., Elsen, E.,
        # Vajda, P., Paluri, M., Tran, J., Catanzaro, B., & Dally, W. J. (2017).
        # DSD: Dense-Sparse-Dense Training for Deep Neural Networks.
        # ArXiv:1607.04381 [Cs]. http://arxiv.org/abs/1607.04381
        importance = self.wrappee.weight.abs()
        un_prunable = ~self.unfrozen_mask
        # Mark un-prunable weights using -1.0 so they can be cutout after sort
        importance[un_prunable] = -1.0
        rank = torch.argsort(importance.flatten())
        # Cut out un-prunable weights
        return rank[un_prunable.count_nonzero() :]

    def _prune_weights(self, indices: Tensor):
        """Given a list of indices, prune the weights at those indices.

        Pruning simply marks the weight as pruned in the `task_index` and
        makes the weight invisible in the `visible_mask`.

        :param indices: A 1D tensor of indices to prune
        """
        self.task_index.flatten()[indices] = self.PRUNED_CODE.item()
        self.visible_mask.flatten()[indices] = False


class _PnLinear(WeightAndBiasPackNetModule):
    """A decorator for `nn.Linear` module making it PackNet compatible."""

    def __init__(self, wrappee: nn.Linear) -> None:
        self.wrappee: nn.Linear
        super().__init__(wrappee)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.available_weights(), self.wrappee.bias)


class _PnConvNd(WeightAndBiasPackNetModule):
    """A decorator for `nn.Linear` module making it PackNet compatible."""

    def __init__(self, wrappee: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]) -> None:
        super().__init__(wrappee)

    def forward(self, input: Tensor) -> Tensor:
        return self.wrappee._conv_forward(
            input, self.available_weights(), self.wrappee.bias
        )


class _PnConvTransposedNd(WeightAndBiasPackNetModule):
    def __init__(
        self, wrappee: Union[nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    ) -> None:
        super().__init__(wrappee)

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


class PackNetModel(PackNetModule, MultiTaskModule):
    """
    PackNet implements the PackNet algorithm for parameter isolation. It
    is designed to automatically upgrade most models to support PackNet.
    But because of the nature of the strategy, it is not possible to use it
    with every model or PyTorch module. Furthermore, PackNet not everything
    has been implemented yet. Here are some basic guidelines:

     - Stateless modules like :class:`torch.nn.ReLU`, :class:`torch.nn.Flatten`,
        or `torch.nn.Dropout` should work fine.
     - Many normalization layers currently do not work.
     - Supports: :class:`nn.Linear`, :class:`nn.Conv1d`, :class:`nn.Conv2d`,
        :class:`nn.Conv3d`, :class:`nn.ConvTranspose1d`, :class:`nn.ConvTranspose2d`,
        :class:`nn.ConvTranspose3d`
     - If you want to use a custom module with state or parameters, ensure it
        implements :class:`PackNetModule`.


    Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a
    Single Network by Iterative Pruning. 2018 IEEE/CVF Conference on Computer
    Vision and Pattern Recognition, 7765-7773.
    https://doi.org/10.1109/CVPR.2018.00810
    """

    @staticmethod
    def wrap(wrappee: nn.Module):
        """Upgrade a PyTorch module and all of its submodules to be PackNet
        compatible. This is a recursive function that will wrap all submodules
        in a PackNet compatible module.

        :param wrappee: The module to wrap
        :raises ValueError: If the module is not supported
        :return: A PackNet compatible module
        """
        # Weight norm is not supported
        if hasattr(wrappee, "weight_g") and hasattr(wrappee, "weight_v"):
            raise ValueError("PackNet does not support weight norm")
        # Other norms are not supported
        if hasattr(wrappee, "running_mean") or hasattr(wrappee, "running_var"):
            raise ValueError(
                "The PackNet implementation does not yet support norms "
                f"{wrappee.__class__.__name__}"
            )

        # Recursive cases
        if isinstance(wrappee, PackNetModule):
            return wrappee
        elif isinstance(wrappee, nn.Linear):
            return _PnLinear(wrappee)
        elif isinstance(wrappee, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return _PnConvNd(wrappee)
        elif isinstance(
            wrappee, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            return _PnConvTransposedNd(wrappee)
        elif isinstance(wrappee, nn.Sequential):
            # Wrap each submodule
            for i, x in enumerate(wrappee):
                wrappee[i] = PackNetModel.wrap(x)
            return wrappee

        # If the module has parameters and has not been wrapped yet, then it is
        # not supported
        if len(list(wrappee.parameters(recurse=False))) != 0:
            raise ValueError(
                f"PackNet does not support the module {wrappee.__class__.__name__}"
            )

        for submodule_name, submodule in wrappee.named_children():
            setattr(wrappee, submodule_name, PackNetModel.wrap(submodule))
        return wrappee

    def __init__(self, wrappee: nn.Module) -> None:
        """Wrap a PyTorch module to make it PackNet compatible.

        :param wrappee: The module to wrap
        """
        super().__init__()
        self.wrappee: nn.Module = PackNetModel.wrap(wrappee)

    def _pn_apply(self, func: t.Callable[["PackNetModel"], None]):
        """Apply a function to all child PackNetModules

        :param func: The function to apply
        """

        @torch.no_grad()
        def __pn_apply(module):
            # Apply function to all child PackNetModule but not other
            # parent PackNet modules
            if isinstance(module, PackNetModule) and not isinstance(
                module, PackNetModel
            ):
                func(module)

        self.apply(__pn_apply)

    def _prune(self, to_prune_proportion: float):
        """Call `prune` on all child PackNetModules

        :param to_prune_proportion: The proportion of parameters to prune in
            each child PackNetModule
        """
        self._pn_apply(lambda x: x.prune(to_prune_proportion))

    def _freeze_pruned(self):
        """Call `freeze_pruned` on all child PackNetModules"""
        self._pn_apply(lambda x: x.freeze_pruned())

    def _activate_task(self, task_id: int):
        """Call `activate_task` on all child PackNetModules

        :param task_id: The task to activate
        """
        self._pn_apply(lambda x: x.activate_task(task_id))

    def forward(self, input: Tensor, task_id: Tensor) -> Tensor:
        task_id_ = task_id[0]
        assert task_id.eq(task_id_).all(), "All task ids must be the same"
        self.activate_task(min(task_id_, self.task_count))
        return self.wrappee.forward(input)


class PackNetPlugin(BaseSGDPlugin):
    """A plugin calling PackNet's pruning and freezing procedures at the
    appropriate times. This plugin can only be used with `PackNet` models.
    """

    def __init__(
        self,
        post_prune_epochs: int,
        prune_proportion: t.Union[float, t.Callable[[int], float], t.List[float]] = 0.5,
    ):
        """The PackNetPlugin calls PackNet's pruning and freezing procedures at
        the appropriate times.

        :param post_prune_epochs: The number of epochs to finetune the model
            after pruning the parameters. Must be less than the number of
            training epochs.
        :param prune_proportion: The proportion of parameters to prune
            during each task. Can be a float, a list of floats, or a function
            that takes the task id and returns a float. Each value must be
            between 0 and 1.
        """
        super().__init__()
        self.post_prune_epochs = post_prune_epochs
        self.total_epochs: Union[int, None] = None
        self.prune_proportion: t.Callable[[int], float] = prune_proportion

        if isinstance(prune_proportion, float):
            assert 0 <= self.prune_proportion <= 1, (
                f"`prune_proportion` must be between 0 and 1, got "
                f"{self.prune_proportion}"
            )
            self.prune_proportion = lambda _: prune_proportion
        elif isinstance(prune_proportion, list):
            assert all(0 <= x <= 1 for x in prune_proportion), (
                "all values in `prune_proportion` must be between 0 and 1,"
                f" got {prune_proportion}"
            )
            self.prune_proportion = lambda i: prune_proportion[i]
        else:
            self.prune_proportion = prune_proportion

    def before_training(self, strategy: "BaseSGDTemplate", *args, **kwargs):
        assert isinstance(
            strategy, BaseSGDTemplate
        ), "Strategy must be a `BaseSGDTemplate` or derived class."

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

    def before_training_exp(self, strategy: "BaseSGDTemplate", *args, **kwargs):
        # Reset the optimizer to prevent momentum from affecting the pruned
        # parameters
        strategy.optimizer = strategy.optimizer.__class__(
            strategy.model.parameters(), **strategy.optimizer.defaults
        )

    def before_training_epoch(self, strategy: "BaseSGDTemplate", *args, **kwargs):
        """When the initial training phase is over, prune the model and
        transition to the post-pruning phase.
        """
        epoch = strategy.clock.train_exp_epochs
        model = self._get_model(strategy)

        if epoch == (self.total_epochs - self.post_prune_epochs):
            model.prune(self.prune_proportion(strategy.clock.train_exp_counter))

    def after_training_exp(self, strategy: "Template", *args, **kwargs):
        """After each experience, commit the model so that the next experience
        does not interfere with the previous one.
        """
        model = self._get_model(strategy)
        model.freeze_pruned()

    def _get_model(self, strategy: "BaseSGDTemplate") -> PackNetModule:
        """Get the model from the strategy."""
        model = strategy.model
        if not isinstance(strategy.model, PackNetModule):
            raise ValueError(
                f"`PackNetPlugin` can only be used with a `PackNet` model, "
                f"got {type(strategy.model)}. Try wrapping your model with "
                "`PackNet` before using this plugin."
            )
        return model


def packnet_simple_mlp(
    num_classes=10,
    input_size=28 * 28,
    hidden_size=512,
    hidden_layers=1,
    drop_rate=0.5,
) -> PackNetModel:
    """
    Convenience function for creating a PackNet compatible :class:`SimpleMLP`
    model.

    :param num_classes: output size
    :param input_size: input size
    :param hidden_size: hidden layer size
    :param hidden_layers: number of hidden layers
    :param drop_rate: dropout rate. 0 to disable
    :return: A PackNet compatible model
    """
    return PackNetModel(
        SimpleMLP(num_classes, input_size, hidden_size, hidden_layers, drop_rate)
    )
