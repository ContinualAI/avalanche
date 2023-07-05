import sys
import warnings
from collections import defaultdict
from typing import Iterable, Sequence, Optional, TypeVar, Union, List

import torch
from torch.nn import Module

from avalanche.benchmarks import CLExperience, CLStream
from avalanche.core import BasePlugin
from avalanche.distributed.distributed_helper import DistributedHelper
from avalanche.training.templates.strategy_mixin_protocol import BaseStrategyProtocol
from avalanche.training.utils import trigger_plugins


TExperienceType = TypeVar("TExperienceType", bound=CLExperience)
TPluginType = TypeVar("TPluginType", bound=BasePlugin, contravariant=True)


class BaseTemplate(BaseStrategyProtocol[TExperienceType]):
    """Base class for continual learning skeletons.

    **Training loop**
    The training loop is organized as follows::

        train
            train_exp  # for each experience

    **Evaluation loop**
    The evaluation loop is organized as follows::

        eval
            eval_exp  # for each experience

    """

    # we need this only for type checking
    PLUGIN_CLASS = BasePlugin

    def __init__(
        self,
        model: Module,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[Sequence[BasePlugin]] = None,
    ):
        super().__init__()
        """Init."""

        self.model: Module = model
        """ PyTorch model. """

        if device is None:
            warnings.warn(
                "When instantiating a strategy, please pass a non-None device."
            )
            device = "cpu"

        self.device = torch.device(device)
        """ PyTorch device where the model will be allocated. """

        self.plugins: List[BasePlugin] = [] if plugins is None else list(plugins)
        """ List of `SupervisedPlugin`s. """

        # check plugin compatibility
        self._check_plugin_compatibility()

        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################
        self.experience: Optional[TExperienceType] = None
        """ Current experience. """

        self.is_training: bool = False
        """ True if the strategy is in training mode. """

        self.current_eval_stream: Iterable[TExperienceType] = []
        """ Current evaluation stream. """

        self._distributed_check: bool = False
        """
        Internal flag used to verify the support for distributed
        training only once.
        """

        ###################################################################
        # Other variables #
        ###################################################################
        self._eval_streams: Optional[List[List[CLExperience]]] = None

    @property
    def is_eval(self):
        """True if the strategy is in evaluation mode."""
        return not self.is_training

    def train(
        self,
        experiences: Union[TExperienceType, Iterable[TExperienceType]],
        eval_streams: Optional[
            Sequence[Union[TExperienceType, Iterable[TExperienceType]]]
        ] = None,
        **kwargs,
    ):
        """Training loop.

        If experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.
        It returns a dictionary with last recorded value for each metric.

        :param experiences: single Experience or sequence.
        :param eval_streams: sequence of streams for evaluation.
            If None: use training experiences for evaluation.
            Use [] if you do not want to evaluate during training.
            Experiences in `eval_streams` are grouped by stream name
            when calling `eval`. If you use multiple streams, they must
            have different names.
        """
        if not self._distributed_check:
            # Checks if the strategy elements are compatible with
            # distributed training
            self._check_distributed_training_compatibility()
            self._distributed_check = True

        self.is_training = True
        self._stop_training = False

        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        experiences_list: Iterable[
            TExperienceType
        ] = _experiences_parameter_as_iterable(experiences)

        if eval_streams is None:
            eval_streams = [experiences_list]

        self._eval_streams = _group_experiences_by_stream(eval_streams)

        self._before_training(**kwargs)

        for self.experience in experiences_list:
            self._before_training_exp(**kwargs)
            self._train_exp(self.experience, eval_streams, **kwargs)
            self._after_training_exp(**kwargs)
        self._after_training(**kwargs)
        self._train_cleanup()

    def _train_cleanup(self):
        # reset _eval_streams for faster serialization
        self._eval_streams = None
        self.experience = None

    def _train_exp(self, experience: CLExperience, eval_streams, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def eval(
        self,
        experiences: Union[TExperienceType, CLStream[TExperienceType]],
        **kwargs,
    ):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        if not self._distributed_check:
            # Checks if the strategy elements are compatible with
            # distributed training
            self._check_distributed_training_compatibility()
            self._distributed_check = True

        # eval can be called inside the train method.
        # Save the shared state here to restore before returning.
        prev_train_state = self._save_train_state()
        self.is_training = False
        self.model.eval()

        experiences_list: Iterable[
            TExperienceType
        ] = _experiences_parameter_as_iterable(experiences)
        self.current_eval_stream = experiences_list

        self._before_eval(**kwargs)
        for self.experience in experiences_list:
            self._before_eval_exp(**kwargs)
            self._eval_exp(**kwargs)
            self._after_eval_exp(**kwargs)

        self._after_eval(**kwargs)
        self._eval_cleanup()

        # restore previous shared state.
        self._load_train_state(prev_train_state)

    def _eval_cleanup(self):
        # reset for faster serialization
        self.current_eval_stream = []
        self.experience = None

    def _eval_exp(self, **kwargs):
        raise NotImplementedError()

    def _save_train_state(self):
        """Save the training state, which may be modified by the eval loop.

        TODO: we probably need a better way to do this.
        """
        # save each layer's training mode, to restore it later
        _prev_model_training_modes = {}
        for name, layer in self.model.named_modules():
            _prev_model_training_modes[name] = layer.training

        _prev_state = {
            "experience": self.experience,
            "is_training": self.is_training,
            "model_training_mode": _prev_model_training_modes,
        }
        return _prev_state

    def _load_train_state(self, prev_state):
        # restore train-state variables and training mode.
        self.experience = prev_state["experience"]
        self.is_training = prev_state["is_training"]

        # restore each layer's training mode to original
        prev_training_modes = prev_state["model_training_mode"]
        for name, layer in self.model.named_modules():
            try:
                prev_mode = prev_training_modes[name]
                layer.train(mode=prev_mode)
            except KeyError:
                # Unknown parameter, probably added during the eval
                # model's adaptation. We set it to train mode.
                layer.train()

    def _check_plugin_compatibility(self):
        """Check that the list of plugins is compatible with the template.

        This means checking that each plugin impements a subset of the
        supported callbacks.
        """
        # TODO: ideally we would like to check the argument's type to check
        #  that it's a supertype of the template.
        # I don't know if it's possible to do it in Python.
        ps = self.plugins

        def get_plugins_from_object(obj):
            def is_callback(x):
                return x.startswith("before") or x.startswith("after")

            return filter(is_callback, dir(obj))

        cb_supported = set(get_plugins_from_object(self.PLUGIN_CLASS))
        for p in ps:
            cb_p = set(get_plugins_from_object(p))

            if not cb_p.issubset(cb_supported):
                warnings.warn(
                    f"Plugin {p} implements incompatible callbacks for template"
                    f" {self}. This may result in errors. Incompatible "
                    f"callbacks: {cb_p - cb_supported}",
                )
                return

    def _check_distributed_training_compatibility(self):
        """
        Check if strategy elements (plugins, ...) are compatible with
        distributed training.
        This check does nothing if not training in distributed mode.
        """
        if not DistributedHelper.is_distributed:
            return True

        unsupported_plugins = []
        for plugin in self.plugins:
            if not getattr(plugin, "supports_distributed", False):
                unsupported_plugins.append(plugin)

        if len(unsupported_plugins) > 0:
            warnings.warn(
                "You are using plugins that are not compatible"
                "with distributed training:"
            )
            for plugin in unsupported_plugins:
                print(type(plugin), file=sys.stderr)

        return len(unsupported_plugins) == 0

    #########################################################
    # Plugin Triggers                                       #
    #########################################################

    def _before_training_exp(self, **kwargs):
        trigger_plugins(self, "before_training_exp", **kwargs)

    def _after_training_exp(self, **kwargs):
        trigger_plugins(self, "after_training_exp", **kwargs)

    def _before_training(self, **kwargs):
        trigger_plugins(self, "before_training", **kwargs)

    def _after_training(self, **kwargs):
        trigger_plugins(self, "after_training", **kwargs)

    def _before_eval(self, **kwargs):
        trigger_plugins(self, "before_eval", **kwargs)

    def _after_eval(self, **kwargs):
        trigger_plugins(self, "after_eval", **kwargs)

    def _before_eval_exp(self, **kwargs):
        trigger_plugins(self, "before_eval_exp", **kwargs)

    def _after_eval_exp(self, **kwargs):
        trigger_plugins(self, "after_eval_exp", **kwargs)


def _group_experiences_by_stream(
    eval_streams: Iterable[Union[Iterable[CLExperience], CLExperience]]
) -> List[List[CLExperience]]:
    exps: List[CLExperience] = []
    # First, we unpack the list of experiences.
    for exp in eval_streams:
        if isinstance(exp, Iterable):
            exps.extend(exp)
        else:
            exps.append(exp)
    # Then, we group them by stream.
    exps_by_stream = defaultdict(list)
    for exp in exps:
        sname = exp.origin_stream.name
        exps_by_stream[sname].append(exp)
    # Finally, we return a list of lists.
    return list(list(exps_by_stream.values()))


def _experiences_parameter_as_iterable(
    experiences: Union[Iterable[TExperienceType], TExperienceType]
) -> Iterable[TExperienceType]:
    if isinstance(experiences, Iterable):
        return experiences
    else:
        return [experiences]


__all__ = ["BaseTemplate"]
