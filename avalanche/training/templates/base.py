import warnings
from typing import Sequence, Optional, Union

import torch
from torch.nn import Module

from avalanche.benchmarks import Experience
from avalanche.core import BasePlugin
from avalanche.training.utils import trigger_plugins


class BaseTemplate:
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
        device="cpu",
        plugins: Optional[Sequence[PLUGIN_CLASS]] = None,
    ):
        """Init."""

        self.model: Module = model
        """ PyTorch model. """

        self.device = device
        """ PyTorch device where the model will be allocated. """

        self.plugins = [] if plugins is None else plugins
        """ List of `SupervisedPlugin`s. """

        # check plugin compatibility
        self.__check_plugin_compatibility()

        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################
        self.experience = None
        """ Current experience. """

        self.is_training: bool = False
        """ True if the strategy is in training mode. """

        self.current_eval_stream = None
        """ Current evaluation stream. """

    @property
    def is_eval(self):
        """True if the strategy is in evaluation mode."""
        return not self.is_training

    def train(
        self,
        experiences: Union[Experience, Sequence[Experience]],
        eval_streams: Optional[
            Sequence[Union[Experience, Sequence[Experience]]]
        ] = None,
        **kwargs,
    ):
        """Training loop.

        If experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.
        It returns a dictionary with last recorded value for each metric.

        :param experiences: single Experience or sequence.
        :param eval_streams: list of streams for evaluation.
            If None: use training experiences for evaluation.
            Use [] if you do not want to evaluate during training.
        """
        self.is_training = True
        self._stop_training = False

        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        if not isinstance(experiences, Sequence):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]
        self._eval_streams = eval_streams

        self._before_training(**kwargs)

        for self.experience in experiences:
            self._before_training_exp(**kwargs)
            self._train_exp(self.experience, eval_streams, **kwargs)
            self._after_training_exp(**kwargs)
        self._after_training(**kwargs)

    def _train_exp(self, experience: Experience, eval_streams, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def eval(self, exp_list: Union[Experience, Sequence[Experience]],
             **kwargs):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        # eval can be called inside the train method.
        # Save the shared state here to restore before returning.
        prev_train_state = self._save_train_state()
        self.is_training = False
        self.model.eval()

        if not isinstance(exp_list, Sequence):
            exp_list = [exp_list]
        self.current_eval_stream = exp_list

        self._before_eval(**kwargs)
        for self.experience in exp_list:
            self._before_eval_exp(**kwargs)
            self._eval_exp(**kwargs)
            self._after_eval_exp(**kwargs)

        self._after_eval(**kwargs)

        # restore previous shared state.
        self._load_train_state(prev_train_state)

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

    def __check_plugin_compatibility(self):
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
                return x.startswith("before") or \
                       x.startswith("after")
            return filter(is_callback, dir(obj))

        cb_supported = set(get_plugins_from_object(self.PLUGIN_CLASS))
        for p in ps:
            cb_p = set(get_plugins_from_object(p))

            if not cb_p.issubset(cb_supported):
                warnings.warn(
                    f"Plugin {p} implements incompatible callbacks for template"
                    f" {self}. This may result in errors."
                )
                return

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
