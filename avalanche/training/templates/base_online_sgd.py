from typing import Iterable, Sequence, Optional, Union, List

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.benchmarks import CLExperience, CLStream
from avalanche.core import BaseSGDPlugin
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.clock import Clock
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.base import BaseTemplate, ExpSequence

from typing import TYPE_CHECKING

from avalanche.training.utils import trigger_plugins

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class BaseOnlineSGDTemplate(BaseTemplate):
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

    PLUGIN_CLASS = BaseSGDPlugin

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        train_mb_size: int = 1,
        train_passes: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[List["SupervisedPlugin"]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        peval_mode="experience",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param train_mb_size: mini-batch size for training.
        :param train_passes: number of training passes.
        :param eval_mb_size: mini-batch size for eval.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean
            that `eval` is called every `eval_every` experience and at the end
            of the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experiences or iterations (Default='experience').
        """
        super().__init__(model=model, device=device, plugins=plugins)

        self.optimizer: Optimizer = optimizer
        """ PyTorch optimizer. """

        self.train_passes: int = train_passes
        """ Number of training passes. """

        self.train_mb_size: int = train_mb_size
        """ Training mini-batch size. """

        self.eval_mb_size: int = (
            train_mb_size if eval_mb_size is None else eval_mb_size
        )
        """ Eval mini-batch size. """

        if evaluator is None:
            evaluator = EvaluationPlugin()
        self.plugins.append(evaluator)
        self.evaluator = evaluator
        """ EvaluationPlugin used for logging and metric computations. """

        # Configure periodic evaluation.
        assert peval_mode in {"experience", "iteration"}
        self.eval_every = eval_every
        peval = PeriodicEval(eval_every, peval_mode)
        self.plugins.append(peval)

        self.clock = Clock()
        """ Incremental counters for strategy events. """
        # WARNING: Clock needs to be the last plugin, otherwise
        # counters will be wrong for plugins called after it.
        self.plugins.append(self.clock)

        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################

        self.dataloader = None
        """ Dataloader. """

        self.mbatch = None
        """ Current mini-batch. """

        self.mb_output = None
        """ Model's output computed on the current mini-batch. """

        self.loss = None
        """ Loss of the current mini-batch. """

        self._stop_training = False

    def train(self,
              experiences: Union[CLExperience,
                                 ExpSequence],
              eval_streams: Optional[Sequence[Union[CLExperience,
                                                    ExpSequence]]] = None,
              **kwargs):
        super().train(experiences, eval_streams, **kwargs)
        return self.evaluator.get_last_metrics()

    @torch.no_grad()
    def eval(self, exp_list: Union[CLExperience, CLStream], **kwargs):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        super().eval(exp_list, **kwargs)
        return self.evaluator.get_last_metrics()

    def _before_training_exp(self, **kwargs):
        self.make_train_dataloader(**kwargs)
        # Model Adaptation (e.g. freeze/add new units)

        # If strategy has access to the task boundaries, and the current
        # sub-experience is the first sub-experience in the online (sub-)stream,
        # then adapt the model with the full origin experience:
        if self.experience.access_task_boundaries:
            if self.experience.is_first_subexp:
                self.model = self.model_adaptation()
                self.make_optimizer()
        # Otherwise, adapt to the current sub-experience:
        else:
            self.model = self.model_adaptation()
            self.make_optimizer()

        super()._before_training_exp(**kwargs)

    def _train_exp(
        self, experience: CLExperience, eval_streams=None, **kwargs
    ):
        """Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Iterable):
                eval_streams[i] = [exp]

        self.training_pass(**kwargs)

    def _before_eval_exp(self, **kwargs):
        self.make_eval_dataloader(**kwargs)
        # Model Adaptation (e.g. freeze/add new units)
        self.model = self.model_adaptation()
        super()._before_eval_exp(**kwargs)

    def _eval_exp(self, **kwargs):
        self.eval_epoch(**kwargs)

    def make_train_dataloader(self, **kwargs):
        """Assign dataloader to self.dataloader."""
        raise NotImplementedError()

    def make_eval_dataloader(self, **kwargs):
        """Assign dataloader to self.dataloader."""
        raise NotImplementedError()

    def make_optimizer(self, **kwargs):
        """Optimizer initialization."""
        raise NotImplementedError()

    def criterion(self):
        """Compute loss function."""
        raise NotImplementedError()

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        raise NotImplementedError()

    def model_adaptation(self, model=None):
        """Adapts the model to the current experience."""
        raise NotImplementedError()

    def stop_training(self):
        """Signals to stop training at the next iteration."""
        self._stop_training = True

    def training_pass(self, **kwargs):
        """Training pass.

        :param kwargs:
        :return:
        """
        for self.pass_itr in range(self.train_passes):
            for self.mbatch in self.dataloader:
                if self._stop_training:
                    break

                self._unpack_minibatch()
                self._before_training_iteration(**kwargs)

                self.optimizer.zero_grad()
                self.loss = 0

                # Forward
                self._before_forward(**kwargs)
                self.mb_output = self.forward()
                self._after_forward(**kwargs)

                # Loss & Backward
                self.loss += self.criterion()

                self._before_backward(**kwargs)
                self.backward()
                self._after_backward(**kwargs)

                # Optimization step
                self._before_update(**kwargs)
                self.optimizer_step()
                self._after_update(**kwargs)

                self._after_training_iteration(**kwargs)

    def backward(self):
        """Run the backward pass."""
        self.loss.backward()

    def optimizer_step(self):
        """Execute the optimizer step (weights update)."""
        self.optimizer.step()

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_eval_forward(**kwargs)
            self.loss = self.criterion()

            self._after_eval_iteration(**kwargs)

    def _unpack_minibatch(self):
        """Move to device"""
        for i in range(len(self.mbatch)):
            self.mbatch[i] = self.mbatch[i].to(self.device)

    #########################################################
    # Plugin Triggers                                       #
    #########################################################

    def _before_training_iteration(self, **kwargs):
        trigger_plugins(self, "before_training_iteration", **kwargs)

    def _before_forward(self, **kwargs):
        trigger_plugins(self, "before_forward", **kwargs)

    def _after_forward(self, **kwargs):
        trigger_plugins(self, "after_forward", **kwargs)

    def _before_backward(self, **kwargs):
        trigger_plugins(self, "before_backward", **kwargs)

    def _after_backward(self, **kwargs):
        trigger_plugins(self, "after_backward", **kwargs)

    def _after_training_iteration(self, **kwargs):
        trigger_plugins(self, "after_training_iteration", **kwargs)

    def _before_update(self, **kwargs):
        trigger_plugins(self, "before_update", **kwargs)

    def _after_update(self, **kwargs):
        trigger_plugins(self, "after_update", **kwargs)

    def _before_eval_iteration(self, **kwargs):
        trigger_plugins(self, "before_eval_iteration", **kwargs)

    def _before_eval_forward(self, **kwargs):
        trigger_plugins(self, "before_eval_forward", **kwargs)

    def _after_eval_forward(self, **kwargs):
        trigger_plugins(self, "after_eval_forward", **kwargs)

    def _after_eval_iteration(self, **kwargs):
        trigger_plugins(self, "after_eval_iteration", **kwargs)


class PeriodicEval(SupervisedPlugin):
    """Schedules periodic evaluation during training.

    This plugin is automatically configured and added by the BaseTemplate.
    """

    def __init__(self, eval_every=-1, peval_mode="experience",
                 do_initial=True):
        """Init.

        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean
            that `eval` is called every `eval_every` experience and at the
            end of the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experience or iterations
            (Default='experience').
        :param do_initial: whether to evaluate before each `train` call.
            Occasionally needed becuase some metrics need to know the
            accuracy before training.
        """
        super().__init__()
        assert peval_mode in {"experience", "iteration"}
        self.eval_every = eval_every
        self.peval_mode = peval_mode
        self.do_initial = do_initial and eval_every > -1
        self.do_final = None
        self._is_eval_updated = False

    def before_training(self, strategy, **kwargs):
        """Eval before each learning experience.

        Occasionally needed because some metrics need the accuracy before
        training.
        """
        if self.do_initial:
            self._peval(strategy, **kwargs)

    def _peval(self, strategy, **kwargs):
        for el in strategy._eval_streams:
            strategy.eval(el, **kwargs)

    def _maybe_peval(self, strategy, counter, **kwargs):
        if self.eval_every > 0 and counter % self.eval_every == 0:
            self._peval(strategy, **kwargs)

    def after_training_exp(self, strategy: "BaseOnlineSGDTemplate", **kwargs):
        """Periodic eval controlled by `self.eval_every` and
        `self.peval_mode`."""
        if self.peval_mode == "experience":
            self._maybe_peval(strategy, strategy.clock.train_exp_counter,
                              **kwargs)

    def after_training_iteration(self, strategy: "BaseOnlineSGDTemplate",
                                 **kwargs):
        """Periodic eval controlled by `self.eval_every` and
        `self.peval_mode`."""
        if self.peval_mode == "iteration":
            self._maybe_peval(strategy, strategy.clock.train_exp_iterations,
                              **kwargs)
