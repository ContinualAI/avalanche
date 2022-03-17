from typing import Sequence, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.benchmarks import Experience
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.clock import Clock
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.base import BaseTemplate

from typing import TYPE_CHECKING

from avalanche.training.utils import trigger_plugins

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class BaseSGDTemplate(BaseTemplate):
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

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
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
        super().__init__(model=model, device=device, plugins=plugins)

        self.optimizer: Optimizer = optimizer
        """ PyTorch optimizer. """

        self.train_epochs: int = train_epochs
        """ Number of training epochs. """

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
        assert peval_mode in {"epoch", "iteration"}
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

    def train(
        self,
        experiences: Union[Experience, Sequence[Experience]],
        eval_streams: Optional[
            Sequence[Union[Experience, Sequence[Experience]]]
        ] = None,
        **kwargs,
    ):
        super().train(experiences, eval_streams, **kwargs)
        return self.evaluator.get_last_metrics()

    @torch.no_grad()
    def eval(self, exp_list: Union[Experience, Sequence[Experience]], **kwargs):
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
        self.model = self.model_adaptation()
        self.make_optimizer()
        super()._before_training_exp(**kwargs)

    def _train_exp(self, experience: Experience, eval_streams=None, **kwargs):
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
            if not isinstance(exp, Sequence):
                eval_streams[i] = [exp]
        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            self._after_training_epoch(**kwargs)

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

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
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

    def _before_training_epoch(self, **kwargs):
        trigger_plugins(self, "before_training_epoch", **kwargs)

    def _after_training_epoch(self, **kwargs):
        trigger_plugins(self, "after_training_epoch", **kwargs)

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

    def __init__(self, eval_every=-1, peval_mode="epoch", do_initial=True):
        """Init.

        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        :param do_initial: whether to evaluate before each `train` call.
            Occasionally needed becuase some metrics need to know the
            accuracy before training.
        """
        super().__init__()
        assert peval_mode in {"epoch", "iteration"}
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
            self._peval(strategy)

    def before_training_exp(self, strategy, **kwargs):
        # We evaluate at the start of each experience because train_epochs
        # could change.
        self.do_final = True
        if self.peval_mode == "epoch":
            if (
                self.eval_every > 0
                and (strategy.train_epochs - 1) % self.eval_every == 0
            ):
                self.do_final = False
        else:  # peval_mode == 'iteration'
            # we may need to fix this but we don't have a way to know
            # the number of total iterations.
            # Right now there may be two eval calls at the last iterations.
            pass
        self.do_final = self.do_final and self.eval_every > -1

    def after_training_exp(self, strategy, **kwargs):
        """Final eval after a learning experience."""
        if self.do_final:
            self._peval(strategy)

    def _peval(self, strategy):
        for el in strategy._eval_streams:
            strategy.eval(el)

    def _maybe_peval(self, strategy, counter):
        if self.eval_every > 0 and counter % self.eval_every == 0:
            self._peval(strategy)

    def after_training_epoch(self, strategy: "SupervisedTemplate", **kwargs):
        """Periodic eval controlled by `self.eval_every` and
        `self.peval_mode`."""
        if self.peval_mode == "epoch":
            self._maybe_peval(strategy, strategy.clock.train_exp_epochs)

    def after_training_iteration(
        self, strategy: "SupervisedTemplate", **kwargs
    ):
        """Periodic eval controlled by `self.eval_every` and
        `self.peval_mode`."""
        if self.peval_mode == "iteration":
            self._maybe_peval(strategy, strategy.clock.train_exp_iterations)
