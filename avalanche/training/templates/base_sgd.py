from typing import Iterable, Sequence, Optional, Union, List
from pkg_resources import parse_version

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.benchmarks import CLExperience, CLStream
from avalanche.core import BaseSGDPlugin
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.clock import Clock
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.base import BaseTemplate, ExpSequence
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader, \
    collate_from_data_or_kwargs
from avalanche.training.utils import trigger_plugins


class BaseSGDTemplate(BaseTemplate):
    """Base SGD class for continual learning skeletons.

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
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[List["SupervisedPlugin"]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        peval_mode="epoch",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
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

        self._criterion = criterion
        """ Criterion. """

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
        assert peval_mode in {"experience", "epoch", "iteration"}
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

        self.adapted_dataset = None
        """ Data used to train. It may be modified by plugins. Plugins can 
        append data to it (e.g. for replay). 

        .. note::

            This dataset may contain samples from different experiences. If you 
            want the original data for the current experience  
            use :attr:`.BaseTemplate.experience`.
        """

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

    def _train_exp(
        self, experience: CLExperience, eval_streams, **kwargs
    ):
        # Should be implemented in Observation Type
        raise NotImplementedError()

    def _eval_exp(self, **kwargs):
        self.eval_epoch(**kwargs)

    def make_optimizer(self, **kwargs):
        """Optimizer initialization."""
        # Should be implemented in Observation Type
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
        # Should be implemented in Update Type
        raise NotADirectoryError()

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

    # ==================================================================> NEW

    def check_model_and_optimizer(self):
        # Should be implemented in observation type
        raise NotImplementedError()

    def _before_training_exp(self, **kwargs):
        """Setup to train on a single experience."""
        # Data Adaptation (e.g. add new samples/data augmentation)
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)

        self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        # self.model = self.model_adaptation()
        # self.make_optimizer()
        self.check_model_and_optimizer()

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
        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            self._after_training_epoch(**kwargs)

    def _save_train_state(self):
        """Save the training state which may be modified by the eval loop.

        This currently includes: experience, adapted_dataset, dataloader,
        is_training, and train/eval modes for each module.

        TODO: we probably need a better way to do this.
        """
        state = super()._save_train_state()
        new_state = {
            "adapted_dataset": self.adapted_dataset,
            "dataloader": self.dataloader,
        }
        return {**state, **new_state}

    def train_dataset_adaptation(self, **kwargs):
        """Initialize `self.adapted_dataset`."""
        self.adapted_dataset = self.experience.dataset
        self.adapted_dataset = self.adapted_dataset.train()

    def _load_train_state(self, prev_state):
        super()._load_train_state(prev_state)
        self.adapted_dataset = prev_state["adapted_dataset"]
        self.dataloader = prev_state["dataloader"]

    def _before_eval_exp(self, **kwargs):

        # Data Adaptation
        self._before_eval_dataset_adaptation(**kwargs)
        self.eval_dataset_adaptation(**kwargs)
        self._after_eval_dataset_adaptation(**kwargs)

        self.make_eval_dataloader(**kwargs)
        # Model Adaptation (e.g. freeze/add new units)
        self.model = self.model_adaptation()

        super()._before_eval_exp(**kwargs)

    def make_train_dataloader(
        self,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
        **kwargs
    ):
        """Data loader initialization.

        Called at the start of each learning experience after the dataset
        adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """

        other_dataloader_args = {}

        if parse_version(torch.__version__) >= parse_version("1.7.0"):
            other_dataloader_args["persistent_workers"] = persistent_workers
        for k, v in kwargs.items():
            other_dataloader_args[k] = v

        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset,
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            **other_dataloader_args
        )

    def make_eval_dataloader(
        self, num_workers=0, pin_memory=True, persistent_workers=False, **kwargs
    ):
        """
        Initializes the eval data loader.
        :param num_workers: How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
            (default: 0).
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param kwargs:
        :return:
        """
        other_dataloader_args = {}

        if parse_version(torch.__version__) >= parse_version("1.7.0"):
            other_dataloader_args["persistent_workers"] = persistent_workers
        for k, v in kwargs.items():
            other_dataloader_args[k] = v

        collate_from_data_or_kwargs(self.adapted_dataset,
                                    other_dataloader_args)
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
            pin_memory=pin_memory,
            **other_dataloader_args
        )

    def eval_dataset_adaptation(self, **kwargs):
        """Initialize `self.adapted_dataset`."""
        self.adapted_dataset = self.experience.dataset
        self.adapted_dataset = self.adapted_dataset.eval()

    def _unpack_minibatch(self):
        """Move to device"""
        # First verify the mini-batch
        self._check_minibatch()

        if isinstance(self.mbatch, tuple):
            self.mbatch = list(self.mbatch)
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

    # ==================================================================> NEW

    def _before_train_dataset_adaptation(self, **kwargs):
        trigger_plugins(self, "before_train_dataset_adaptation", **kwargs)

    def _after_train_dataset_adaptation(self, **kwargs):
        trigger_plugins(self, "after_train_dataset_adaptation", **kwargs)

    def _before_eval_dataset_adaptation(self, **kwargs):
        trigger_plugins(self, "before_eval_dataset_adaptation", **kwargs)

    def _after_eval_dataset_adaptation(self, **kwargs):
        trigger_plugins(self, "after_eval_dataset_adaptation", **kwargs)


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
        assert peval_mode in {"experience", "epoch", "iteration"}
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

    def _peval(self, strategy, **kwargs):
        for el in strategy._eval_streams:
            strategy.eval(el, **kwargs)

    def _maybe_peval(self, strategy, counter, **kwargs):
        if self.eval_every > 0 and counter % self.eval_every == 0:
            self._peval(strategy, **kwargs)

    def after_training_epoch(self, strategy: "BaseSGDTemplate",
                             **kwargs):
        """Periodic eval controlled by `self.eval_every` and
        `self.peval_mode`."""
        if self.peval_mode == "epoch":
            self._maybe_peval(strategy, strategy.clock.train_exp_epochs,
                              **kwargs)

    def after_training_iteration(self, strategy: "BaseSGDTemplate",
                                 **kwargs):
        """Periodic eval controlled by `self.eval_every` and
        `self.peval_mode`."""
        if self.peval_mode == "iteration":
            self._maybe_peval(strategy, strategy.clock.train_exp_iterations,
                              **kwargs)

    # ---> New
    def after_training_exp(self, strategy, **kwargs):
        """Final eval after a learning experience."""
        if self.do_final:
            self._peval(strategy, **kwargs)

    # def after_training_exp(self, strategy: "BaseOnlineSGDTemplate", **kwargs):
    #     """Periodic eval controlled by `self.eval_every` and
    #     `self.peval_mode`."""
    #     if self.peval_mode == "experience":
    #         self._maybe_peval(strategy, strategy.clock.train_exp_counter,
    #                           **kwargs)
