################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import logging
import warnings

import torch
from torch.utils.data import DataLoader
from typing import Optional, Sequence, Union, List

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.models import DynamicModule
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.clock import Clock
from avalanche.training.plugins.evaluation import default_logger
from typing import TYPE_CHECKING

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins import StrategyPlugin

if TYPE_CHECKING:
    from avalanche.core import StrategyCallbacks


logger = logging.getLogger(__name__)


class BaseStrategy:
    """ Base class for continual learning strategies.

    BaseStrategy is the super class of all task-based continual learning
    strategies. It implements a basic training loop and callback system
    that allows to execute code at each experience of the training loop.
    Plugins can be used to implement callbacks to augment the training
    loop with additional behavior (e.g. a memory buffer for replay).

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
                    # backward
                    # model update

    """
    DISABLED_CALLBACKS: Sequence[str] = ()
    """Internal class attribute used to disable some callbacks if a strategy
    does not support them."""

    def __init__(self, model: Module, optimizer: Optimizer,
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, eval_every=-1, peval_mode='epoch'):
        """ Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
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
        self._criterion = criterion

        self.model: Module = model
        """ PyTorch model. """

        self.optimizer: Optimizer = optimizer
        """ PyTorch optimizer. """

        self.train_epochs: int = train_epochs
        """ Number of training epochs. """

        self.train_mb_size: int = train_mb_size
        """ Training mini-batch size. """

        self.eval_mb_size: int = train_mb_size if eval_mb_size is None \
            else eval_mb_size
        """ Eval mini-batch size. """

        self.device = device
        """ PyTorch device where the model will be allocated. """

        self.plugins = [] if plugins is None else plugins
        """ List of `StrategyPlugin`s. """

        if evaluator is None:
            evaluator = EvaluationPlugin()
        self.plugins.append(evaluator)
        self.evaluator = evaluator
        """ EvaluationPlugin used for logging and metric computations. """

        # Configure periodic evaluation.
        assert peval_mode in {'epoch', 'iteration'}
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
        self.experience = None
        """ Current experience. """

        self.adapted_dataset = None
        """ Data used to train. It may be modified by plugins. Plugins can 
        append data to it (e.g. for replay). 
         
        .. note::

            This dataset may contain samples from different experiences. If you 
            want the original data for the current experience  
            use :attr:`.BaseStrategy.experience`.
        """

        self.dataloader = None
        """ Dataloader. """

        self.mbatch = None
        """ Current mini-batch. """

        self.mb_output = None
        """ Model's output computed on the current mini-batch. """

        self.loss = None
        """ Loss of the current mini-batch. """

        self.is_training: bool = False
        """ True if the strategy is in training mode. """

        self.current_eval_stream = None
        """ Current evaluation stream. """

        self._stop_training = False

        self._warn_for_disabled_plugins_callbacks()
        self._warn_for_disabled_metrics_callbacks()

    @property
    def training_exp_counter(self):
        """ Counts the number of training steps. +1 at the end of each
        experience. """
        warnings.warn(
            "Deprecated attribute. You should use self.clock.train_exp_counter"
            " instead.", DeprecationWarning)
        return self.clock.train_exp_counter

    @property
    def epoch(self):
        """ Epoch counter. """
        warnings.warn(
            "Deprecated attribute. You should use self.clock.train_exp_epochs"
            " instead.", DeprecationWarning)
        return self.clock.train_exp_epochs

    @property
    def mb_it(self):
        """ Iteration counter. Reset at the start of a new epoch. """
        warnings.warn(
            "Deprecated attribute. You should use "
            "self.clock.train_epoch_iterations"
            " instead.", DeprecationWarning)
        return self.clock.train_epoch_iterations

    @property
    def is_eval(self):
        """ True if the strategy is in evaluation mode. """
        return not self.is_training

    @property
    def mb_x(self):
        """ Current mini-batch input. """
        return self.mbatch[0]

    @property
    def mb_y(self):
        """ Current mini-batch target. """
        return self.mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        assert len(self.mbatch) >= 3
        return self.mbatch[-1]

    def criterion(self):
        """ Loss function. """
        return self._criterion(self.mb_output, self.mb_y)

    def train(self, experiences: Union[Experience, Sequence[Experience]],
              eval_streams: Optional[Sequence[Union[Experience,
                                                    Sequence[
                                                        Experience]]]] = None,
              **kwargs):
        """ Training loop. if experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.
        It returns a dictionary with last recorded value for each metric.

        :param experiences: single Experience or sequence.
        :param eval_streams: list of streams for evaluation.
            If None: use training experiences for evaluation.
            Use [] if you do not want to evaluate during training.

        :return: dictionary containing last recorded value for
            each metric name.
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
            self.train_exp(self.experience, eval_streams, **kwargs)
        self._after_training(**kwargs)

        res = self.evaluator.get_last_metrics()
        return res

    def train_exp(self, experience: Experience, eval_streams=None, **kwargs):
        """ Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        self.experience = experience
        self.model.train()

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Sequence):
                eval_streams[i] = [exp]

        # Data Adaptation (e.g. add new samples/data augmentation)
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)
        self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        self.model = self.model_adaptation()
        self.make_optimizer()

        self._before_training_exp(**kwargs)

        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            self._after_training_epoch(**kwargs)

        self._after_training_exp(**kwargs)

    def _load_train_state(self, _prev_model_training_modes, _prev_state):
        # restore train-state variables and training mode.
        self.experience, self.adapted_dataset = _prev_state[:2]
        self.dataloader = _prev_state[2]
        self.is_training = _prev_state[3]
        # restore each layer's training mode to original
        for name, layer in self.model.named_modules():
            try:
                prev_mode = _prev_model_training_modes[name]
                layer.train(mode=prev_mode)
            except KeyError:
                # Unknown parameter, probably added during the eval
                # model's adaptation. We set it to train mode.
                layer.train()

    def _save_train_state(self):
        """Save the training state which may be modified by the eval loop.

        This currently includes: experience, adapted_dataset, dataloader,
        is_training, and train/eval modes for each module.

        TODO: we probably need a better way to do this.
        """
        _prev_state = (
            self.experience,
            self.adapted_dataset,
            self.dataloader,
            self.is_training)
        # save each layer's training mode, to restore it later
        _prev_model_training_modes = {}
        for name, layer in self.model.named_modules():
            _prev_model_training_modes[name] = layer.training
        return _prev_model_training_modes, _prev_state

    def stop_training(self):
        """ Signals to stop training at the next iteration. """
        self._stop_training = True

    def train_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """
        self.adapted_dataset = self.experience.dataset
        self.adapted_dataset = self.adapted_dataset.train()

    @torch.no_grad()
    def eval(self,
             exp_list: Union[Experience, Sequence[Experience]],
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
        train_state = self._save_train_state()
        self.is_training = False
        self.model.eval()

        if not isinstance(exp_list, Sequence):
            exp_list = [exp_list]
        self.current_eval_stream = exp_list

        self._before_eval(**kwargs)
        for self.experience in exp_list:
            # Data Adaptation
            self._before_eval_dataset_adaptation(**kwargs)
            self.eval_dataset_adaptation(**kwargs)
            self._after_eval_dataset_adaptation(**kwargs)
            self.make_eval_dataloader(**kwargs)

            # Model Adaptation (e.g. freeze/add new units)
            self.model = self.model_adaptation()

            self._before_eval_exp(**kwargs)
            self.eval_epoch(**kwargs)
            self._after_eval_exp(**kwargs)

        self._after_eval(**kwargs)
        res = self.evaluator.get_last_metrics()

        # restore previous shared state.
        self._load_train_state(*train_state)
        return res

    def _before_training_exp(self, **kwargs):
        """
        Called  after the dataset and data loader creation and
        before the training loop.
        """
        for p in self.plugins:
            p.before_training_exp(self, **kwargs)

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        """ Data loader initialization.

        Called at the start of each learning experience after the dataset 
        adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """
        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset,
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory)

    def make_eval_dataloader(self, num_workers=0, pin_memory=True,
                             **kwargs):
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
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
            pin_memory=pin_memory)

    def _after_train_dataset_adaptation(self, **kwargs):
        """
        Called after the dataset adaptation and before the
        dataloader initialization. Allows to customize the dataset.
        :param kwargs:
        :return:
        """
        for p in self.plugins:
            p.after_train_dataset_adaptation(self, **kwargs)

    def _before_training_epoch(self, **kwargs):
        """
        Called at the beginning of a new training epoch.
        :param kwargs:
        :return:
        """
        for p in self.plugins:
            p.before_training_epoch(self, **kwargs)

    def training_epoch(self, **kwargs):
        """ Training epoch.
        
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
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _unpack_minibatch(self):
        """ We assume mini-batches have the form <x, y, ..., t>.
        This allows for arbitrary tensors between y and t.
        Keep in mind that in the most general case mb_task_id is a tensor
        which may contain different labels for each sample.
        """
        assert len(self.mbatch) >= 3
        for i in range(len(self.mbatch)):
            self.mbatch[i] = self.mbatch[i].to(self.device)

    def _before_training(self, **kwargs):
        for p in self.plugins:
            p.before_training(self, **kwargs)

    def _after_training(self, **kwargs):
        for p in self.plugins:
            p.after_training(self, **kwargs)

    def _before_training_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_training_iteration(self, **kwargs)

    def _before_forward(self, **kwargs):
        for p in self.plugins:
            p.before_forward(self, **kwargs)

    def _after_forward(self, **kwargs):
        for p in self.plugins:
            p.after_forward(self, **kwargs)

    def _before_backward(self, **kwargs):
        for p in self.plugins:
            p.before_backward(self, **kwargs)

    def _after_backward(self, **kwargs):
        for p in self.plugins:
            p.after_backward(self, **kwargs)

    def _after_training_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_training_iteration(self, **kwargs)

    def _before_update(self, **kwargs):
        for p in self.plugins:
            p.before_update(self, **kwargs)

    def _after_update(self, **kwargs):
        for p in self.plugins:
            p.after_update(self, **kwargs)

    def _after_training_epoch(self, **kwargs):
        for p in self.plugins:
            p.after_training_epoch(self, **kwargs)

    def _after_training_exp(self, **kwargs):
        for p in self.plugins:
            p.after_training_exp(self, **kwargs)

    def _before_eval(self, **kwargs):
        for p in self.plugins:
            p.before_eval(self, **kwargs)

    def _before_eval_exp(self, **kwargs):
        for p in self.plugins:
            p.before_eval_exp(self, **kwargs)

    def eval_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """
        self.adapted_dataset = self.experience.dataset
        self.adapted_dataset = self.adapted_dataset.eval()

    def _before_eval_dataset_adaptation(self, **kwargs):
        for p in self.plugins:
            p.before_eval_dataset_adaptation(self, **kwargs)

    def _after_eval_dataset_adaptation(self, **kwargs):
        for p in self.plugins:
            p.after_eval_dataset_adaptation(self, **kwargs)

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

    def _after_eval_exp(self, **kwargs):
        for p in self.plugins:
            p.after_eval_exp(self, **kwargs)

    def _after_eval(self, **kwargs):
        for p in self.plugins:
            p.after_eval(self, **kwargs)

    def _before_eval_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_eval_iteration(self, **kwargs)

    def _before_eval_forward(self, **kwargs):
        for p in self.plugins:
            p.before_eval_forward(self, **kwargs)

    def _after_eval_forward(self, **kwargs):
        for p in self.plugins:
            p.after_eval_forward(self, **kwargs)

    def _after_eval_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_eval_iteration(self, **kwargs)

    def _before_train_dataset_adaptation(self, **kwargs):
        for p in self.plugins:
            p.before_train_dataset_adaptation(self, **kwargs)

    def model_adaptation(self, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model

        for module in model.modules():
            if isinstance(module, DynamicModule):
                module.adaptation(self.experience.dataset)
        return model.to(self.device)

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def make_optimizer(self):
        """Optimizer initialization.

        Called before each training experiene to configure the optimizer.
        """
        # we reset the optimizer's state after each experience.
        # This allows to add new parameters (new heads) and
        # freezing old units during the model's adaptation phase.
        reset_optimizer(self.optimizer, self.model)

    def _warn_for_disabled_plugins_callbacks(self):
        self._warn_for_disabled_callbacks(self.plugins)

    def _warn_for_disabled_metrics_callbacks(self):
        self._warn_for_disabled_callbacks(self.evaluator.metrics)

    def _warn_for_disabled_callbacks(
            self,
            plugins: List["StrategyCallbacks"]
    ):
        """
        Will log some warnings in case some plugins appear to be using callbacks
        that have been de-activated by the strategy class.
        """
        for disabled_callback_name in self.DISABLED_CALLBACKS:
            for plugin in plugins:
                callback = getattr(plugin, disabled_callback_name)
                callback_class = callback.__qualname__.split('.')[0]
                if callback_class not in (
                    "StrategyPlugin",
                    "PluginMetric",
                    "EvaluationPlugin",
                    "GenericPluginMetric",
                ):
                    logger.warning(
                        f"{plugin.__class__.__name__} seems to use "
                        f"the callback {disabled_callback_name} "
                        f"which is disabled by {self.__class__.__name__}"
                    )


class PeriodicEval(StrategyPlugin):
    """Schedules periodic evaluation during training.

    This plugin is automatically configured and added by the BaseStrategy.
    """

    def __init__(self, eval_every=-1, peval_mode='epoch', do_initial=True):
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
        assert peval_mode in {'epoch', 'iteration'}
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
        if self.peval_mode == 'epoch':
            if self.eval_every > 0 and \
                    (strategy.train_epochs - 1) % self.eval_every == 0:
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

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        """Periodic eval controlled by `self.eval_every` and
        `self.peval_mode`."""
        if self.peval_mode == 'epoch':
            self._maybe_peval(strategy, strategy.clock.train_exp_epochs)

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        """Periodic eval controlled by `self.eval_every` and
        `self.peval_mode`."""
        if self.peval_mode == 'iteration':
            self._maybe_peval(strategy, strategy.clock.train_exp_iterations)


__all__ = ['BaseStrategy']
