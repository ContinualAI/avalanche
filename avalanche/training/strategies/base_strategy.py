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

if TYPE_CHECKING:
    from avalanche.core import StrategyCallbacks
    from avalanche.training.plugins import StrategyPlugin


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

    def __init__(self, model: Module, optimizer: Optimizer,
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, eval_every=-1):
        """ 
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
            training loop.
                if -1: no evaluation during training.
                if  0: calls `eval` after the final epoch of each training
                    experience and before training on the first experience.
                if >0: calls `eval` every `eval_every` epochs, at the end
                    of all the epochs for a single experience and before
                    training on the first experience.
        """
        self._criterion = criterion

        self.model: Module = model
        """ PyTorch model. """

        self.optimizer = optimizer
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

        self.clock = Clock()
        """ Incremental counters for strategy events. """
        # WARNING: Clock needs to be the last plugin, otherwise
        # counters will be wrong for plugins called after it.
        self.plugins.append(self.clock)

        self.eval_every = eval_every
        """ Frequency of the evaluation during training. """

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
        """ User-provided evaluation stream on `eval` call. """

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

        self._before_training(**kwargs)

        self._periodic_eval(eval_streams, do_final=False, do_initial=True)

        for self.experience in experiences:
            self.train_exp(self.experience, eval_streams, **kwargs)
        self._after_training(**kwargs)

        res = self.evaluator.get_last_metrics()
        return res

    def train_exp(self, experience: Experience, eval_streams=None, **kwargs):
        """
        Training loop over a single Experience object.

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
        
        do_final = True
        if self.eval_every > 0 and \
                (self.train_epochs - 1) % self.eval_every == 0:
            do_final = False

        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            self._after_training_epoch(**kwargs)
            self._periodic_eval(eval_streams, do_final=False)

        # Final evaluation
        self._periodic_eval(eval_streams, do_final=do_final)
        self._after_training_exp(**kwargs)

    def _periodic_eval(self, eval_streams, do_final, do_initial=False):
        """ Periodic eval controlled by `self.eval_every`. """
        # Since we are switching from train to eval model inside the training
        # loop, we need to save the training state, and restore it after the
        # eval is done.
        _prev_state = (
            self.experience,
            self.adapted_dataset,
            self.dataloader,
            self.is_training)
        
        # save each layer's training mode, to restore it later
        _prev_model_training_modes = {}
        for name, layer in self.model.named_modules():
            _prev_model_training_modes[name] = layer.training
        
        curr_epoch = self.clock.train_exp_epochs
        if (self.eval_every == 0 and (do_final or do_initial)) or \
           (self.eval_every > 0 and do_initial) or \
                (self.eval_every > 0 and curr_epoch % self.eval_every == 0):
            # in the first case we are outside epoch loop
            # in the second case we are within epoch loop
            for exp in eval_streams:
                self.eval(exp)

        # restore train-state variables and training mode.
        self.experience, self.adapted_dataset = _prev_state[:2]
        self.dataloader = _prev_state[2]
        self.is_training = _prev_state[3]
        
        # restore each layer's training mode to original 
        for name, layer in self.model.named_modules():
            prev_mode = _prev_model_training_modes[name]
            layer.train(mode=prev_mode)

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
        """
        Called after the dataset adaptation. Initializes the data loader.
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
        """
        Training epoch.
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
        if model is None:
            model = self.model

        for module in model.modules():
            if isinstance(module, DynamicModule):
                module.adaptation(self.experience.dataset)
        return model.to(self.device)

    def forward(self):
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def make_optimizer(self):
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


__all__ = ['BaseStrategy']
