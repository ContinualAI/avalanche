"""
    We test the new SupervisedTemplate against the old BaseStrategy.
    A sort of regression test to check that the new organization
    did not break anything.
"""

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
import copy
import logging
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, Sequence, Union

from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.models import DynamicModule, SimpleMLP, MTSimpleMLP
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.models.utils import avalanche_forward
from avalanche.training import Cumulative
from avalanche.training.plugins.clock import Clock

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.utils import trigger_plugins
from tests.test_dataloaders import get_fast_benchmark

logger = logging.getLogger(__name__)


class OldBaseStrategy:
    """The old BaseStrategy (avalanche 0.1.0)
    just a bit of refactoring to remove useless components for testing.
    """
    def __init__(self, model: Module, optimizer,
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins=None,
                 evaluator=None, eval_every=-1, peval_mode='epoch'):
        self._criterion = criterion
        self.model: Module = model
        self.optimizer = optimizer
        self.train_epochs: int = train_epochs
        self.train_mb_size: int = train_mb_size
        self.eval_mb_size: int = train_mb_size if eval_mb_size is None \
            else eval_mb_size
        self.device = device
        self.plugins = [] if plugins is None else plugins

        if evaluator is None:
            evaluator = EvaluationPlugin()
        self.plugins.append(evaluator)
        self.evaluator = evaluator
        """ EvaluationPlugin used for logging and metric computations. """

        # Configure periodic evaluation.
        assert peval_mode in {'epoch', 'iteration'}
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
        self.experience = None
        self.adapted_dataset = None
        self.dataloader = None
        self.mbatch = None
        self.mb_output = None
        self.loss = None
        self.is_training: bool = False
        self.current_eval_stream = None
        self._stop_training = False

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

        trigger_plugins(self, 'before_training')
        for self.experience in experiences:
            self.train_exp(self.experience, eval_streams, **kwargs)
        trigger_plugins(self, 'after_training')
        res = self.evaluator.get_last_metrics()
        return res

    def train_exp(self, experience: Experience, eval_streams=None, **kwargs):
        self.experience = experience
        self.model.train()

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Sequence):
                eval_streams[i] = [exp]

        # Data Adaptation (e.g. add new samples/data augmentation)
        trigger_plugins(self, 'before_train_dataset_adaptation')
        self.train_dataset_adaptation(**kwargs)
        trigger_plugins(self, 'after_train_dataset_adaptation')
        self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        self.model = self.model_adaptation()
        self.make_optimizer()

        trigger_plugins(self, 'before_training_exp')
        for _ in range(self.train_epochs):
            trigger_plugins(self, 'before_training_epoch')
            if self._stop_training:  # Early stopping
                self._stop_training = False
                break
            self.training_epoch(**kwargs)
            trigger_plugins(self, 'after_training_epoch')
        trigger_plugins(self, 'after_training_exp')

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

        trigger_plugins(self, 'before_eval')
        for self.experience in exp_list:
            # Data Adaptation
            trigger_plugins(self, 'before_eval_dataset_adaptation')
            self.eval_dataset_adaptation(**kwargs)
            trigger_plugins(self, 'after_eval_dataset_adaptation')
            self.make_eval_dataloader(**kwargs)

            # Model Adaptation (e.g. freeze/add new units)
            self.model = self.model_adaptation()

            trigger_plugins(self, 'before_eval_exp')
            self.eval_epoch(**kwargs)
            trigger_plugins(self, 'after_eval_exp')

        trigger_plugins(self, 'after_eval')
        res = self.evaluator.get_last_metrics()

        # restore previous shared state.
        self._load_train_state(*train_state)
        return res

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset,
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory)

    def make_eval_dataloader(self, num_workers=0, pin_memory=True,
                             **kwargs):
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
            pin_memory=pin_memory)

    def training_epoch(self, **kwargs):
        """ Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            trigger_plugins(self, 'before_training_iteration')

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            trigger_plugins(self, 'before_forward')
            self.mb_output = self.forward()
            trigger_plugins(self, 'after_forward')

            # Loss & Backward
            self.loss += self.criterion()

            trigger_plugins(self, 'before_backward')
            self.loss.backward()
            trigger_plugins(self, 'after_backward')

            # Optimization step
            trigger_plugins(self, 'before_update')
            self.optimizer.step()
            trigger_plugins(self, 'after_update')
            trigger_plugins(self, 'after_training_iteration')

    def _unpack_minibatch(self):
        assert len(self.mbatch) >= 3
        for i in range(len(self.mbatch)):
            self.mbatch[i] = self.mbatch[i].to(self.device)

    def eval_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """
        self.adapted_dataset = self.experience.dataset
        self.adapted_dataset = self.adapted_dataset.eval()

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            trigger_plugins(self, 'before_eval_iteration')

            trigger_plugins(self, 'before_eval_forward')
            self.mb_output = self.forward()
            trigger_plugins(self, 'after_eval_forward')
            self.loss = self.criterion()
            trigger_plugins(self, 'after_eval_iteration')

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
        reset_optimizer(self.optimizer, self.model)


class OldCumulative(OldBaseStrategy):
    """Old Cumulative training strategy."""

    def __init__(self, model: Module, optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins=None, evaluator=None, eval_every=-1):
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)
        self.dataset = None  # cumulative dataset

    def train_dataset_adaptation(self, **kwargs):
        """
            Concatenates all the previous experiences.
        """
        if self.dataset is None:
            self.dataset = self.experience.dataset
        else:
            self.dataset = AvalancheConcatDataset(
                [self.dataset, self.experience.dataset])
        self.adapted_dataset = self.dataset


class PeriodicEval:
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

    def after_training_epoch(self, strategy, **kwargs):
        """Periodic eval controlled by `self.eval_every` and
        `self.peval_mode`."""
        if self.peval_mode == 'epoch':
            self._maybe_peval(strategy, strategy.clock.train_exp_epochs)

    def after_training_iteration(self, strategy, **kwargs):
        """Periodic eval controlled by `self.eval_every` and
        `self.peval_mode`."""
        if self.peval_mode == 'iteration':
            self._maybe_peval(strategy, strategy.clock.train_exp_iterations)


class StoreLosses:
    def __init__(self):
        self.values = []

    def before_backward(self, strategy, **kwargs):
        self.values.append(strategy.loss.item())


class TestStrategyReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.model = SimpleMLP(input_size=6, hidden_size=10)
        cls.benchmark = get_fast_benchmark()

    def test_reproduce_old_base_strategy(self):
        ######################################
        # OLD BASE STRATEGY
        ######################################
        old_model = copy.deepcopy(self.model)
        p_old = StoreLosses()
        old_strategy = OldBaseStrategy(
            old_model,
            SGD(old_model.parameters(), lr=0.01),
            train_epochs=2,
            plugins=[p_old],
            evaluator=None
        )
        torch.manual_seed(42)
        old_strategy.train(self.benchmark.train_stream, shuffle=False)
        old_losses = np.array(p_old.values)

        ######################################
        # NEW SUPERVISED STRATEGY
        ######################################
        new_model = copy.deepcopy(self.model)
        p_new = StoreLosses()
        # if you want to check whether the seeds are set correctly
        # switch SupervisedTemplate with OldBaseStrategy and check that
        # values are exactly equal.
        new_strategy = SupervisedTemplate(
            new_model,
            SGD(new_model.parameters(), lr=0.01),
            train_epochs=2,
            plugins=[p_new],
            evaluator=None
        )
        torch.manual_seed(42)
        new_strategy.train(self.benchmark.train_stream, shuffle=False)
        new_losses = np.array(p_new.values)

        print(old_losses)
        print(new_losses)
        np.testing.assert_allclose(old_losses, new_losses)
        for par_old, par_new in zip(old_model.parameters(),
                                    new_model.parameters()):
            np.testing.assert_allclose(par_old.detach(), par_new.detach())

    def test_reproduce_old_cumulative_strategy(self):
        mt_model = MTSimpleMLP(input_size=6, hidden_size=10)
        criterion = CrossEntropyLoss()
        ######################################
        # OLD BASE STRATEGY
        ######################################
        old_model = copy.deepcopy(mt_model)
        p_old = StoreLosses()
        old_strategy = OldCumulative(
            old_model,
            SGD(old_model.parameters(), lr=0.01),
            criterion,
            train_epochs=2,
            plugins=[p_old],
            evaluator=None,
            train_mb_size=128,
        )
        torch.manual_seed(42)
        old_strategy.train(self.benchmark.train_stream, shuffle=False)
        old_losses = np.array(p_old.values)

        ######################################
        # NEW SUPERVISED STRATEGY
        ######################################
        new_model = copy.deepcopy(mt_model)
        p_new = StoreLosses()
        new_strategy = Cumulative(
            new_model,
            SGD(new_model.parameters(), lr=0.01),
            criterion,
            train_epochs=2,
            plugins=[p_new],
            evaluator=None,
            train_mb_size=128,
        )
        torch.manual_seed(42)
        new_strategy.train(self.benchmark.train_stream, shuffle=False)
        new_losses = np.array(p_new.values)

        print(old_losses)
        print(new_losses)
        np.testing.assert_allclose(old_losses, new_losses)
        for par_old, par_new in zip(old_model.parameters(),
                                    new_model.parameters()):
            np.testing.assert_allclose(par_old.detach(), par_new.detach())
