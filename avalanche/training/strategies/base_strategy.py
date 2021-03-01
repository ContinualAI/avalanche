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
from collections import defaultdict
from typing import Optional, Sequence, Union

from torch.nn import Module
from torch.optim import Optimizer

from avalanche.benchmarks.scenarios import IExperience
from avalanche.benchmarks.utils.data_loader import \
    MultiTaskMultiBatchDataLoader, MultiTaskDataLoader
from avalanche.logging import default_logger
from typing import TYPE_CHECKING

from avalanche.training.plugins import EvaluationPlugin
if TYPE_CHECKING:
    from avalanche.training.plugins import StrategyPlugin


class BaseStrategy:
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger):
        """
        BaseStrategy is the super class of all task-based continual learning
        strategies. It implements a basic training loop and callback system
        that allows to execute code at each experience of the training loop.
        Plugins can be used to implement callbacks to augment the training
        loop with additional behavior (e.g. a memory buffer for replay).

        This strategy supports several continual learning scenarios:
        - class-incremental scenarios (no task labels)
        - multi-task scenarios, where task labels are provided)
        - multi-incremental scenarios, where the same task may be revisited

        The exact scenario depends on the data stream and whether it provides
        the task labels.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device to run the model.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_epochs = train_epochs
        self.train_mb_size = train_mb_size
        self.eval_mb_size = train_mb_size if eval_mb_size is None \
            else eval_mb_size
        self.device = device
        self.plugins = [] if plugins is None else plugins
        if evaluator is None:
            evaluator = EvaluationPlugin()
        self.plugins.append(evaluator)
        self.evaluator = evaluator

        # Flow state variables
        self.training_exp_counter = 0  # +1 at the end of each experience.
        self.eval_exp_id = None  # eval-flow only
        self.epoch = None
        self.experience = None
        # data used to train. May be modified by plugins. In general, plugins
        # can append data to it but should not read it (e.g. for replay)
        # because it may contain extra samples from previous experiences.
        self.adapted_dataset = None
        self.current_dataloader = None
        self.mb_it = None  # train-flow only. minibatch iteration.
        self.mb_x, self.mb_y = None, None
        self.loss = None
        self.logits = None
        self.train_task_label: Optional[int] = None
        self.eval_task_label: Optional[int] = None
        self.is_training: bool = False

    @property
    def is_eval(self):
        return not self.is_training

    def update_optimizer(self, old_params, new_params, reset_state=True):
        """ Update the optimizer by substituting old_params with new_params.

        :param old_params: List of old trainable parameters.
        :param new_params: List of new trainable parameters.
        :param reset_state: Wheter to reset the optimizer's state.
            Defaults to True.
        :return:
        """
        for old_p, new_p in zip(old_params, new_params):
            found = False
            # iterate over group and params for each group.
            for group in self.optimizer.param_groups:
                for i, curr_p in enumerate(group['params']):
                    if hash(curr_p) == hash(old_p):
                        # update parameter reference
                        group['params'][i] = new_p
                        found = True
                        break
                if found:
                    break
            if not found:
                raise Exception(f"Parameter {old_params} not found in the "
                                f"current optimizer.")
        if reset_state:
            # State contains parameter-specific information.
            # We reset it because the model is (probably) changed.
            self.optimizer.state = defaultdict(dict)

    def add_new_params_to_optimizer(self, new_params):
        """ Add new parameters to the trainable parameters.

        :param new_params: list of trainable parameters
        """
        self.optimizer.add_param_group({'params': new_params})

    def train(self, experiences: Union[IExperience, Sequence[IExperience]],
              **kwargs):
        """ Training loop. if experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.

        :param experiences: single IExperience or sequence.
        """
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        if isinstance(experiences, IExperience):
            experiences = [experiences]

        res = []
        self.before_training(**kwargs)
        for exp in experiences:
            self.train_task_label = exp.task_label
            self.train_exp(exp, **kwargs)
            res.append(self.evaluator.current_metrics.copy())

        self.after_training(**kwargs)
        return res

    def train_exp(self, experience: IExperience, **kwargs):
        """
        Training loop over a single IExperience object.

        :param experience: CL experience information.
        :param kwargs: custom arguments.
        """
        self.experience = experience
        self.model.train()
        self.model.to(self.device)

        self.adapted_dataset = experience.dataset
        self.adapted_dataset = self.adapted_dataset.train()
        self.adapt_train_dataset(**kwargs)
        self.make_train_dataloader(**kwargs)

        self.before_training_exp(**kwargs)
        self.epoch = 0
        for self.epoch in range(self.train_epochs):
            self.before_training_epoch(**kwargs)
            self.training_epoch(**kwargs)
            self.after_training_epoch(**kwargs)
        self.after_training_exp(**kwargs)

    def eval(self,
             exp_list: Union[IExperience, Sequence[IExperience]],
             **kwargs):
        """
        Evaluate the current model on a series of experiences.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.
        """
        self.is_training = False
        self.model.eval()
        self.model.to(self.device)

        if isinstance(exp_list, IExperience):
            exp_list = [exp_list]

        res = []
        self.before_eval(**kwargs)
        for exp in exp_list:
            self.eval_task_label = exp.task_label
            self.experience = exp
            self.eval_exp_id = exp.current_experience

            self.adapted_dataset = exp.dataset
            self.adapted_dataset = self.adapted_dataset.eval()

            self.adapt_eval_dataset(**kwargs)
            self.make_eval_dataloader(**kwargs)

            self.before_eval_exp(**kwargs)
            self.eval_epoch(**kwargs)
            self.after_eval_exp(**kwargs)
            res.append(self.evaluator.current_metrics.copy())

        self.after_eval(**kwargs)
        return res

    def before_training_exp(self, **kwargs):
        """
        Called  after the dataset and data loader creation and
        before the training loop.
        """
        for p in self.plugins:
            p.before_training_exp(self, **kwargs)

    def make_train_dataloader(self, num_workers=0, shuffle=True, **kwargs):
        """
        Called after the dataset instantiation. Initialize the data loader.
        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """
        self.current_dataloader = MultiTaskMultiBatchDataLoader(
            self.adapted_dataset,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle)

    def make_eval_dataloader(self, num_workers=0, **kwargs):
        """
        Initialize the eval data loader.
        :param num_workers:
        :param kwargs:
        :return:
        """
        self.current_dataloader = MultiTaskDataLoader(
            self.adapted_dataset,
            oversample_small_tasks=False,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
        )

    def adapt_train_dataset(self, **kwargs):
        """
        Called after the dataset initialization and before the
        dataloader initialization. Allows to customize the dataset.
        :param kwargs:
        :return:
        """
        self.adapted_dataset = {
            self.experience.task_label: self.adapted_dataset}
        for p in self.plugins:
            p.adapt_train_dataset(self, **kwargs)

    def before_training_epoch(self, **kwargs):
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
        for self.mb_it, batches in \
                enumerate(self.current_dataloader):
            self.before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0
            for self.mb_task_id, (self.mb_x, self.mb_y) in batches.items():
                self.mb_x = self.mb_x.to(self.device)
                self.mb_y = self.mb_y.to(self.device)

                # Forward
                self.before_forward(**kwargs)
                self.logits = self.model(self.mb_x)
                self.after_forward(**kwargs)

                # Loss & Backward
                self.loss += self.criterion(self.logits, self.mb_y)

            self.before_backward(**kwargs)
            self.loss.backward()
            self.after_backward(**kwargs)

            # Optimization step
            self.before_update(**kwargs)
            self.optimizer.step()
            self.after_update(**kwargs)

            self.after_training_iteration(**kwargs)

    def before_training(self, **kwargs):
        for p in self.plugins:
            p.before_training(self, **kwargs)

    def after_training(self, **kwargs):
        for p in self.plugins:
            p.after_training(self, **kwargs)

    def before_training_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_training_iteration(self, **kwargs)

    def before_forward(self, **kwargs):
        for p in self.plugins:
            p.before_forward(self, **kwargs)

    def after_forward(self, **kwargs):
        for p in self.plugins:
            p.after_forward(self, **kwargs)

    def before_backward(self, **kwargs):
        for p in self.plugins:
            p.before_backward(self, **kwargs)

    def after_backward(self, **kwargs):
        for p in self.plugins:
            p.after_backward(self, **kwargs)

    def after_training_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_training_iteration(self, **kwargs)

    def before_update(self, **kwargs):
        for p in self.plugins:
            p.before_update(self, **kwargs)

    def after_update(self, **kwargs):
        for p in self.plugins:
            p.after_update(self, **kwargs)

    def after_training_epoch(self, **kwargs):
        for p in self.plugins:
            p.after_training_epoch(self, **kwargs)

    def after_training_exp(self, **kwargs):
        for p in self.plugins:
            p.after_training_exp(self, **kwargs)

        # Reset flow-state variables. They should not be used outside the flow
        self.epoch = None
        self.experience = None
        self.adapted_dataset = None
        self.current_dataloader = None
        self.mb_it = None
        self.mb_it, self.mb_x, self.mb_y = None, None, None
        self.loss = None
        self.logits = None

    def before_eval(self, **kwargs):
        for p in self.plugins:
            p.before_eval(self, **kwargs)

    def before_eval_exp(self, **kwargs):
        for p in self.plugins:
            p.before_eval_exp(self, **kwargs)

    def adapt_eval_dataset(self, **kwargs):
        self.adapted_dataset = {
            self.experience.task_label: self.adapted_dataset}
        for p in self.plugins:
            p.adapt_eval_dataset(self, **kwargs)

    def eval_epoch(self, **kwargs):
        for self.mb_it, (self.mb_task_id, self.mb_x, self.mb_y) in \
                enumerate(self.current_dataloader):
            self.before_eval_iteration(**kwargs)

            self.mb_x = self.mb_x.to(self.device)
            self.mb_y = self.mb_y.to(self.device)

            self.before_eval_forward(**kwargs)
            self.logits = self.model(self.mb_x)
            self.after_eval_forward(**kwargs)
            self.loss = self.criterion(self.logits, self.mb_y)

            self.after_eval_iteration(**kwargs)

    def after_eval_exp(self, **kwargs):
        for p in self.plugins:
            p.after_eval_exp(self, **kwargs)

    def after_eval(self, **kwargs):
        for p in self.plugins:
            p.after_eval(self, **kwargs)
        # Reset flow-state variables. They should not be used outside the flow
        self.eval_exp_id = None
        self.experience = None
        self.adapted_dataset = None
        self.current_dataloader = None
        self.mb_it = None
        self.mb_it, self.mb_x, self.mb_y = None, None, None
        self.loss = None
        self.logits = None

        self.training_exp_counter += 1

    def before_eval_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_eval_iteration(self, **kwargs)

    def before_eval_forward(self, **kwargs):
        for p in self.plugins:
            p.before_eval_forward(self, **kwargs)

    def after_eval_forward(self, **kwargs):
        for p in self.plugins:
            p.after_eval_forward(self, **kwargs)

    def after_eval_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_eval_iteration(self, **kwargs)


__all__ = ['BaseStrategy']
