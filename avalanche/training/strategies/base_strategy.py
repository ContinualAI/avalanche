################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################
from collections import defaultdict
from typing import Optional, Sequence, Union

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.benchmarks.scenarios import IStepInfo
from avalanche.logging import default_logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avalanche.training.plugins import StrategyPlugin


class BaseStrategy:
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger):
        self.model = model
        """
        BaseStrategy is the super class of all continual learning strategies.
        It implements a basic training loop and callback system that can be
        customized by child strategies. Additionally, it supports plugins,
        a mechanism to augment existing strategies with additional
        behavior (e.g. a memory buffer for replay).

        This strategy does not use task identities. See
        :class:MultiTaskStrategy: if you need them.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param test_mb_size: mini-batch size for test.
        :param device: PyTorch device to run the model.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging 
            and metric computations.
        """
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_epochs = train_epochs
        self.train_mb_size = train_mb_size
        self.test_mb_size = train_mb_size if test_mb_size is None \
            else test_mb_size
        self.device = device
        self.plugins = [] if plugins is None else plugins
        self.plugins.append(evaluator)
        self.evaluator = evaluator

        # Flow state variables

        # Counter of each training step.
        # Incremented by 1 at the end of each step.
        self.training_step_counter = 0
        # test-flow only
        self.test_step_id = None
        self.epoch = None
        self.step_info = None
        self.current_data = None
        self.current_dataloader = None
        self.mb_it = None  # train-flow only. minibatch iteration.
        self.mb_x, self.mb_y = None, None
        self.loss = None
        self.logits = None
        self.train_task_label: Optional[int] = None
        self.test_task_label: Optional[int] = None
        self.is_training: bool = False

    @property
    def is_testing(self):
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

    def train(self, step_infos: Union[IStepInfo, Sequence[IStepInfo]],
              **kwargs):
        """ Training loop. if step_infos is a single element trains on it.
        If it is a sequence, trains the model on each step in order.
        This is different from joint training on the entire stream.

        :param step_infos: single IStepInfo or sequence.
        """
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        if isinstance(step_infos, IStepInfo):
            step_infos = [step_infos]

        res = []
        self.before_training(**kwargs)
        for step_info in step_infos:
            self.train_task_label = step_info.task_label
            self.train_step(step_info, **kwargs)
            res.append(self.evaluator.current_metrics.copy())

        self.after_training(**kwargs)
        return res

    def train_step(self, step_info: IStepInfo, **kwargs):
        """
        Training loop over a single IStepInfo object.

        :param step_info: CL step information.
        :param kwargs: custom arguments.
        """
        self.step_info = step_info
        self.model.train()
        self.model.to(self.device)

        self.current_data = step_info.dataset
        self.current_data.train()
        self.adapt_train_dataset(**kwargs)
        self.make_train_dataloader(**kwargs)

        self.before_training_step(**kwargs)
        self.epoch = 0
        for self.epoch in range(self.train_epochs):
            self.before_training_epoch(**kwargs)
            self.training_epoch(**kwargs)
            self.after_training_epoch(**kwargs)
        self.after_training_step(**kwargs)

    def test(self, step_list: Union[IStepInfo, Sequence[IStepInfo]], **kwargs):
        """
        Test the current model on a series of steps, as defined by test_part.

        :param step_info: CL step information.
        :param test_part: determines which steps to test on.
        :param kwargs: custom arguments.
        """
        self.is_training = False
        self.model.eval()
        self.model.to(self.device)

        if isinstance(step_list, IStepInfo):
            step_list = [step_list]

        res = []
        self.before_test(**kwargs)
        for step_info in step_list:
            self.test_task_label = step_info.task_label
            self.step_info = step_info
            self.test_step_id = step_info.current_step

            self.current_data = step_info.dataset
            self.current_data.eval()

            self.adapt_test_dataset(**kwargs)
            self.make_test_dataloader(**kwargs)

            self.before_test_step(**kwargs)
            self.test_epoch(**kwargs)
            self.after_test_step(**kwargs)
            res.append(self.evaluator.current_metrics.copy())

        self.after_test(**kwargs)
        return res

    def before_training_step(self, **kwargs):
        """
        Called  after the dataset and data loader creation and
        before the training loop.
        """
        for p in self.plugins:
            p.before_training_step(self, **kwargs)

    def make_train_dataloader(self, num_workers=0, shuffle=True, **kwargs):
        """
        Called after the dataset instantiation. Initialize the data loader.
        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """
        self.current_dataloader = DataLoader(self.current_data,
                                             num_workers=num_workers, 
                                             batch_size=self.train_mb_size,
                                             shuffle=shuffle)

    def make_test_dataloader(self, num_workers=0, **kwargs):
        """
        Initialize the test data loader.
        :param num_workers:
        :param kwargs:
        :return:
        """
        self.current_dataloader = DataLoader(
              self.current_data,
              num_workers=num_workers, 
              batch_size=self.test_mb_size)

    def adapt_train_dataset(self, **kwargs):
        """
        Called after the dataset initialization and before the
        dataloader initialization. Allows to customize the dataset.
        :param kwargs:
        :return:
        """
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
        for self.mb_it, (self.mb_x, self.mb_y) in \
                enumerate(self.current_dataloader):
            self.before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.mb_x = self.mb_x.to(self.device)
            self.mb_y = self.mb_y.to(self.device)

            # Forward
            self.before_forward(**kwargs)
            self.logits = self.model(self.mb_x)
            self.after_forward(**kwargs)

            # Loss & Backward
            self.loss = self.criterion(self.logits, self.mb_y)
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

    def after_training_step(self, **kwargs):
        for p in self.plugins:
            p.after_training_step(self, **kwargs)

        self.training_step_counter += 1

        # Reset flow-state variables. They should not be used outside the flow
        self.epoch = None
        self.step_info = None
        self.current_data = None
        self.current_dataloader = None
        self.mb_it = None
        self.mb_x, self.mb_y = None, None
        self.loss = None
        self.logits = None

    def before_test(self, **kwargs):
        for p in self.plugins:
            p.before_test(self, **kwargs)

    def before_test_step(self, **kwargs):
        for p in self.plugins:
            p.before_test_step(self, **kwargs)

    def adapt_test_dataset(self, **kwargs):
        for p in self.plugins:
            p.adapt_test_dataset(self, **kwargs)

    def test_epoch(self, **kwargs):
        for self.mb_it, (self.mb_x, self.mb_y) in \
                enumerate(self.current_dataloader):
            self.before_test_iteration(**kwargs)

            self.mb_x = self.mb_x.to(self.device)
            self.mb_y = self.mb_y.to(self.device)

            self.before_test_forward(**kwargs)
            self.logits = self.model(self.mb_x)
            self.after_test_forward(**kwargs)
            self.loss = self.criterion(self.logits, self.mb_y)

            self.after_test_iteration(**kwargs)

    def after_test_step(self, **kwargs):
        for p in self.plugins:
            p.after_test_step(self, **kwargs)

    def after_test(self, **kwargs):
        for p in self.plugins:
            p.after_test(self, **kwargs)
        # Reset flow-state variables. They should not be used outside the flow
        self.test_step_id = None
        self.step_info = None
        self.current_data = None
        self.current_dataloader = None
        self.mb_it = None
        self.mb_x, self.mb_y = None, None
        self.loss = None
        self.logits = None

    def before_test_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_test_iteration(self, **kwargs)

    def before_test_forward(self, **kwargs):
        for p in self.plugins:
            p.before_test_forward(self, **kwargs)

    def after_test_forward(self, **kwargs):
        for p in self.plugins:
            p.after_test_forward(self, **kwargs)

    def after_test_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_test_iteration(self, **kwargs)


__all__ = ['BaseStrategy']
