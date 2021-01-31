################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-11-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

from typing import Optional, Sequence, Union, TYPE_CHECKING

from torch.nn import Module, Linear
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset
import torch
import logging

from avalanche.benchmarks.scenarios import IStepInfo
if TYPE_CHECKING:
    from avalanche.training.plugins import StrategyPlugin


class JointTraining:
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 classifier_field: str = 'classifier',
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None):
        """
        JointStrategy is a super class for all the joint training strategies.
        This means that it is not a continual learning strategy but it can be
        used as an "offline" upper bound for them. This strategy takes in
        input an entire stream and learn from it one shot. It supports unique
        tasks (i.e. streams with a unique task label) or multiple tasks.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param classifier_field: (optional) to specify the name of output layer.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param test_mb_size: mini-batch size for test.
        :param device: PyTorch device to run the model.
        :param plugins: (optional) list of StrategyPlugins.
        """

        # attributes similat to the BaseStrategy
        self.log = logging.getLogger("avalanche")
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_epochs = train_epochs
        self.train_mb_size = train_mb_size
        self.test_mb_size = train_mb_size if test_mb_size is None \
            else test_mb_size
        self.device = device
        self.plugins = [] if plugins is None else plugins

        # attributes specific to the JointTraning
        self.task_to_concat_dataset = {}
        self.current_dataloaders = {}
        self.classifier_field = classifier_field
        self.task_layers = {}

        # Flow state variables
        self.test_step_id = None  # test-flow only.
        self.epoch = None
        self.step_info = None  # we need to keep this for the eval plugin
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

    @torch.no_grad()
    def set_task_layer(self, task_label):
        """
        Sets the correct task layer. Creates a new head for previously
        unseen tasks.

        :param task_label: the task label integer identifying the task.
        :return: None
        """

        # set correct head
        setattr(self.model, self.classifier_field,
                self.task_layers[task_label])
        # this to make sure everything is on the correct device
        self.model.to(self.device)

    @torch.no_grad()
    def create_task_layer(self, n_output_units: int, previous_task_layer=None):
        """
        Creates a new task layer.

        By default, this method will create a new :class:`Linear` layer with
        n_output_units" output units. If  "previous_task_layer" is None,
        the name of the classifier field is used to retrieve the amount of
        input features.

        This method will also be used to create a new layer when expanding
        an existing task head.

        This method can be overridden by the user so that a layer different
        from :class:`Linear` can be created.

        :param n_output_units: The number of output units.
        :param previous_task_layer: If not None, the previously created layer
             for the same task.
        :return: The new layer.
        """
        if previous_task_layer is None:
            current_task_layer: Linear = getattr(self.model,
                                                 self.classifier_field)
            in_features = current_task_layer.in_features
            has_bias = current_task_layer.bias is not None
        else:
            in_features = previous_task_layer.in_features
            has_bias = previous_task_layer.bias is not None

        new_layer = Linear(in_features, n_output_units, bias=has_bias)
        return new_layer

    def train(self, step_infos: Sequence[IStepInfo], **kwargs):
        """ Training loop. it trains only on a sequence of steps (a stream).
        WARNING: Please take in mind that it trains on it "in parallel" not
        iteratively as in the BaseStrategy train method. This is the main
        difference from the JointTraining and BaseStrategy classes.

        :param step_infos: sequence of IStepInfo (a stream).
        :return:
        """
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        if isinstance(step_infos, IStepInfo):
            step_infos = [step_infos]

        self.before_training(**kwargs)

        # here we concat steps of the same tasks
        task_labels = []
        task_to_datasets = {}
        for step_info in step_infos:
            self.train_task_label = step_info.task_label
            task_labels.append(step_info.task_label)
            if step_info.task_label not in task_to_datasets.keys():
                task_to_datasets[step_info.task_label] = []
            task_to_datasets[step_info.task_label].append(step_info.dataset)

        for t in task_to_datasets.keys():
            self.task_to_concat_dataset[t] = ConcatDataset(task_to_datasets[t])

        # This is useful to the eval plugin
        self.step_info = step_infos[0]

        # create a different head for each task
        for t, concat_ds in self.task_to_concat_dataset.items():
            szs = []
            for ds in concat_ds.datasets:
                szs.append(max(ds.targets) + 1)
            self.task_layers[t] = self.create_task_layer(max(szs))
            # self.add_new_params_to_optimizer(self.task_layers[t].parameters())

        self.log.info("starting training...")
        self.model.train()
        self.model.to(self.device)

        self.adapt_train_dataset(**kwargs)
        self.make_train_dataloader(**kwargs)

        self.before_training_step(joint_training=True, **kwargs)
        self.epoch = 0
        for self.epoch in range(self.train_epochs):
            self.before_training_epoch(**kwargs)
            self.training_epoch(**kwargs)
            self.after_training_epoch(**kwargs)
        self.after_training_step(**kwargs)

        self.after_training(**kwargs)

    def make_train_dataloader(self, num_workers=0, shuffle=True, **kwargs):
        """
        Called after the dataset instantiation. Initialize the data loader.
        :param num_workers: number of thread workers for the data laoding.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """

        # We keep a separate data loader for each task.
        # Please note that "current_dataloaders" and "current_dataloader" are
        # two different attributes of the class
        for t in self.task_to_concat_dataset.keys():
            self.current_data = self.task_to_concat_dataset[t]
            self.current_dataloaders[t] = DataLoader(
                self.current_data, num_workers=num_workers,
                batch_size=self.train_mb_size, shuffle=shuffle
            )

    def training_epoch(self, **kwargs):
        """
        Training epoch. How does it work:
        1. From each the n data loader (one for task) we load a mini-batch.
        2. net forward with the right head accumlating gradients for all the n
        mini-batches.
        3. Update the gradient.
        1., 2. and 3. are repeated till all the data of each of the data
        loader have been processed at least once. If a data loader finishes
        his mini-batches, it starts again from the first mini-batch. This
        assumes that each task as roughly the same amount of data.

        :param kwargs: named arguments eventually passed to this method.
        :return: None.
        """

        # compute the max num of iterations to cover one epoch.
        iter_dataloaders = dict()
        max_size = 0
        for t in self.task_to_concat_dataset.keys():
            iter_dataloaders[t] = iter(self.current_dataloaders[t])
            sz = len(self.task_to_concat_dataset[t])
            if sz > max_size:
                max_size = sz

        n_iters = max_size // self.train_mb_size

        for it in range(n_iters):
            self.mb_it = it
            self.before_training_iteration(**kwargs)
            self.optimizer.zero_grad()

            # we accumulate the gradients across tasks
            for t in self.task_to_concat_dataset.keys():
                self.current_dataloader = iter_dataloaders[t]
                self.set_task_layer(t)

                try:
                    self.mb_x, self.mb_y = next(self.current_dataloader)
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    iter_dataloaders[t] = iter(self.current_dataloader)
                    self.current_dataloader = iter_dataloaders[t]
                    self.mb_x, self.mb_y = next(self.current_dataloader)

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

    def add_new_params_to_optimizer(self, new_params):
        """ Add new parameters to the trainable parameters.

        :param new_params: list of trainable parameters
        """
        self.optimizer.add_param_group({'params': new_params})

    def test(self, step_list: Union[IStepInfo, Sequence[IStepInfo]], **kwargs):
        """
        Test the current model on a series of steps, as defined by test_part.

        :param step_info: CL step information.
        :param test_part: determines which steps to test on.
        :param kwargs: custom arguments.
        :return: evaluation plugin test results.
        """
        self.is_training = False
        self.model.eval()
        self.model.to(self.device)

        if isinstance(step_list, IStepInfo):
            step_list = [step_list]

        self.before_test(**kwargs)
        for step_info in step_list:
            self.test_task_label = step_info.task_label
            self.step_info = step_info
            self.test_step_id = step_info.current_step

            self.current_data = step_info.dataset
            self.adapt_test_dataset(**kwargs)
            self.make_test_dataloader(**kwargs)

            self.before_test_step(**kwargs)
            self.test_epoch(**kwargs)
            self.after_test_step(**kwargs)

        self.after_test(**kwargs)

    def before_training_step(self, **kwargs):
        """
        Called  after the dataset and data loader creation and
        before the training loop.
        """
        for p in self.plugins:
            p.before_training_step(self, **kwargs)

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
        # Reset flow-state variables. They should not be used outside the flow
        self.test_step_id = None
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

        self.set_task_layer(self.step_info.task_label)

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


__all__ = ['JointTraining']
