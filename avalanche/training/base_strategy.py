from typing import Optional, Sequence

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.benchmarks.scenarios import IStepInfo, DatasetPart
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.evaluation.eval_protocol import EvalProtocol


class BaseStrategy:
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 evaluation_protocol: Optional[EvalProtocol] = None, train_mb_size: int = 1,
                 train_epochs: int = 1, test_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence[StrategyPlugin]] = None):
        """
        BaseStrategy is the super class of all continual learning strategies.
        It implements a basic training loop and callback system that can be
        customized by child strategies. Additionally, it supports plugins,
        a mechanisms to augment existing strategies with additional
        behavior (e.g. a memory buffer for replay).

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param evaluation_protocol: evaluation plugin.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param test_mb_size: mini-batch size for test.
        :param device: PyTorch device to run the model.
        :param plugins: (optional) list of StrategyPlugins.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_epochs = train_epochs
        self.train_mb_size = train_mb_size
        self.test_mb_size = train_mb_size if test_mb_size is None else test_mb_size
        self.device = device

        if evaluation_protocol is None:
            self.evaluation_plugin = EvaluationPlugin(EvalProtocol())
        else:
            self.evaluation_plugin = EvaluationPlugin(evaluation_protocol)
        self.plugins = [] if plugins is None else plugins
        self.plugins.append(self.evaluation_plugin)

        # Flow state variables
        self.step_id = None  # test-flow only.
        self.epoch = None
        self.step_info = None
        self.current_data = None
        self.current_dataloader = None
        self.mb_it = None  # train-flow only. minibatch iteration.
        self.mb_x, self.mb_y = None, None
        self.loss = None
        self.logits = None

    def train(self, step_info: IStepInfo, **kwargs):
        """
        Training loop.

        :param step_info: CL step information.
        :param kwargs: custom arguments.
        :return: train results from the evalution plugin.
        """
        self.step_info = step_info
        self.model.train()
        self.model.to(self.device)

        self.current_data = step_info.current_training_set()[0]
        self.adapt_train_dataset(**kwargs)
        self.make_train_dataloader(**kwargs)

        self.before_training(**kwargs)
        self.epoch = 0
        for self.epoch in range(self.train_epochs):
            self.before_training_epoch(**kwargs)
            self.training_epoch(**kwargs)
            self.after_training_epoch(**kwargs)
        self.after_training(**kwargs)
        return self.evaluation_plugin.get_train_result()

    def test(self, step_info: IStepInfo, test_part: DatasetPart, **kwargs):
        """
        Test the current model on a series of steps, as defined by test_part.

        :param step_info: CL step information.
        :param test_part: determines which steps to test on.
        :param kwargs: custom arguments.
        :return: evaluation plugin test results.
        """
        self._set_initial_test_step_id(step_info, test_part)
        self.step_info = step_info
        self.model.eval()
        self.model.to(self.device)

        self.before_test(**kwargs)
        while self._has_test_steps_left(step_info):
            self.current_data = step_info.step_specific_test_set(self.step_id)[0]
            self.adapt_test_dataset(**kwargs)
            self.make_test_dataloader(**kwargs)

            self.before_test_step(**kwargs)
            self.test_epoch(**kwargs)
            self.after_test_step(**kwargs)

            self.step_id += 1
        self.after_test(**kwargs)
        return self.evaluation_plugin.get_test_result()

    def before_training(self, **kwargs):
        """
        Called  after the dataset and data loader creation and
        before the training loop.
        """
        for p in self.plugins:
            p.before_training(self, **kwargs)

    def make_train_dataloader(self, num_workers=0, **kwargs):
        """
        Called after the dataset instantiation. Initialize the data loader.
        :param num_workers:
        """
        self.current_dataloader = DataLoader(self.current_data,
            num_workers=num_workers, batch_size=self.train_mb_size)

    def _set_initial_test_step_id(self, step_info: IStepInfo,
                                  dataset_part: DatasetPart = None):
        """
        Initialize self.step_id for the test loop.
        :param step_info:
        :param dataset_part:
        :return:
        """
        # TODO: if we remove DatasetPart this may become unnecessary
        self.step_id = -1
        if dataset_part is None:
            dataset_part = DatasetPart.COMPLETE

        if dataset_part == DatasetPart.CURRENT:
            self.step_id = step_info.current_step
        if dataset_part in [DatasetPart.CUMULATIVE, DatasetPart.OLD,
                            DatasetPart.COMPLETE]:
            self.step_id = 0
        if dataset_part == DatasetPart.FUTURE:
            self.step_id = step_info.current_step + 1

        if self.step_id < 0:
            raise ValueError('Invalid dataset part')

    def _has_test_steps_left(self, step_info: IStepInfo,
                             test_part: DatasetPart = None):
        """
        Check if the next CL step must be tested.
        :param step_info:
        :param test_part:
        :return:
        """
        # TODO: if we remove DatasetPart this may become unnecessary
        step_id = self.step_id
        if test_part is None:
            test_part = DatasetPart.COMPLETE

        if test_part == DatasetPart.CURRENT:
            return step_id == step_info.current_step
        if test_part == DatasetPart.CUMULATIVE:
            return step_id <= step_info.current_step
        if test_part == DatasetPart.OLD:
            return step_id < step_info.current_step
        if test_part == DatasetPart.FUTURE:
            return step_info.current_step < step_id < step_info.n_steps
        if test_part == DatasetPart.COMPLETE:
            return step_id < step_info.n_steps

        raise ValueError('Invalid dataset part')

    def make_test_dataloader(self, num_workers=0, **kwargs):
        """
        Initialize the test data loader.
        :param num_workers:
        :param kwargs:
        :return:
        """
        self.current_dataloader = DataLoader(self.current_data,
              num_workers=num_workers, batch_size=self.test_mb_size)

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
        for self.mb_it, (self.mb_x, self.mb_y) in enumerate(self.current_dataloader):
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

    def after_training(self, **kwargs):
        for p in self.plugins:
            p.after_training(self, **kwargs)
        # Reset flow-state variables. They should not be used outside the flow
        self.epoch = None
        self.step_info = None
        self.current_data = None
        self.current_loader = None
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
        for self.mb_it, (self.mb_x, self.mb_y) in enumerate(self.current_dataloader):
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
        self.step_id = None
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
