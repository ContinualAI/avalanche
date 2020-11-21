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

from typing import Optional, Sequence

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset

from avalanche.benchmarks.scenarios import IStepInfo
from avalanche.training.plugins import StrategyPlugin
from avalanche.evaluation.eval_protocol import EvalProtocol
from avalanche.training.strategies import BaseStrategy


class JointTraining(BaseStrategy):
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 evaluation_protocol: Optional[EvalProtocol] = None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence[StrategyPlugin]] = None):
        """
        JointStrategy is a super class for all the joint training strategies.
        This means that it is not a continual learning strategy but it can be
        used as an "offline" upper bound for them. This strategy takes in
        input an entire stream and learn from it one shot. It supports unique
        tasks (i.e. streams with a unique task label) or multiple tasks.

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

        super().__init__(
            model, optimizer, criterion, evaluation_protocol,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins)

    def train(self, step_infos: Sequence[IStepInfo], **kwargs):
        """ Training loop. if step_infos is a single element trains on it.
        If it is a sequence, trains the model on each step in order.
        This is different from joint training on the entire stream.

        :param step_infos: single IStepInfo or sequence.
        :return:
        """
        self.model.train()
        self.model.to(self.device)

        if isinstance(step_infos, IStepInfo):
            step_infos = [step_infos]

        task_labels = []
        datasets = []
        for step_info in step_infos:
            task_labels.append(step_info.task_label)
            datasets.append(step_info.dataset)

        if len(set(task_labels)) == 1:
            # this means it's only one task and we can train on the concat
            # of the datasets.
            self.current_data = ConcatDataset(datasets)
            self.step_info = step_infos[0]
        else:
            raise NotImplementedError

        print("starting training...")
        self.model.train()
        self.model.to(self.device)

        self.adapt_train_dataset(**kwargs)
        self.make_train_dataloader(**kwargs)

        self.before_training_step(**kwargs)
        self.epoch = 0
        for self.epoch in range(self.train_epochs):
            self.before_training_epoch(**kwargs)
            self.training_epoch(**kwargs)
            self.after_training_epoch(**kwargs)
        self.after_training_step(**kwargs)
        return self.evaluation_plugin.get_train_result()


__all__ = ['JointTraining']
